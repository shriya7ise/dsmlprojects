import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import json
import feedparser
import hashlib
from urllib.parse import urlparse
from textblob import TextBlob
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import schedule
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class NewsArticle:
    company: str
    title: str
    summary: str
    url: str
    source: str
    publish_date: str
    content: str = ""
    sentiment: str = ""
    method: str = ""
    scraped_at: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__

    def get_hash(self) -> str:
        content_for_hash = f"{self.title.lower()}{self.url}{self.company.lower()}"
        return hashlib.md5(content_for_hash.encode()).hexdigest()

class IPONewsAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.articles: List[NewsArticle] = []
        self.seen_hashes: Set[str] = set()
        self.memory_file = config.get('memory_file', 'agent_memory.json')
        self.output_file = config.get('output_file', 'ipo_news.csv')
        self.use_selenium = config.get('use_selenium', False)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ipo_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.driver = None
        if self.use_selenium:
            self._setup_selenium()

        self._load_existing_hashes()

        self.methods = {
            'newsapi': self.method1_newsapi,
            'rss': self.method2_rss_feeds,
            'fallback': self.method3_fallback,
            'selenium': self.method4_selenium
        }

        self.memory = self._load_memory()

        # Enhanced RSS feeds for IPO news
        self.rss_feeds = [
            {'url': 'https://economictimes.indiatimes.com/markets/ipo/rssfeeds/13357785.cms', 'name': 'Economic Times IPO'},
            {'url': 'https://www.moneycontrol.com/rss/ipo.xml', 'name': 'MoneyControl IPO'},
            {'url': 'https://feeds.bloomberg.com/markets/news.rss', 'name': 'Bloomberg Markets'},
            {'url': 'https://feeds.reuters.com/reuters/businessNews', 'name': 'Reuters Business'}
        ]

    def select_methods(self) -> List[str]:
        """Select methods based on performance and preferences"""
        preferred = self.memory['user_preferences']['preferred_methods']
        if preferred:
            return preferred
        
        # Auto-select based on performance
        available_methods = []
        perf = self.memory['method_performance']
        
        for method, stats in perf.items():
            success_rate = stats['success'] / max(1, stats['success'] + stats['failures'])
            if success_rate > 0.5 or stats['success'] + stats['failures'] == 0:
                available_methods.append(method)
        
        return available_methods or ['rss', 'selenium', 'fallback']

    def save_results(self, df: pd.DataFrame) -> Optional[str]:
        """Save results to CSV file"""
        try:
            if df.empty:
                self.logger.warning("No data to save")
                return None
            
            # Check if output file exists
            if os.path.exists(self.output_file):
                # Load existing data to avoid duplicates
                existing_df = pd.read_csv(self.output_file)
                
                # Create hash column for comparison if it doesn't exist
                if 'article_hash' not in existing_df.columns:
                    existing_df['article_hash'] = existing_df.apply(
                        lambda row: hashlib.md5(f"{row['title'].lower()}{row['url']}{row['company'].lower()}".encode()).hexdigest(),
                        axis=1
                    )
                
                # Add hash to new data
                df['article_hash'] = df.apply(
                    lambda row: hashlib.md5(f"{row['title'].lower()}{row['url']}{row['company'].lower()}".encode()).hexdigest(),
                    axis=1
                )
                
                # Filter out duplicates
                new_articles = df[~df['article_hash'].isin(existing_df['article_hash'])]
                
                if not new_articles.empty:
                    # Append new articles
                    combined_df = pd.concat([existing_df, new_articles], ignore_index=True)
                    combined_df.to_csv(self.output_file, index=False)
                    self.logger.info(f"Added {len(new_articles)} new articles to {self.output_file}")
                    return self.output_file
                else:
                    self.logger.info("No new articles to add (all duplicates)")
                    return None
            else:
                # Create new file
                df['article_hash'] = df.apply(
                    lambda row: hashlib.md5(f"{row['title'].lower()}{row['url']}{row['company'].lower()}".encode()).hexdigest(),
                    axis=1
                )
                df.to_csv(self.output_file, index=False)
                self.logger.info(f"Created new file {self.output_file} with {len(df)} articles")
                return self.output_file
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return None

    def scrape(self, companies: List[str], methods: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute scraping"""
        self.logger.info(f"🔍 Starting scrape for {len(companies)} companies: {', '.join(companies)}")
        methods = methods or self.select_methods()
        self.logger.info(f"Using methods: {', '.join(methods)}")
        
        articles_added = 0
        self.articles = []  # Reset articles for this session

        for method_name in methods:
            self.logger.info(f"📡 Executing method: {method_name}")
            try:
                count = self.methods[method_name](companies)
                articles_added += count
                self.logger.info(f"✅ Method {method_name} added {count} articles")
            except Exception as e:
                self.logger.error(f"❌ Method {method_name} failed: {e}")

        if self.articles:
            df = pd.DataFrame([article.to_dict() for article in self.articles])
            df['scraping_session'] = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.memory['last_run'] = datetime.now().isoformat()
            self._save_memory()
            self.logger.info(f"🎉 Total articles collected: {len(df)}")
            return df
        
        self.logger.warning("No articles found")
        return pd.DataFrame()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # Execute script to hide webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("✅ Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Selenium WebDriver: {e}")
            self.use_selenium = False

    def _cleanup_selenium(self):
        """Cleanup Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("🔧 Selenium WebDriver closed")
            except Exception as e:
                self.logger.error(f"Failed to close Selenium WebDriver: {e}")
            self.driver = None

    def _load_existing_hashes(self):
        """Load existing article hashes to prevent duplicates"""
        try:
            if os.path.exists(self.output_file):
                df = pd.read_csv(self.output_file)
                for _, row in df.iterrows():
                    article_hash = hashlib.md5(f"{row['title'].lower()}{row['url']}{row['company'].lower()}".encode()).hexdigest()
                    self.seen_hashes.add(article_hash)
                self.logger.info(f"Loaded {len(self.seen_hashes)} existing article hashes")
        except Exception as e:
            self.logger.warning(f"Failed to load existing hashes from CSV: {e}")

    def _load_memory(self) -> Dict:
        """Load agent memory"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                    self.logger.info("📚 Loaded agent memory")
                    return memory
        except Exception as e:
            self.logger.warning(f"Failed to load memory: {e}")
        
        default_memory = {
            'method_performance': {method: {'success': 0, 'failures': 0} for method in self.methods},
            'user_preferences': {'preferred_methods': [], 'output_format': 'csv'},
            'last_run': None,
            'total_articles_collected': 0
        }
        self.logger.info("📚 Created new agent memory")
        return default_memory

    def _save_memory(self):
        """Save agent memory"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        try:
            polarity = TextBlob(text).sentiment.polarity
            return "POSITIVE" if polarity > 0.1 else "NEGATIVE" if polarity < -0.1 else "NEUTRAL"
        except:
            return "NEUTRAL"

    def fetch_full_article(self, url: str) -> str:
        """Fetch full article content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'noscript', 'nav', 'header', 'footer']):
                tag.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all(['p', 'div'], class_=lambda x: x and 'content' in x.lower() if x else False)
            if not paragraphs:
                paragraphs = soup.find_all('p')
            
            text = ' '.join(p.get_text(strip=True) for p in paragraphs[:10])  # Limit to first 10 paragraphs
            return text[:2000] if len(text) > 100 else "Full content not available."
        except Exception as e:
            self.logger.debug(f"Failed to fetch article content from {url}: {e}")
            return "Full content not available."

    def add_article(self, article: NewsArticle) -> bool:
        """Add article if not duplicate"""
        h = article.get_hash()
        if h not in self.seen_hashes:
            article.sentiment = self.analyze_sentiment(article.summary or article.title)
            article.scraped_at = datetime.now().isoformat()
            self.articles.append(article)
            self.seen_hashes.add(h)
            self.logger.debug(f"Added article: {article.title[:50]}...")
            return True
        else:
            self.logger.debug(f"Duplicate article skipped: {article.title[:50]}...")
            return False

    def method1_newsapi(self, companies: List[str]) -> int:
        """Scrape using NewsAPI"""
        if not self.config.get('newsapi_key'):
            self.logger.warning("NewsAPI key missing")
            self.memory['method_performance']['newsapi']['failures'] += 1
            self._save_memory()
            return 0
        
        articles_added = 0
        base_url = 'https://newsapi.org/v2/everything'
        
        try:
            for company in companies:
                params = {
                    'q': f'"{company}" IPO OR "{company}" "initial public offering"',
                    'apiKey': self.config['newsapi_key'],
                    'pageSize': 5,
                    'sortBy': 'publishedAt',
                    'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'language': 'en'
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for article_data in data.get('articles', []):
                    if not article_data.get('title') or not article_data.get('url'):
                        continue
                        
                    url = article_data.get('url', '')
                    content = article_data.get('content', '') or self.fetch_full_article(url)
                    
                    article = NewsArticle(
                        company=company,
                        title=article_data.get('title', ''),
                        summary=article_data.get('description', ''),
                        content=content,
                        url=url,
                        source=article_data.get('source', {}).get('name', 'NewsAPI'),
                        publish_date=article_data.get('publishedAt', ''),
                        method='NewsAPI'
                    )
                    if self.add_article(article):
                        articles_added += 1
                time.sleep(1)  # Rate limiting
                
            self.memory['method_performance']['newsapi']['success'] += articles_added
            self._save_memory()
        except Exception as e:
            self.logger.error(f"NewsAPI error: {e}")
            self.memory['method_performance']['newsapi']['failures'] += 1
            self._save_memory()
        return articles_added
    
    def method2_rss_feeds(self, companies: List[str]) -> int:
        """Scrape RSS feeds"""
        articles_added = 0
        
        try:
            for feed in self.rss_feeds:
                self.logger.info(f"Processing RSS feed: {feed['name']}")
                try:
                    feed_data = feedparser.parse(feed['url'])
                    
                    for entry in feed_data.entries[:10]:  # Limit to 10 entries per feed
                        title = entry.get('title', '')
                        summary = entry.get('summary', entry.get('description', ''))
                        url = entry.get('link', '')
                        
                        # Check if any company is mentioned or if it's IPO related
                        content_text = f"{title} {summary}".lower()
                        
                        for company in companies:
                            company_mentioned = company.lower() in content_text
                            ipo_mentioned = any(keyword in content_text for keyword in ['ipo', 'initial public offering', 'public listing', 'share sale'])
                            
                            if company_mentioned or ipo_mentioned:
                                full_content = self.fetch_full_article(url) if url else summary
                                
                                article = NewsArticle(
                                    company=company if company_mentioned else "IPO Related",
                                    title=title,
                                    summary=summary,
                                    content=full_content,
                                    url=url,
                                    source=feed['name'],
                                    publish_date=entry.get('published', ''),
                                    method='RSS'
                                )
                                if self.add_article(article):
                                    articles_added += 1
                                break  # Avoid duplicate entries for multiple companies
                    
                    time.sleep(2)  # Rate limiting between feeds
                    
                except Exception as feed_error:
                    self.logger.error(f"Error processing RSS feed {feed['name']}: {feed_error}")
                    continue
            
            self.memory['method_performance']['rss']['success'] += articles_added
            self._save_memory()
        except Exception as e:
            self.logger.error(f"RSS method error: {e}")
            self.memory['method_performance']['rss']['failures'] += 1
            self._save_memory()
        
        return articles_added
    
    def method3_fallback(self, companies: List[str]) -> int:
        """Fallback method - creates placeholder articles"""
        articles_added = 0
        
        try:
            for company in companies:
                # Create a meaningful fallback article
                current_date = datetime.now().strftime('%Y-%m-%d')
                
                article = NewsArticle(
                    company=company,
                    title=f"{company} IPO Update - {current_date}",
                    summary=f"Latest information about {company} IPO status and market developments. This is a fallback entry when other sources are unavailable.",
                    content=f"No detailed content available for {company} IPO at this time. Please check financial news sources for the latest updates.",
                    url=f"https://example.com/{company.lower().replace(' ', '-')}-ipo",
                    source='Fallback System',
                    publish_date=current_date,
                    method='Fallback'
                )
                if self.add_article(article):
                    articles_added += 1
            
            self.memory['method_performance']['fallback']['success'] += articles_added
            self._save_memory()
        except Exception as e:
            self.logger.error(f"Fallback method error: {e}")
            self.memory['method_performance']['fallback']['failures'] += 1
            self._save_memory()
        
        return articles_added

    def method4_selenium(self, companies: List[str]) -> int:
        """Enhanced Selenium method with multiple sources"""
        if not self.driver:
            self._setup_selenium()
        if not self.driver:
            self.logger.warning("Selenium not available")
            return 0

        articles_added = 0
        search_sources = [
            "https://www.google.com/search?q={query}+IPO+news&tbm=nws",
            "https://www.google.com/search?q={query}+initial+public+offering",
        ]
        
        try:
            for company in companies:
                self.logger.info(f"Selenium search for: {company}")
                
                for source_template in search_sources:
                    try:
                        query = company.replace(' ', '+')
                        search_url = source_template.format(query=query)
                        
                        self.logger.info(f"Fetching: {search_url}")
                        self.driver.get(search_url)
                        
                        # Wait for results to load
                        WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.g, div[data-sokoban-container]'))
                        )
                        
                        # Parse results
                        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                        
                        # Look for news results
                        results = soup.select('div.g, div[data-sokoban-container]')[:5]  # Limit to 5 results
                        
                        for result in results:
                            try:
                                # Extract title
                                title_tag = result.select_one('h3, [role="heading"]')
                                if not title_tag:
                                    continue
                                title = title_tag.get_text(strip=True)
                                
                                # Extract link
                                link_tag = result.select_one('a[href^="http"]')
                                if not link_tag:
                                    continue
                                url = link_tag.get('href', '')
                                
                                # Extract description
                                desc_selectors = [
                                    'div[data-sncf="2"]',
                                    'span[data-ved]',
                                    '.st',
                                    'div.s'
                                ]
                                description = ""
                                for selector in desc_selectors:
                                    desc_tag = result.select_one(selector)
                                    if desc_tag:
                                        description = desc_tag.get_text(strip=True)
                                        break
                                
                                # Skip if essential info is missing
                                if not title or not url or 'google.com' in url:
                                    continue
                                
                                # Fetch full content
                                full_content = self.fetch_full_article(url)
                                
                                article = NewsArticle(
                                    company=company,
                                    title=title,
                                    summary=description,
                                    url=url,
                                    content=full_content,
                                    source='Google News (Selenium)',
                                    publish_date=datetime.now().strftime('%Y-%m-%d'),
                                    method='Selenium'
                                )
                                
                                if self.add_article(article):
                                    articles_added += 1
                                    
                            except Exception as result_error:
                                self.logger.debug(f"Error processing search result: {result_error}")
                                continue
                        
                        time.sleep(3)  # Delay between searches
                        
                    except Exception as search_error:
                        self.logger.error(f"Error with search source {source_template}: {search_error}")
                        continue
                
                time.sleep(2)  # Delay between companies
            
            self.memory['method_performance']['selenium']['success'] += articles_added
            self._save_memory()
            
        except Exception as e:
            self.logger.error(f"Selenium method error: {e}")
            self.memory['method_performance']['selenium']['failures'] += 1
            self._save_memory()
        
        return articles_added

    def process_command(self, command: str) -> str:
        """Process user input command"""
        command_lower = command.lower().strip()
        
        if 'help' in command_lower or '?' in command_lower:
            return (
                "🤖 IPO News Agent Commands:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📊 scan [company1, company2, ...] - Scan for IPO news\n"
                "⚙️  set method [newsapi,rss,fallback,selenium] - Set preferred methods\n"
                "📈 status - Show agent status and performance\n"
                "📋 show data - Display recent articles\n"
                "🧹 clear data - Clear all stored articles\n"
                "🚪 exit - Exit agent\n"
                "\nExample: scan [Apple, Microsoft, Tesla]"
            )
        
        elif 'scan' in command_lower:
            try:
                # Extract companies from command
                if '[' in command and ']' in command:
                    companies_str = command.split('[')[1].split(']')[0]
                    companies = [c.strip() for c in companies_str.split(',') if c.strip()]
                else:
                    return "❌ Invalid format. Use: scan [Company A, Company B]\nExample: scan [Apple, Microsoft]"
                
                if companies:
                    print(f"🚀 Starting scan for {len(companies)} companies...")
                    df = self.scrape(companies)
                    
                    if not df.empty:
                        filename = self.save_results(df)
                        if filename:
                            # Update memory
                            self.memory['total_articles_collected'] = self.memory.get('total_articles_collected', 0) + len(df)
                            self._save_memory()
                            
                            # Create summary
                            summary = (
                                f"✅ Successfully collected {len(df)} articles!\n"
                                f"📁 Saved to: {filename}\n"
                                f"🏢 Companies: {', '.join(companies)}\n"
                                f"🕒 Session: {df['scraping_session'].iloc[0] if not df.empty else 'N/A'}\n\n"
                                f"📋 Sample Articles:\n"
                                f"{'─' * 50}\n"
                            )
                            
                            # Add sample articles
                            for idx, (_, row) in enumerate(df.head(3).iterrows()):
                                summary += f"{idx + 1}. 🏢 {row['company']}\n"
                                summary += f"   📰 {row['title'][:80]}{'...' if len(row['title']) > 80 else ''}\n"
                                summary += f"   📊 Sentiment: {row['sentiment']} | Method: {row['method']}\n\n"
                            
                            return summary
                        else:
                            return "📊 Scan completed but no new articles were found (all duplicates)"
                    else:
                        return "❌ No articles found. Try different companies or check your internet connection."
                else:
                    return "❌ Please specify companies. Example: scan [Apple, Microsoft, Tesla]"
                    
            except Exception as e:
                self.logger.error(f"Command processing error: {e}")
                return f"❌ Error processing command: {str(e)}"
        
        elif 'set method' in command_lower:
            try:
                if '[' in command and ']' in command:
                    methods_str = command.split('[')[1].split(']')[0]
                    methods = [m.strip() for m in methods_str.split(',') if m.strip() in self.methods]
                else:
                    return "❌ Invalid format. Use: set method [newsapi,rss,selenium]\nAvailable: newsapi, rss, fallback, selenium"
                
                if methods:
                    self.memory['user_preferences']['preferred_methods'] = methods
                    self._save_memory()
                    return f"✅ Preferred methods set to: {', '.join(methods)}"
                else:
                    return f"❌ Invalid methods. Available: {', '.join(self.methods.keys())}"
            except Exception as e:
                return f"❌ Error setting methods: {str(e)}"
        
        elif 'status' in command_lower:
            perf = self.memory['method_performance']
            status_lines = [
                "🤖 IPO News Agent Status",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🕒 Last run: {self.memory.get('last_run', 'Never')}",
                f"📊 Articles this session: {len(self.articles)}",
                f"🗃️  Total unique articles: {len(self.seen_hashes)}",
                f"📈 Total collected: {self.memory.get('total_articles_collected', 0)}",
                f"⚙️  Preferred methods: {', '.join(self.memory['user_preferences']['preferred_methods']) or 'Auto-select'}",
                f"🤖 Selenium enabled: {'✅ Yes' if self.use_selenium else '❌ No'}",
                "",
                "📊 Method Performance:",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ]
            
            for method, stats in perf.items():
                total = stats['success'] + stats['failures']
                success_rate = (stats['success'] / max(1, total)) * 100
                status_lines.append(f"{method:10} | ✅ {stats['success']:3d} | ❌ {stats['failures']:3d} | Success: {success_rate:5.1f}%")
            
            return "\n".join(status_lines)
        
        elif 'show data' in command_lower:
            try:
                if os.path.exists(self.output_file):
                    df = pd.read_csv(self.output_file)
                    if not df.empty:
                        recent = df.tail(10)  # Show last 10 articles
                        display_text = f"📋 Recent Articles ({len(recent)} of {len(df)} total):\n{'─' * 60}\n"
                        
                        for idx, (_, row) in enumerate(recent.iterrows(), 1):
                            display_text += f"{idx:2d}. 🏢 {row['company']}\n"
                            display_text += f"    📰 {row['title'][:70]}{'...' if len(str(row['title'])) > 70 else ''}\n"
                            display_text += f"    📊 {row['sentiment']} | 🔗 {row['source']} | 📅 {row['publish_date'][:10]}\n\n"
                        
                        return display_text
                    else:
                        return "📭 No articles found in database"
                else:
                    return "📭 No data file found. Run a scan first!"
            except Exception as e:
                return f"❌ Error displaying data: {str(e)}"
        
        elif 'clear data' in command_lower:
            try:
                confirm = input("⚠️ Are you sure you want to clear all data? (yes/no): ").lower()
                if confirm in ['yes', 'y']:
                    if os.path.exists(self.output_file):
                        os.remove(self.output_file)
                    if os.path.exists(self.memory_file):
                        os.remove(self.memory_file)
                    self.seen_hashes.clear()
                    self.articles.clear()
                    self.memory = self._load_memory()  # Reset memory
                    return "🧹 All data cleared successfully!"
                else:
                    return "❌ Operation cancelled"
            except Exception as e:
                return f"❌ Error clearing data: {str(e)}"
        
        elif 'exit' in command_lower or 'quit' in command_lower:
            self._cleanup_selenium()
            return "exit"
        
        else:
            return (
                "❓ Unknown command. Available commands:\n"
                "• help - Show all commands\n" 
                "• scan [companies] - Scan for IPO news\n"
                "• status - Show agent status\n"
                "• show data - Display recent articles\n"
                "• exit - Exit agent"
            )

    def run(self):
        """Main agent loop"""
        print("🚀 IPO News Agent v2.0 Started!")
        print("=" * 50)
        print("🤖 Your AI-powered IPO news collection assistant")
        print("📊 Type 'help' to see available commands")
        print("💡 Example: scan [Apple, Microsoft, Tesla]")
        print("=" * 50)
        
        try:
            while True:
                try:
                    command = input("\n🤖 IPO Agent > ").strip()
                    if not command:
                        continue
                        
                    print()  # Add spacing
                    response = self.process_command(command)
                    print(response)
                    
                    if response == "exit":
                        print("\n👋 Thank you for using IPO News Agent!")
                        break
                        
                    # Run any pending scheduled tasks
                    schedule.run_pending()
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️ Interrupted by user")
                    confirm = input("Exit agent? (y/n): ").lower()
                    if confirm in ['y', 'yes']:
                        break
                except Exception as e:
                    self.logger.error(f"Command loop error: {e}")
                    print(f"❌ Error: {str(e)}")
                    
        finally:
            self._cleanup_selenium()
            print("🔧 Cleanup completed. Goodbye!")

def main():
    """Main function to run the IPO News Agent"""
    
    # Configuration - Update these with your API keys
    config = {
        'newsapi_key': '89d1452cc28b48f091975f77b1a28e90',  # Your NewsAPI key
        'gnews_key': 'aa6852910b0bb6186ea46ed339ccaa63',    # Optional: GNews API key
        'memory_file': 'agent_memory.json',
        'output_file': 'ipo_news.csv',
        'use_selenium': True  # Set to False if you don't want to use Selenium
    }
    
    # Validate API key
    if not config['newsapi_key'] or config['newsapi_key'] == 'your_newsapi_key_here':
        print("⚠️ Warning: NewsAPI key not configured. Some features may be limited.")
        print("💡 Get your free API key from: https://newsapi.org/")
        print("📝 Update the 'newsapi_key' in the config section")
        print()
    
    try:
        # Create and run agent
        agent = IPONewsAgent(config)
        agent.run()
        
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")
        print("💡 Make sure you have installed all required packages:")
        print("   pip install requests pandas beautifulsoup4 textblob feedparser selenium webdriver-manager schedule")

# Demo function for testing
def demo_run():
    """Demo function to test the agent"""
    print("🧪 Running IPO News Agent Demo")
    print("=" * 40)
    
    config = {
        'newsapi_key': '89d1452cc28b48f091975f77b1a28e90',
        'memory_file': 'demo_memory.json',
        'output_file': 'demo_ipo_news.csv',
        'use_selenium': True
    }
    
    agent = IPONewsAgent(config)
    
    # Demo companies
    demo_companies = ['Apple', 'Microsoft', 'Tesla']
    
    print(f"🔍 Scanning for: {', '.join(demo_companies)}")
    
    try:
        # Run the scrape
        df = agent.scrape(demo_companies)
        
        if not df.empty:
            # Save results
            filename = agent.save_results(df)
            
            print(f"\n✅ Demo completed successfully!")
            print(f"📊 Found {len(df)} articles")
            print(f"📁 Saved to: {filename}")
            
            # Show sample
            print(f"\n📋 Sample Results:")
            print("─" * 50)
            for idx, (_, row) in enumerate(df.head(3).iterrows(), 1):
                print(f"{idx}. {row['company']} - {row['title'][:60]}...")
                print(f"   Source: {row['source']} | Sentiment: {row['sentiment']}")
                print()
        else:
            print("❌ No articles found in demo")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        agent._cleanup_selenium()

if __name__ == "__main__":
    import sys
    
    # Check if demo mode requested
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_run()
    else:
        main()