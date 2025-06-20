import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
import time
import os
import re
from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import quote_plus, urljoin, urlparse
import hashlib
import requests
from bs4 import BeautifulSoup
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class CompanyNewsScraperAgent:
    def __init__(self, ipo_csv_path: str, news_csv_path: str = "company_news_database.csv"):
        """
        Initialize the Company News Scraper
        
        Args:
            ipo_csv_path: Path to your IPO CSV file
            news_csv_path: Path to save/load news database
        """
        self.ipo_csv_path = ipo_csv_path
        self.news_csv_path = news_csv_path
        self.driver = None
        self.session = requests.Session()
        self.companies_db = self.load_companies()
        self.news_db = self.load_news_database()
        
        # Enhanced news sources with multiple search strategies
        self.news_sources = {
            'moneycontrol': {
                'base_url': 'https://www.moneycontrol.com',
                'search_strategies': [
                    'https://www.moneycontrol.com/news/tags/{company}.html',
                    'https://www.moneycontrol.com/india/stockpricequote/{company}',
                    'https://www.google.com/search?q=site:moneycontrol.com {company} news'
                ],
                'selectors': {
                    'articles': [
                        'a[href*="/news/business"]',
                        'a[href*="/news/company"]', 
                        '.news_links a',
                        'h2 a'
                    ],
                    'title': ['h1', '.artTitle', '.news_title', 'h2'],
                    'date': ['time', '.publish_on', '.date', '.news_date'],
                    'content': ['article p', '.content p', '.arti-content p', '.news_content p'],
                    'summary': ['meta[name="description"]', '.summary', '.excerpt']
                }
            },
            'livemint': {
                'base_url': 'https://www.livemint.com',
                'search_strategies': [
                    'https://www.livemint.com/search?q={company}',
                    'https://www.google.com/search?q=site:livemint.com {company}'
                ],
                'selectors': {
                    'articles': [
                        'a[href*="/companies"]', 
                        'a[href*="/market"]',
                        'a[href*="/news"]',
                        '.story a'
                    ],
                    'title': ['h1', '.headline', '.story-headline'],
                    'date': ['time', '.publish-date', '.story-date'],
                    'content': ['article p', '.story-content p', '.paywall p'],
                    'summary': ['meta[name="description"]', '.summary']
                }
            },
            'economictimes': {
                'base_url': 'https://economictimes.indiatimes.com',
                'search_strategies': [
                    'https://economictimes.indiatimes.com/topic/{company}',
                    'https://economictimes.indiatimes.com/markets/stocks/news?q={company}',
                    'https://www.google.com/search?q=site:economictimes.indiatimes.com {company}'
                ],
                'selectors': {
                    'articles': [
                        'a[href*="/news/"]',
                        'a[href*="/markets/"]',
                        '.eachStory a'
                    ],
                    'title': ['h1', '.story-headline', 'h2'],
                    'date': ['time', '.publish-date', '.date'],
                    'content': ['article p', '.story-content p', 'p'],
                    'summary': ['meta[name="description"]']
                }
            },
            'business_standard': {
                'base_url': 'https://www.business-standard.com',
                'search_strategies': [
                    'https://www.business-standard.com/search?q={company}',
                    'https://www.google.com/search?q=site:business-standard.com {company}'
                ],
                'selectors': {
                    'articles': [
                        'a[href*="/article/"]',
                        'a[href*="/companies/"]',
                        '.listing-page a'
                    ],
                    'title': ['h1', '.headline'],
                    'date': ['time', '.date'],
                    'content': ['article p', '.article-content p'],
                    'summary': ['meta[name="description"]']
                }
            }
        }
        
        # Request headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        self.session.headers.update(self.headers)
        
        # Initialize Enhanced NLP
        self.nlp = EnhancedNLP()
        
    def load_companies(self) -> pd.DataFrame:
        """Load IPO companies from CSV with flexible column detection"""
        try:
            df = pd.read_csv(self.ipo_csv_path)
            
            # Detect company name column
            possible_name_columns = ['Company Name', 'Company', 'Name', 'company_name', 'COMPANY']
            company_col = None
            
            for col in possible_name_columns:
                if col in df.columns:
                    company_col = col
                    break
            
            if company_col:
                # Clean company names
                df[company_col] = df[company_col].astype(str).str.strip()
                df = df[df[company_col].notna() & (df[company_col] != '')]
                logging.info(f"Loaded {len(df)} companies from {self.ipo_csv_path} using column '{company_col}'")
            else:
                logging.warning(f"No company name column found. Available columns: {list(df.columns)}")
            
            return df
        except Exception as e:
            logging.error(f"Error loading companies CSV: {e}")
            return pd.DataFrame()
    
    def load_news_database(self) -> pd.DataFrame:
        """Load existing news database or create new one"""
        if os.path.exists(self.news_csv_path):
            try:
                df = pd.read_csv(self.news_csv_path)
                # Clean up any NaN values
                df = df.fillna('')
                logging.info(f"Loaded existing news database with {len(df)} articles")
                return df
            except Exception as e:
                logging.error(f"Error loading news database: {e}")
        
        # Create new database structure
        columns = [
            'Company_Name', 'Search_Query', 'Article_ID', 'Title', 'Date', 
            'Summary', 'Content', 'URL', 'Source', 'Sentiment_Score',
            'Key_Entities', 'Article_Hash', 'Scrape_Date', 'Last_Updated',
            'Relevance_Score', 'Word_Count', 'Language'
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.news_csv_path, index=False)
        logging.info(f"Created new news database: {self.news_csv_path}")
        return df
    
    def setup_driver(self):
        """Setup Selenium WebDriver with enhanced options"""
        if self.driver:
            return
            
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(f"--user-agent={self.headers['User-Agent']}")
        
        # Additional performance options
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_settings.popups": 0
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_page_load_timeout(30)
            logging.info("WebDriver initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def find_company(self, query: str) -> Optional[Dict]:
        """Enhanced company finding with fuzzy matching"""
        query_lower = query.lower().strip()
        
        if self.companies_db.empty:
            return None
        
        # Direct matches first
        for idx, row in self.companies_db.iterrows():
            # Check all string columns for matches
            for col_value in row.values:
                if pd.isna(col_value):
                    continue
                col_str = str(col_value).lower()
                if query_lower == col_str or query_lower in col_str:
                    company_info = row.to_dict()
                    company_info['matched_query'] = query
                    company_info['match_score'] = 1.0
                    logging.info(f"Direct match found: {company_info}")
                    return company_info
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for idx, row in self.companies_db.iterrows():
            for col_value in row.values:
                if pd.isna(col_value):
                    continue
                col_str = str(col_value).lower()
                score = self.calculate_string_similarity(query_lower, col_str)
                if score > 0.7 and score > best_score:
                    best_score = score
                    best_match = row.to_dict()
                    best_match['matched_query'] = query
                    best_match['match_score'] = score
        
        if best_match:
            logging.info(f"Fuzzy match found: {best_match} (score: {best_score})")
            return best_match
        
        logging.warning(f"No company found matching: {query}")
        return None
    
    def calculate_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple string similarity"""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_article_hash(self, title: str, url: str) -> str:
        """Generate unique hash for article to avoid duplicates"""
        content = f"{title.strip()}{url.strip()}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_duplicate_article(self, article_hash: str) -> bool:
        """Check if article already exists in database"""
        if self.news_db.empty:
            return False
        return article_hash in self.news_db['Article_Hash'].values
    
    def scrape_with_requests(self, url: str) -> Optional[BeautifulSoup]:
        """Try scraping with requests first (faster)"""
        try:
            response = self.session.get(url, timeout=10, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logging.debug(f"Requests failed for {url}: {e}")
            return None
    
    def scrape_company_news(self, company_name: str, max_articles: int = 10) -> List[Dict]:
        """Enhanced news scraping with multiple strategies"""
        all_articles = []
        search_terms = self.generate_search_terms(company_name)
        
        logging.info(f"Generated search terms: {search_terms}")
        
        for source_name, source_config in self.news_sources.items():
            if len(all_articles) >= max_articles:
                break
                
            logging.info(f"Scraping {source_name} for {company_name}")
            
            try:
                articles = self.scrape_source_enhanced(source_name, source_config, search_terms, company_name)
                all_articles.extend(articles)
                
                logging.info(f"Found {len(articles)} articles from {source_name}")
                time.sleep(2)  # Be respectful to servers
                
            except Exception as e:
                logging.error(f"Error scraping {source_name}: {e}")
                continue
        
        # Filter and deduplicate
        unique_articles = self.filter_and_deduplicate(all_articles, company_name)
        logging.info(f"Found {len(unique_articles)} unique relevant articles for {company_name}")
        
        return unique_articles[:max_articles]
    
    def generate_search_terms(self, company_name: str) -> List[str]:
        """Generate comprehensive search terms"""
        terms = [company_name.strip()]
        
        # Clean variations
        clean_name = company_name
        for suffix in ['Ltd', 'Limited', 'Pvt', 'Private', 'Inc', 'Corporation', 'Corp']:
            clean_name = re.sub(rf'\b{suffix}\.?\b', '', clean_name, flags=re.IGNORECASE).strip()
        
        if clean_name != company_name:
            terms.append(clean_name)
        
        # Add keyword combinations
        base_terms = [company_name, clean_name] if clean_name != company_name else [company_name]
        
        for base_term in base_terms:
            terms.extend([
                f"{base_term} IPO",
                f"{base_term} shares",
                f"{base_term} stock",
                f"{base_term} listing",
                f"{base_term} market",
                f"{base_term} financial results",
                f"{base_term} earnings"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms[:8]  # Limit to avoid too many requests
    
    def scrape_source_enhanced(self, source_name: str, source_config: Dict, search_terms: List[str], company_name: str) -> List[Dict]:
        """Enhanced source scraping with multiple strategies"""
        articles = []
        
        for i, search_term in enumerate(search_terms[:3]):  # Try top 3 terms
            for strategy_url in source_config['search_strategies']:
                try:
                    # Construct search URL
                    company_slug = search_term.lower().replace(' ', '-').replace('.', '')
                    search_url = strategy_url.format(company=company_slug, search_term=quote_plus(search_term))
                    
                    logging.info(f"Trying: {search_url}")
                    
                    # Try requests first, then Selenium
                    soup = self.scrape_with_requests(search_url)
                    
                    if not soup:
                        # Fallback to Selenium
                        if not self.driver:
                            self.setup_driver()
                        articles_selenium = self.scrape_with_selenium(search_url, source_config, company_name, search_term)
                        articles.extend(articles_selenium)
                    else:
                        # Process with BeautifulSoup
                        articles_bs = self.extract_articles_from_soup(soup, source_config, source_name, company_name, search_term, search_url)
                        articles.extend(articles_bs)
                    
                    if articles:
                        break  # Found articles, move to next search term
                        
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error with strategy {strategy_url}: {e}")
                    continue
            
            if len(articles) >= 5:  # Enough articles from this source
                break
        
        return articles
    
    def extract_articles_from_soup(self, soup: BeautifulSoup, source_config: Dict, source_name: str, company_name: str, search_term: str, base_url: str) -> List[Dict]:
        """Extract articles using BeautifulSoup"""
        articles = []
        
        # Find article links
        article_links = set()
        for selector in source_config['selectors']['articles']:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        if href.startswith('/'):
                            href = urljoin(source_config['base_url'], href)
                        if self.is_valid_news_url(href):
                            article_links.add(href)
            except Exception as e:
                logging.debug(f"Selector failed {selector}: {e}")
                continue
        
        logging.info(f"Found {len(article_links)} potential article links")
        
        # Scrape individual articles
        for i, url in enumerate(list(article_links)[:5]):  # Limit per source
            try:
                article_data = self.scrape_single_article_enhanced(url, source_name, company_name, search_term)
                if article_data:
                    articles.append(article_data)
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error scraping article {url}: {e}")
                continue
        
        return articles
    
    def scrape_with_selenium(self, search_url: str, source_config: Dict, company_name: str, search_term: str) -> List[Dict]:
        """Fallback scraping with Selenium"""
        articles = []
        
        try:
            self.driver.get(search_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
            
            # Extract article links
            article_links = self.extract_article_links_selenium(source_config)
            
            for link in article_links[:5]:
                try:
                    article_data = self.scrape_single_article_enhanced(link, search_url.split('//')[1].split('/')[0], company_name, search_term)
                    if article_data:
                        articles.append(article_data)
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error scraping article {link}: {e}")
                    continue
        
        except TimeoutException:
            logging.error(f"Timeout loading {search_url}")
        except Exception as e:
            logging.error(f"Error in Selenium scraping: {e}")
        
        return articles
    
    def extract_article_links_selenium(self, source_config: Dict) -> List[str]:
        """Extract article links using Selenium"""
        links = []
        
        for selector in source_config['selectors']['articles']:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    href = elem.get_attribute('href')
                    if href and self.is_valid_news_url(href):
                        links.append(href)
                
                if links:
                    break
            except Exception as e:
                logging.debug(f"Selenium selector failed {selector}: {e}")
                continue
        
        return list(set(links))
    
    def scrape_single_article_enhanced(self, url: str, source: str, company_name: str, search_query: str) -> Optional[Dict]:
        """Enhanced single article scraping"""
        try:
            # Check for duplicates first
            temp_hash = self.generate_article_hash("temp", url)
            if self.is_duplicate_article(temp_hash):
                return None
            
            # Try requests first
            soup = self.scrape_with_requests(url)
            
            if soup:
                article_data = self.extract_article_data_soup(soup, url, source, company_name, search_query)
            else:
                # Fallback to Selenium
                if not self.driver:
                    self.setup_driver()
                article_data = self.extract_article_data_selenium(url, source, company_name, search_query)
            
            return article_data
            
        except Exception as e:
            logging.error(f"Error scraping article {url}: {e}")
            return None
    
    def extract_article_data_soup(self, soup: BeautifulSoup, url: str, source: str, company_name: str, search_query: str) -> Optional[Dict]:
        """Extract article data using BeautifulSoup"""
        try:
            # Extract title
            title = ""
            for selector in ['h1', 'title', '.headline', '.entry-title']:
                elem = soup.select_one(selector)
                if elem and elem.get_text().strip():
                    title = elem.get_text().strip()
                    break
            
            if not title or len(title) < 10:
                return None
            
            # Extract date
            date = ""
            for selector in ['time', '[datetime]', '.date', 'meta[property="article:published_time"]']:
                elem = soup.select_one(selector)
                if elem:
                    date = elem.get('datetime') or elem.get('content') or elem.get_text().strip()
                    if date:
                        break
            
            # Extract content
            content_parts = []
            for selector in ['article p', '.content p', '.entry-content p', 'p']:
                paragraphs = soup.select(selector)
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 20:
                        content_parts.append(text)
                if content_parts:
                    break
            
            content = ' '.join(content_parts[:10])  # First 10 paragraphs
            
            # Extract summary
            summary = ""
            meta_desc = soup.select_one('meta[name="description"]')
            if meta_desc:
                summary = meta_desc.get('content', '').strip()
            
            if not summary and content_parts:
                summary = content_parts[0][:200] + "..." if len(content_parts[0]) > 200 else content_parts[0]
            
            # Generate article hash
            article_hash = self.generate_article_hash(title, url)
            
            # Check relevance
            relevance_score = self.nlp.calculate_relevance(f"{title} {summary} {content}", company_name)
            if relevance_score < 0.2:
                return None
            
            # Extract entities and calculate sentiment
            entities = self.nlp.extract_entities(f"{title} {content}")
            sentiment = self.nlp.calculate_sentiment(f"{title} {summary}")
            
            return {
                'Company_Name': company_name,
                'Search_Query': search_query,
                'Article_ID': article_hash,
                'Title': title,
                'Date': date,
                'Summary': summary,
                'Content': content,
                'URL': url,
                'Source': source,
                'Sentiment_Score': sentiment,
                'Key_Entities': entities,
                'Article_Hash': article_hash,
                'Scrape_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Relevance_Score': relevance_score,
                'Word_Count': len(content.split()) if content else 0,
                'Language': 'en'
            }
            
        except Exception as e:
            logging.error(f"Error extracting article data from soup: {e}")
            return None
    
    def extract_article_data_selenium(self, url: str, source: str, company_name: str, search_query: str) -> Optional[Dict]:
        """Extract article data using Selenium (fallback)"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
            
            # Extract using existing methods
            title = self.extract_title()
            date = self.extract_date()
            content = self.extract_content()
            summary = self.extract_summary()
            
            if not title or len(title) < 10:
                return None
            
            article_hash = self.generate_article_hash(title, url)
            relevance_score = self.nlp.calculate_relevance(f"{title} {summary} {content}", company_name)
            
            if relevance_score < 0.2:
                return None
            
            entities = self.nlp.extract_entities(f"{title} {content}")
            sentiment = self.nlp.calculate_sentiment(f"{title} {summary}")
            
            return {
                'Company_Name': company_name,
                'Search_Query': search_query,
                'Article_ID': article_hash,
                'Title': title,
                'Date': date,
                'Summary': summary,
                'Content': content,
                'URL': url,
                'Source': source,
                'Sentiment_Score': sentiment,
                'Key_Entities': entities,
                'Article_Hash': article_hash,
                'Scrape_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Relevance_Score': relevance_score,
                'Word_Count': len(content.split()) if content else 0,
                'Language': 'en'
            }
            
        except Exception as e:
            logging.error(f"Error extracting article data with Selenium: {e}")
            return None
    
    def is_valid_news_url(self, url: str) -> bool:
        """Enhanced URL validation"""
        if not url or not url.startswith('http'):
            return False
        
        # Skip unwanted URLs
        skip_patterns = [
            'javascript:', 'mailto:', '#', 'login', 'register', 'subscribe', 
            'advertisement', 'popup', 'social', 'share', 'video', 'photo',
            'gallery', 'poll', 'quiz', 'widget', 'redirect', 'api.',
            'rss', 'feed', 'sitemap', 'robots.txt'
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False
        
        # Must contain news-related keywords or patterns
        news_patterns = [
            'news', 'article', 'story', 'business', 'companies', 'market', 
            'ipo', 'financial', 'economy', 'stock', 'share', 'earnings'
        ]
        
        # Check URL path and domain
        parsed = urlparse(url)
        url_text = f"{parsed.netloc} {parsed.path}".lower()
        
        return any(pattern in url_text for pattern in news_patterns)
    
    def extract_title(self) -> str:
        """Extract article title using Selenium"""
        selectors = [
            'h1', '.headline', '.title', '.story-headline', 
            '.entry-title', '.article-title', 'title'
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                title = element.text.strip()
                if title and len(title) > 5:
                    return title
            except:
                continue
        
        return self.driver.title.strip()
    
    def extract_date(self) -> str:
        """Extract article date using Selenium"""
        selectors = [
            'time', '[datetime]', '.date', '.publish-date', 
            '.story-date', '[class*="date"]', 'meta[property="article:published_time"]'
        ]
        
        for selector in selectors:
            try:
                if selector.startswith('meta'):
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    return element.get_attribute('content') or ""
                else:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    date_text = element.text.strip() or element.get_attribute('datetime') or ""
                    if date_text:
                        return date_text
            except:
                continue
        
        return ""
    
    def extract_content(self) -> str:
        """Extract article content using Selenium"""
        selectors = [
            'article p', '.story-content p', '.entry-content p',
            '.article-content p', '.content p', '[class*="article"] p', 
            '[class*="story"] p', 'p'
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                content_parts = []
                
                for elem in elements:
                    text = elem.text.strip()
                    if text and len(text) > 20:  # Skip short paragraphs
                        content_parts.append(text)
                
                if content_parts:
                    return ' '.join(content_parts[:10])  # First 10 paragraphs
            except:
                continue
        
        return ""
    
    def extract_summary(self) -> str:
        """Extract article summary using Selenium"""
        try:
            meta_desc = self.driver.find_element(By.CSS_SELECTOR, 'meta[name="description"]')
            return meta_desc.get_attribute('content').strip()
        except:
            return ""
    
    def filter_and_deduplicate(self, articles: List[Dict], company_name: str) -> List[Dict]:
        """Enhanced filtering and deduplication"""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            article_hash = article.get('Article_Hash')
            if not article_hash:
                continue
                
            if article_hash not in seen_hashes and not self.is_duplicate_article(article_hash):
                # Enhanced relevance and quality checks
                title = article.get('Title', '')
                content = article.get('Content', '')
                
                # Skip if too short or poor quality
                if len(title) < 10 or len(content) < 50:
                    continue
                
                # Check relevance score
                relevance_score = article.get('Relevance_Score', 0)
                if relevance_score < 0.2:
                    continue
                
                # Additional quality checks
                if self.is_quality_article(article):
                    unique_articles.append(article)
                    seen_hashes.add(article_hash)
        
        # Sort by relevance score and date
        unique_articles.sort(key=lambda x: (x.get('Relevance_Score', 0), x.get('Date', '')), reverse=True)
        
        return unique_articles
    
    def is_quality_article(self, article: Dict) -> bool:
        """Check if article meets quality standards"""
        title = article.get('Title', '').lower()
        content = article.get('Content', '').lower()
        
        # Skip promotional or low-quality content
        spam_indicators = [
            'click here', 'subscribe now', 'advertisement', 'sponsored',
            'buy now', 'limited time', 'exclusive offer', 'download app'
        ]
        
        for indicator in spam_indicators:
            if indicator in title or indicator in content:
                return False
        
        # Must have reasonable word count
        word_count = article.get('Word_Count', 0)
        if word_count < 50:
            return False
        
        # Must have some business/financial relevance
        business_keywords = [
            'ipo', 'stock', 'share', 'market', 'financial', 'earnings',
            'revenue', 'profit', 'investment', 'business', 'company'
        ]
        
        text_content = f"{title} {content}"
        if not any(keyword in text_content for keyword in business_keywords):
            return False
        
        return True
    
    def save_articles_to_database(self, articles: List[Dict]):
        """Enhanced database saving with error handling"""
        if not articles:
            logging.info("No articles to save")
            return
        
        try:
            # Convert to DataFrame
            new_df = pd.DataFrame(articles)
            
            # Ensure all required columns exist
            required_columns = [
                'Company_Name', 'Search_Query', 'Article_ID', 'Title', 'Date', 
                'Summary', 'Content', 'URL', 'Source', 'Sentiment_Score',
                'Key_Entities', 'Article_Hash', 'Scrape_Date', 'Last_Updated',
                'Relevance_Score', 'Word_Count', 'Language'
            ]
            
            for col in required_columns:
                if col not in new_df.columns:
                    new_df[col] = ''
            
            # Clean data
            new_df = new_df.fillna('')
            new_df = new_df.replace([float('inf'), float('-inf')], 0)
            
            # Append to existing database
            if not self.news_db.empty:
                combined_df = pd.concat([self.news_db, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['Article_Hash'], keep='last')
            
            # Save to CSV with backup
            backup_path = f"{self.news_csv_path}.backup"
            if os.path.exists(self.news_csv_path):
                import shutil
                shutil.copy2(self.news_csv_path, backup_path)
            
            combined_df.to_csv(self.news_csv_path, index=False)
            self.news_db = combined_df
            
            logging.info(f"Saved {len(articles)} new articles. Total: {len(combined_df)}")
            
        except Exception as e:
            logging.error(f"Error saving articles: {e}")
            # Try to restore from backup
            backup_path = f"{self.news_csv_path}.backup"
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, self.news_csv_path)
                logging.info("Restored from backup")
    
    def query_company_news(self, company_query: str, max_articles: int = 10) -> bool:
        """Enhanced main query method with better error handling"""
        logging.info(f"Querying news for: {company_query}")
        
        try:
            # Find company in IPO database
            company_info = self.find_company(company_query)
            if not company_info:
                print(f"❌ Company '{company_query}' not found in IPO database")
                print("💡 Try using a partial name or check the list of available companies")
                return False
            
            # Get company name from the matched result
            company_name = self.get_company_name_from_info(company_info)
            match_score = company_info.get('match_score', 1.0)
            
            print(f"✅ Found company: {company_name} (match score: {match_score:.2f})")
            
            # Scrape news
            print(f"🔍 Scraping news articles...")
            articles = self.scrape_company_news(company_name, max_articles)
            
            if articles:
                # Save to database
                self.save_articles_to_database(articles)
                
                print(f"\n🎉 Successfully scraped {len(articles)} articles for {company_name}")
                print(f"📄 Articles saved to: {self.news_csv_path}")
                
                # Display summary
                self.display_articles_summary(articles, company_name)
                return True
            else:
                print(f"❌ No relevant articles found for {company_name}")
                print("💡 Try searching for a different company or check if the company name is correct")
                return False
                
        except Exception as e:
            logging.error(f"Error in query_company_news: {e}")
            print(f"❌ Error occurred while scraping: {e}")
            return False
    
    def get_company_name_from_info(self, company_info: Dict) -> str:
        """Extract company name from company info dictionary"""
        # Try common column names
        possible_names = ['Company Name', 'Company', 'Name', 'company_name', 'COMPANY', 'COMPANY NAME']
        
        for name_col in possible_names:
            if name_col in company_info and company_info[name_col]:
                return str(company_info[name_col]).strip()
        
        # Fallback to first non-empty string value
        for value in company_info.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        
        return "Unknown Company"
    
    def display_articles_summary(self, articles: List[Dict], company_name: str):
        """Display formatted summary of scraped articles"""
        print(f"\n{'='*80}")
        print(f"SCRAPED ARTICLES SUMMARY FOR {company_name.upper()}")
        print(f"{'='*80}")
        
        # Calculate summary statistics
        total_articles = len(articles)
        sources = list(set(article.get('Source', 'Unknown') for article in articles))
        avg_sentiment = sum(article.get('Sentiment_Score', 0) for article in articles) / total_articles if total_articles > 0 else 0
        avg_relevance = sum(article.get('Relevance_Score', 0) for article in articles) / total_articles if total_articles > 0 else 0
        
        print(f"📊 Total Articles: {total_articles}")
        print(f"📰 Sources: {', '.join(sources)}")
        print(f"💭 Average Sentiment: {avg_sentiment:.2f}")
        print(f"🎯 Average Relevance: {avg_relevance:.2f}")
        print(f"\n{'='*80}")
        
        for i, article in enumerate(articles, 1):
            print(f"\n--- Article {i} ---")
            print(f"📰 Title: {article.get('Title', 'No Title')[:100]}...")
            print(f"📅 Date: {article.get('Date', 'Unknown')}")
            print(f"🏢 Source: {article.get('Source', 'Unknown')}")
            print(f"💭 Sentiment: {article.get('Sentiment_Score', 0):.2f}")
            print(f"🎯 Relevance: {article.get('Relevance_Score', 0):.2f}")
            print(f"📝 Word Count: {article.get('Word_Count', 0)}")
            if article.get('Key_Entities'):
                print(f"🔍 Key Entities: {article.get('Key_Entities')}")
            print(f"🔗 URL: {article.get('URL', 'No URL')}")
            
            # Show summary if available
            summary = article.get('Summary', '')
            if summary:
                print(f"📄 Summary: {summary[:200]}...")
    
    def list_companies(self, limit: int = 20, search_term: str = ""):
        """Enhanced company listing with search functionality"""
        if self.companies_db.empty:
            print("❌ No companies found in IPO database")
            return
        
        # Filter by search term if provided
        if search_term:
            filtered_db = self.companies_db[
                self.companies_db.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1
                )
            ]
            print(f"\n{'='*80}")
            print(f"COMPANIES MATCHING '{search_term.upper()}' (showing first {limit})")
            print(f"{'='*80}")
        else:
            filtered_db = self.companies_db
            print(f"\n{'='*80}")
            print(f"AVAILABLE IPO COMPANIES (showing first {limit})")
            print(f"{'='*80}")
        
        if filtered_db.empty:
            print(f"❌ No companies found matching '{search_term}'")
            return
        
        # Display companies
        for i, (idx, row) in enumerate(filtered_db.head(limit).iterrows(), 1):
            company_name = self.get_company_name_from_info(row.to_dict())
            print(f"{i:2d}. {company_name}")
            
            # Show additional info if available
            if len(row) > 1:
                other_info = []
                for col, val in row.items():
                    if col != company_name and pd.notna(val) and str(val).strip():
                        other_info.append(f"{col}: {val}")
                
                if other_info:
                    print(f"    {' | '.join(other_info[:3])}")  # Show first 3 additional fields
        
        if len(filtered_db) > limit:
            print(f"\n... and {len(filtered_db) - limit} more companies")
    
    def get_news_stats(self):
        """Enhanced news database statistics"""
        if self.news_db.empty:
            print("❌ No news articles in database")
            return
        
        print(f"\n{'='*80}")
        print(f"NEWS DATABASE STATISTICS")
        print(f"{'='*80}")
        
        total_articles = len(self.news_db)
        unique_companies = self.news_db['Company_Name'].nunique()
        sources = self.news_db['Source'].unique()
        
        print(f"📊 Total articles: {total_articles}")
        print(f"🏢 Unique companies: {unique_companies}")
        print(f"📰 Sources: {', '.join(sources)}")
        
        # Date range
        if 'Date' in self.news_db.columns:
            dates = self.news_db['Date'].dropna()
            if not dates.empty:
                print(f"📅 Date range: {dates.min()} to {dates.max()}")
        
        # Sentiment analysis
        if 'Sentiment_Score' in self.news_db.columns:
            sentiment_scores = pd.to_numeric(self.news_db['Sentiment_Score'], errors='coerce').dropna()
            if not sentiment_scores.empty:
                avg_sentiment = sentiment_scores.mean()
                positive_articles = len(sentiment_scores[sentiment_scores > 0.1])
                negative_articles = len(sentiment_scores[sentiment_scores < -0.1])
                neutral_articles = total_articles - positive_articles - negative_articles
                
                print(f"💭 Average sentiment: {avg_sentiment:.2f}")
                print(f"😊 Positive articles: {positive_articles} ({positive_articles/total_articles*100:.1f}%)")
                print(f"😐 Neutral articles: {neutral_articles} ({neutral_articles/total_articles*100:.1f}%)")
                print(f"😞 Negative articles: {negative_articles} ({negative_articles/total_articles*100:.1f}%)")
        
        # Top companies by article count
        company_counts = self.news_db['Company_Name'].value_counts().head(10)
        print(f"\n📈 Top companies by article count:")
        for company, count in company_counts.items():
            print(f"  {company}: {count} articles")
        
        # Recent articles
        if 'Scrape_Date' in self.news_db.columns:
            recent_articles = self.news_db.sort_values('Scrape_Date', ascending=False).head(5)
            print(f"\n🕒 Most recent articles:")
            for _, article in recent_articles.iterrows():
                print(f"  {article.get('Title', 'No Title')[:60]}... ({article.get('Company_Name', 'Unknown')})")
    
    def export_company_news(self, company_name: str, output_file: str = None):
        """Export news for a specific company"""
        if self.news_db.empty:
            print("❌ No news articles in database")
            return
        
        # Filter articles for the company
        company_articles = self.news_db[
            self.news_db['Company_Name'].str.contains(company_name, case=False, na=False)
        ]
        
        if company_articles.empty:
            print(f"❌ No articles found for company: {company_name}")
            return
        
        # Generate output filename if not provided
        if not output_file:
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', company_name.lower())
            output_file = f"{safe_name}_news_export.csv"
        
        try:
            company_articles.to_csv(output_file, index=False)
            print(f"✅ Exported {len(company_articles)} articles for {company_name} to {output_file}")
        except Exception as e:
            print(f"❌ Error exporting articles: {e}")
    
    def cleanup(self):
        """Enhanced cleanup with better error handling"""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("WebDriver closed successfully")
            except Exception as e:
                logging.error(f"Error closing WebDriver: {e}")
        
        # Close requests session
        try:
            self.session.close()
            logging.info("Requests session closed")
        except Exception as e:
            logging.error(f"Error closing requests session: {e}")


class EnhancedNLP:
    """Enhanced NLP for better text analysis"""
    
    def __init__(self):
        self.positive_words = [
            'good', 'great', 'excellent', 'positive', 'growth', 'profit', 'success', 
            'strong', 'rise', 'gain', 'bullish', 'optimistic', 'promising', 'robust',
            'surge', 'boom', 'rally', 'upward', 'upgrade', 'outperform', 'beat',
            'exceed', 'impressive', 'solid', 'healthy', 'momentum'
        ]
        
        self.negative_words = [
            'bad', 'poor', 'negative', 'decline', 'loss', 'weak', 'fall', 'drop',
            'concern', 'risk', 'bearish', 'pessimistic', 'disappointing', 'vulnerable',
            'plunge', 'crash', 'slump', 'downward', 'downgrade', 'underperform',
            'miss', 'below', 'worse', 'trouble', 'challenge', 'pressure'
        ]
        
        self.business_keywords = [
            'ipo', 'shares', 'stock', 'market', 'business', 'company', 'financial',
            'earnings', 'revenue', 'profit', 'investment', 'listing', 'trading',
            'valuation', 'dividend', 'acquisition', 'merger', 'partnership',
            'expansion', 'growth', 'performance', 'results', 'quarter'
        ]
        
        self.entity_patterns = {
            'money': r'₹\s*\d+(?:,\d{2,3})*(?:\.\d{2})?\s*(?:crore|lakh|billion|million|thousand)?',
            'percentage': r'\d+(?:\.\d+)?%',
            'dates': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'numbers': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        }
    
    def calculate_relevance(self, text: str, company_name: str) -> float:
        """Enhanced relevance calculation"""
        if not text or not company_name:
            return 0.0
        
        text_lower = text.lower()
        company_lower = company_name.lower()
        
        score = 0.0
        
        # Direct company name match (highest weight)
        if company_lower in text_lower:
            score += 0.6
        
        # Partial company name matches
        company_words = [word for word in company_lower.split() if len(word) > 2]
        matched_words = sum(1 for word in company_words if word in text_lower)
        if company_words:
            score += (matched_words / len(company_words)) * 0.3
        
        # Business keywords relevance
        business_matches = sum(1 for keyword in self.business_keywords if keyword in text_lower)
        score += min(business_matches * 0.02, 0.1)
        
        return min(score, 1.0)
    
    def calculate_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count weighted sentiment words
        positive_score = 0
        negative_score = 0
        
        # Simple word counting with weights
        for word in self.positive_words:
            count = text_lower.count(word)
            if word in ['excellent', 'outstanding', 'exceptional']:
                positive_score += count * 2  # Higher weight for strong positive words
            else:
                positive_score += count
        
        for word in self.negative_words:
            count = text_lower.count(word)
            if word in ['terrible', 'awful', 'disastrous']:
                negative_score += count * 2  # Higher weight for strong negative words
            else:
                negative_score += count
        
        # Normalize sentiment
        total_sentiment_words = positive_score + negative_score
        
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_score - negative_score) / total_sentiment_words
    
    def extract_entities(self, text: str) -> str:
        """Enhanced entity extraction"""
        if not text:
            return ""
        
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take unique matches and limit to avoid too long strings
                unique_matches = list(set(matches))[:3]
                entities.extend(unique_matches)
        
        return '; '.join(entities) if entities else ""


# Enhanced Interactive Interface
def main():
    """Enhanced main interactive interface"""
    print("🚀 Enhanced Company News Scraper Agent v2.0")
    print("=" * 60)
    print("📊 Scrapes news from MoneyControl, LiveMint, Economic Times, and Business Standard")
    print("🔍 Uses both BeautifulSoup and Selenium for reliable scraping")
    print("🧠 Enhanced NLP for better relevance and sentiment analysis")
    print("=" * 60)
    
    # Get IPO CSV path
    ipo_csv_path = input("\n📁 Enter path to your IPO CSV file: ").strip()
    
    if not os.path.exists(ipo_csv_path):
        print(f"❌ File not found: {ipo_csv_path}")
        print("💡 Make sure the file path is correct and the file exists")
        return
    
    # Get news database path (optional)
    news_csv_path = input("📄 Enter news database path (press Enter for default): ").strip()
    if not news_csv_path:
        news_csv_path = "company_news_database.csv"
    
    # Initialize scraper
    try:
        print("\n🔧 Initializing scraper...")
        scraper = CompanyNewsScraperAgent(ipo_csv_path, news_csv_path)
        print("✅ Scraper initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing scraper: {e}")
        return
    
    # Main interaction loop
    while True:
        print(f"\n{'='*60}")
        print("🎯 MAIN MENU")
        print("=" * 60)
        print("1. 🔍 Query company news")
        print("2. 📋 List available companies")
        print("3. 🔍 Search companies")
        print("4. 📊 View news database stats")
        print("5. 📤 Export company news")
        print("6. 🧹 Cleanup and exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            company_query = input("\n🏢 Enter company name to search: ").strip()
            if company_query:
                max_articles = input("📄 Max articles to scrape (default 10): ").strip()
                max_articles = int(max_articles) if max_articles.isdigit() else 10
                
                print(f"\n🚀 Starting news scraping for '{company_query}'...")
                try:
                    success = scraper.query_company_news(company_query, max_articles)
                    if success:
                        print("\n✅ Scraping completed successfully!")
                    else:
                        print("\n❌ Scraping failed or no articles found")
                except KeyboardInterrupt:
                    print("\n⚠️ Scraping interrupted by user")
                except Exception as e:
                    print(f"\n❌ Error during scraping: {e}")
        
        elif choice == '2':
            limit = input("\n📊 Number of companies to show (default 20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            scraper.list_companies(limit)
        
        elif choice == '3':
            search_term = input("\n🔍 Enter search term for companies: ").strip()
            if search_term:
                limit = input("📊 Number of results to show (default 20): ").strip()
                limit = int(limit) if limit.isdigit() else 20
                scraper.list_companies(limit, search_term)
        
        elif choice == '4':
            scraper.get_news_stats()
        
        elif choice == '5':
            company_name = input("\n🏢 Enter company name to export: ").strip()
            if company_name:
                output_file = input("📁 Output file name (press Enter for auto): ").strip()
                output_file = output_file if output_file else None
                scraper.export_company_news(company_name, output_file)
        
        elif choice == '6':
            print("\n🧹 Cleaning up...")
            scraper.cleanup()
            print("👋 Thank you for using Company News Scraper Agent!")
            print("📄 Your scraped data is saved in:", news_csv_path)
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
        
        # Wait for user input before showing menu again
        if choice in ['1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.error(f"Unexpected error in main: {e}")
    finally:
        print("🔚 Program ended")