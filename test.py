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

@dataclass
class NewsArticle:
    """Data class for news articles"""
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
        return {
            'company': self.company,
            'title': self.title,
            'summary': self.summary,
            'url': self.url,
            'source': self.source,
            'publish_date': self.publish_date,
            'content': self.content,
            'sentiment': self.sentiment,
            'method': self.method,
            'scraped_at': self.scraped_at
        }
    
    def get_hash(self) -> str:
        """Generate unique hash for deduplication"""
        content_for_hash = f"{self.title.lower()}{self.url}{self.company.lower()}"
        return hashlib.md5(content_for_hash.encode()).hexdigest()

class IPONewsAgent:
    def __init__(self, config: Dict):
        """
        Initialize the AI agent with configuration
        config = {
            'newsapi_key': 'your_key',
            'gnews_key': 'your_key',
            'memory_file': 'agent_memory.json',
            'use_selenium': False
        }
        """
        self.config = config
        self.articles: List[NewsArticle] = []
        self.seen_hashes: Set[str] = set()
        self.memory_file = config.get('memory_file', 'agent_memory.json')
        
        # Define methods first
        self.methods = {
            'newsapi': self.method1_newsapi,
            'rss': self.method2_rss_feeds,
            'fallback': self.method3_fallback
        }
        
        # Load memory after methods are defined
        self.memory = self._load_memory()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # RSS feeds
        self.rss_feeds = [
            {'url': 'https://economictimes.indiatimes.com/markets/ipo/rssfeeds/13357785.cms', 'name': 'Economic Times IPO'},
            {'url': 'https://www.moneycontrol.com/rss/ipo.xml', 'name': 'MoneyControl IPO'}
        ]
    
    def _load_memory(self) -> Dict:
        """Load agent memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load memory: {e}")
        return {
            'method_performance': {method: {'success': 0, 'failures': 0} for method in self.methods},
            'user_preferences': {'preferred_methods': [], 'output_format': 'csv'},
            'last_run': None
        }
    
    def _save_memory(self):
        """Save agent memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return "POSITIVE" if polarity > 0.1 else "NEGATIVE" if polarity < -0.1 else "NEUTRAL"
        except:
            return "NEUTRAL"
    
    def add_article(self, article: NewsArticle) -> bool:
        """Add article if not duplicate"""
        article_hash = article.get_hash()
        if article_hash not in self.seen_hashes:
            article.sentiment = self.analyze_sentiment(article.summary or article.title)
            article.scraped_at = datetime.now().isoformat()
            self.articles.append(article)
            self.seen_hashes.add(article_hash)
            return True
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
                    'q': f"{company} IPO",
                    'apiKey': self.config['newsapi_key'],
                    'pageSize': 3,
                    'sortBy': 'publishedAt',
                    'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'language': 'en'
                }
                
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for article_data in data.get('articles', []):
                    article = NewsArticle(
                        company=company,
                        title=article_data.get('title', ''),
                        summary=article_data.get('description', ''),
                        url=article_data.get('url', ''),
                        source=article_data.get('source', {}).get('name', 'NewsAPI'),
                        publish_date=article_data.get('publishedAt', ''),
                        method='NewsAPI'
                    )
                    if self.add_article(article):
                        articles_added += 1
                time.sleep(1)
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
        for feed in self.rss_feeds:
            try:
                feed_data = feedparser.parse(feed['url'])
                for entry in feed_data.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    for company in companies:
                        if company.lower() in title.lower() or 'ipo' in summary.lower():
                            article = NewsArticle(
                                company=company,
                                title=title,
                                summary=summary,
                                url=entry.get('link', ''),
                                source=feed['name'],
                                publish_date=entry.get('published', ''),
                                method='RSS'
                            )
                            if self.add_article(article):
                                articles_added += 1
                            break
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"RSS error: {e}")
                self.memory['method_performance']['rss']['failures'] += 1
            self.memory['method_performance']['rss']['success'] += articles_added
            self._save_memory()
        return articles_added
    
    def method3_fallback(self, companies: List[str]) -> int:
        """Fallback method"""
        articles_added = 0
        for company in companies:
            article = NewsArticle(
                company=company,
                title=f"{company} IPO Update",
                summary=f"Summary of {company} IPO activities.",
                url=f"https://example.com/{company.lower().replace(' ', '-')}-ipo",
                source='Fallback',
                publish_date=datetime.now().strftime('%Y-%m-%d'),
                method='Fallback'
            )
            if self.add_article(article):
                articles_added += 1
        self.memory['method_performance']['fallback']['success'] += articles_added
        self._save_memory()
        return articles_added
    
    def select_methods(self) -> List[str]:
        """Select scraping methods based on performance and preferences"""
        preferred = self.memory['user_preferences'].get('preferred_methods', [])
        if preferred:
            return [m for m in preferred if m in self.methods]
        
        method_scores = {
            k: self.memory['method_performance'][k].get('success', 0) /
               max(self.memory['method_performance'][k].get('failures', 0), 1)
            for k in self.methods
        }
        return sorted(method_scores, key=method_scores.get, reverse=True)
    
    def scrape(self, companies: List[str], methods: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute scraping"""
        self.logger.info(f"Scraping for {len(companies)} companies")
        methods = methods or self.select_methods()
        articles_added = 0
        
        for method_name in methods:
            try:
                articles_added += self.methods[method_name](companies)
            except Exception as e:
                self.logger.error(f"Method {method_name} failed: {e}")
        
        if self.articles:
            df = pd.DataFrame([article.to_dict() for article in self.articles])
            df['scraping_session'] = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.memory['last_run'] = datetime.now().isoformat()
            self._save_memory()
            return df
        return pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename or f"ipo_news_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"Saved to: {filename}")
        return filename
    
    def process_command(self, command: str) -> str:
        """Process user input command"""
        command_lower = command.lower()
        if 'help' in command_lower or '?' in command_lower:
            return (
                "Commands:\n"
                "  scan [company1, company2, ...] - Scan for IPO news\n"
                "  set method [newsapi|rss|fallback] - Set preferred methods\n"
                "  schedule [daily|weekly] - Schedule periodic scanning\n"
                "  status - Show agent status\n"
                "  exit - Exit agent"
            )
        
        elif 'scan' in command_lower:
            try:
                companies = [c.strip() for c in command.split('[')[1].split(']')[0].split(',') if c.strip()]
                if companies:
                    df = self.scrape(companies)
                    if not df.empty:
                        filename = self.save_results(df)
                        return f"Found {len(df)} articles. Saved to {filename}\nSample:\n{df[['company', 'title']].head().to_string()}"
                    return "No articles found."
                return "Please specify companies, e.g., 'scan [Company A, Company B]'"
            except IndexError:
                return "Invalid format. Use: scan [Company A, Company B]"
        
        elif 'set method' in command_lower:
            try:
                methods = [m.strip() for m in command_lower.split('[')[1].split(']')[0].split(',') if m in self.methods]
                if methods:
                    self.memory['user_preferences']['preferred_methods'] = methods
                    self._save_memory()
                    return f"Preferred methods set to {', '.join(methods)}"
                return "Invalid methods specified."
            except IndexError:
                return "Invalid format. Use: set method [newsapi|rss|fallback]"
        
        elif 'schedule' in command_lower:
            if 'daily' in command_lower:
                schedule.every().day.at("09:00").do(self.scrape, companies=['Sample Company'])
                return "Daily scan scheduled at 09:00."
            elif 'weekly' in command_lower:
                schedule.every().monday.at("09:00").do(self.scrape, companies=['Sample Company'])
                return "Weekly scan scheduled for Mondays at 09:00."
            return "Specify 'daily' or 'weekly'."
        
        elif 'status' in command_lower:
            return (
                f"Agent Status:\n"
                f"Last run: {self.memory.get('last_run', 'Never')}\n"
                f"Articles collected: {len(self.articles)}\n"
                f"Method performance: {self.memory['method_performance']}\n"
                f"Preferred methods: {self.memory['user_preferences']['preferred_methods']}"
            )
        
        elif 'exit' in command_lower:
            return "exit"
        
        return "Unknown command. Type 'help' for available commands."
    
    def run(self):
        """Run the conversational agent"""
        print("🤖 IPO News Agent started. Type 'help' to see commands.")
        while True:
            command = input("> ").strip()
            response = self.process_command(command)
            print(response)
            if response == "exit":
                break
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    config = {
        'newsapi_key': 'YOUR_NEWSAPI_KEY',
        'gnews_key': '',
        'memory_file': 'agent_memory.json',
        'use_selenium': False
    }
    agent = IPONewsAgent(config)
    agent.run()