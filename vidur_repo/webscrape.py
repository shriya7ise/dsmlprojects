import pandas as pd
import logging
from datetime import datetime
import time
import os
import re
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from urllib.parse import quote_plus, urljoin, urlparse
import hashlib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import argparse
import signal
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()])

class CompanyNewsScraperAgent:
    def __init__(self):
        load_dotenv()
        self.ipo_csv_path = os.getenv("IPO_CSV_PATH", "/Users/shriya/Documents/GitHub/logo_detect/dsmlprojects222/List_IPO.csv")
        self.news_csv_path = os.getenv("NEWS_CSV_PATH", "company_news_database.csv")
        self.max_articles = int(os.getenv("MAX_ARTICLES", 10))
        self.driver = None
        self.session = requests.Session()
        self.companies_db = self.load_companies()
        self.news_db = self.load_news_database()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5', 'Accept-Encoding': 'gzip, deflate', 'Connection': 'keep-alive'
        }
        self.session.headers.update(self.headers)
        self.nlp = EnhancedNLP()
        self.news_sources = {
            'moneycontrol': {
                'base_url': 'https://www.moneycontrol.com',
                'search_strategies': [
                    'https://www.moneycontrol.com/news/tags/{company}.html',
                    'https://www.moneycontrol.com/india/stockpricequote/{company}',
                    'https://www.google.com/search?q=site:moneycontrol.com {company} news'
                ],
                'selectors': {
                    'articles': ['a[href*="/news/business"]', 'a[href*="/news/company"]', '.news_links a', 'h2 a'],
                    'title': ['h1', '.artTitle', '.news_title', 'h2'],
                    'date': ['time', '.publish_on', '.date', '.news_date'],
                    'content': ['article p', '.content p', '.arti-content p', '.news_content p'],
                    'summary': ['meta[name="description"]', '.summary', '.excerpt']
                }
            },
            'livemint': {
                'base_url': 'https://www.livemint.com',
                'search_strategies': ['https://www.livemint.com/search?q={company}', 'https://www.google.com/search?q=site:livemint.com {company}'],
                'selectors': {
                    'articles': ['a[href*="/companies"]', 'a[href*="/market"]', 'a[href*="/news"]', '.story a'],
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
                    'articles': ['a[href*="/news/"]', 'a[href*="/markets/"]', '.eachStory a'],
                    'title': ['h1', '.story-headline', 'h2'],
                    'date': ['time', '.publish-date', '.date'],
                    'content': ['article p', '.story-content p', 'p'],
                    'summary': ['meta[name="description"]']
                }
            },
            'business_standard': {
                'base_url': 'https://www.business-standard.com',
                'search_strategies': ['https://www.business-standard.com/search?q={company}', 'https://www.google.com/search?q=site:business-standard.com {company}'],
                'selectors': {
                    'articles': ['a[href*="/article/"]', 'a[href*="/companies/"]', '.listing-page a'],
                    'title': ['h1', '.headline'],
                    'date': ['time', '.date'],
                    'content': ['article p', '.article-content p'],
                    'summary': ['meta[name="description"]']
                }
            }
        }

    def load_companies(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.ipo_csv_path)
            company_col = next((col for col in ['Company Name', 'Company', 'Name', 'company_name', 'COMPANY NAME'] if col in df.columns), None)
            if company_col:
                df[company_col] = df[company_col].astype(str).str.strip()
                df = df[df[company_col].notna() & (df[company_col] != '')]
                logging.info(f"Loaded {len(df)} companies from {self.ipo_csv_path}")
            else:
                logging.warning(f"No company name column found in {self.ipo_csv_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading companies: {e}")
            return pd.DataFrame()

    def load_news_database(self) -> pd.DataFrame:
        if os.path.exists(self.news_csv_path):
            try:
                df = pd.read_csv(self.news_csv_path).fillna('')
                logging.info(f"Loaded news database with {len(df)} articles")
                return df
            except Exception as e:
                logging.error(f"Error loading news database: {e}")
        df = pd.DataFrame(columns=[
            'Company_Name', 'Search_Query', 'Article_ID', 'Title', 'Date', 'Summary', 'Content', 'URL', 'Source',
            'Sentiment_Score', 'Key_Entities', 'Article_Hash', 'Scrape_Date', 'Last_Updated', 'Relevance_Score', 'Word_Count', 'Language'
        ])
        df.to_csv(self.news_csv_path, index=False)
        logging.info(f"Created new news database: {self.news_csv_path}")
        return df

    def setup_driver(self):
        if self.driver:
            return
        options = Options()
        for arg in ["--headless", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--window-size=1920,1080",
                    "--disable-blink-features=AutomationControlled", "--disable-extensions", "--disable-plugins", "--disable-images"]:
            options.add_argument(arg)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f"--user-agent={self.headers['User-Agent']}")
        options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2, "profile.default_content_settings.popups": 0})
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_page_load_timeout(30)
            logging.info("WebDriver initialized")
        except Exception as e:
            logging.error(f"Failed to initialize WebDriver: {e}")
            raise

    def find_company(self, query: str) -> Optional[Dict]:
        query_lower = query.lower().strip()
        if self.companies_db.empty:
            return None
        for idx, row in self.companies_db.iterrows():
            for col_value in row.values:
                if pd.isna(col_value):
                    continue
                if query_lower in str(col_value).lower():
                    company_info = row.to_dict()
                    company_info.update({'matched_query': query, 'match_score': 1.0})
                    logging.info(f"Match found: {company_info}")
                    return company_info
        best_match, best_score = None, 0.0
        for idx, row in self.companies_db.iterrows():
            for col_value in row.values:
                if pd.isna(col_value):
                    continue
                score = self.calculate_string_similarity(query_lower, str(col_value).lower())
                if score > 0.7 and score > best_score:
                    best_score = score
                    best_match = row.to_dict()
                    best_match.update({'matched_query': query, 'match_score': score})
        if best_match:
            logging.info(f"Fuzzy match found: {best_match} (score: {best_score})")
            return best_match
        logging.warning(f"No company found for: {query}")
        return None

    def calculate_string_similarity(self, s1: str, s2: str) -> float:
        words1, words2 = set(s1.split()), set(s2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def generate_article_hash(self, title: str, url: str) -> str:
        return hashlib.md5(f"{title.strip()}{url.strip()}".encode('utf-8')).hexdigest()

    def is_duplicate_article(self, article_hash: str) -> bool:
        return article_hash in self.news_db['Article_Hash'].values if not self.news_db.empty else False

    def scrape_with_requests(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=10, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logging.debug(f"Requests failed for {url}: {e}")
            return None

    def scrape_company_news(self, company_name: str, max_articles: int = 10) -> List[Dict]:
        all_articles = []
        search_terms = self.generate_search_terms(company_name)[:3]
        logging.info(f"Search terms: {search_terms}")
        for source_name, source_config in self.news_sources.items():
            if len(all_articles) >= max_articles:
                break
            logging.info(f"Scraping {source_name} for {company_name}")
            try:
                articles = self.scrape_source_enhanced(source_name, source_config, search_terms, company_name)
                all_articles.extend(articles)
                logging.info(f"Found {len(articles)} articles from {source_name}")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error scraping {source_name}: {e}")
        unique_articles = self.filter_and_deduplicate(all_articles, company_name)[:max_articles]
        logging.info(f"Found {len(unique_articles)} unique articles for {company_name}")
        return unique_articles

    def generate_search_terms(self, company_name: str) -> List[str]:
        terms = [company_name.strip()]
        clean_name = re.sub(r'\b(Ltd|Limited|Pvt|Private|Inc|Corporation|Corp)\.?\b', '', company_name, flags=re.IGNORECASE).strip()
        if clean_name != company_name:
            terms.append(clean_name)
        base_terms = [company_name, clean_name] if clean_name != company_name else [company_name]
        for base_term in base_terms:
            terms.extend([f"{base_term} {suffix}" for suffix in ["IPO", "shares", "stock", "listing", "market", "financial results", "earnings"]])
        return list(dict.fromkeys(term.lower() for term in terms))[:8]

    def scrape_source_enhanced(self, source_name: str, source_config: Dict, search_terms: List[str], company_name: str) -> List[Dict]:
        articles = []
        for search_term in search_terms:
            for strategy_url in source_config['search_strategies']:
                try:
                    search_url = strategy_url.format(company=search_term.lower().replace(' ', '-').replace('.', ''), search_term=quote_plus(search_term))
                    logging.info(f"Trying: {search_url}")
                    soup = self.scrape_with_requests(search_url)
                    if soup:
                        articles.extend(self.extract_articles_from_soup(soup, source_config, source_name, company_name, search_term, search_url))
                    else:
                        if not self.driver:
                            self.setup_driver()
                        articles.extend(self.scrape_with_selenium(search_url, source_config, company_name, search_term))
                    if articles:
                        break
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error with strategy {strategy_url}: {e}")
            if len(articles) >= 5:
                break
        return articles

    def extract_articles_from_soup(self, soup: BeautifulSoup, source_config: Dict, source_name: str, company_name: str, search_term: str, base_url: str) -> List[Dict]:
        articles = []
        article_links = set()
        for selector in source_config['selectors']['articles']:
            try:
                for link in soup.select(selector):
                    href = link.get('href')
                    if href:
                        href = urljoin(source_config['base_url'], href) if href.startswith('/') else href
                        if self.is_valid_news_url(href):
                            article_links.add(href)
            except Exception as e:
                logging.debug(f"Selector failed {selector}: {e}")
        logging.info(f"Found {len(article_links)} article links")
        for url in list(article_links)[:5]:
            try:
                if article_data := self.scrape_single_article_enhanced(url, source_name, company_name, search_term):
                    articles.append(article_data)
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error scraping article {url}: {e}")
        return articles

    def scrape_with_selenium(self, search_url: str, source_config: Dict, company_name: str, search_term: str) -> List[Dict]:
        articles = []
        try:
            self.driver.get(search_url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
            for link in self.extract_article_links_selenium(source_config)[:5]:
                try:
                    if article_data := self.scrape_single_article_enhanced(link, search_url.split('//')[1].split('/')[0], company_name, search_term):
                        articles.append(article_data)
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error scraping article {link}: {e}")
        except TimeoutException:
            logging.error(f"Timeout loading {search_url}")
        except Exception as e:
            logging.error(f"Selenium error: {e}")
        return articles

    def extract_article_links_selenium(self, source_config: Dict) -> List[str]:
        links = set()
        for selector in source_config['selectors']['articles']:
            try:
                for elem in self.driver.find_elements(By.CSS_SELECTOR, selector):
                    if href := elem.get_attribute('href'):
                        if self.is_valid_news_url(href):
                            links.add(href)
                if links:
                    break
            except Exception as e:
                logging.debug(f"Selenium selector failed {selector}: {e}")
        return list(links)

    def scrape_single_article_enhanced(self, url: str, source: str, company_name: str, search_query: str) -> Optional[Dict]:
        if self.is_duplicate_article(self.generate_article_hash("temp", url)):
            return None
        soup = self.scrape_with_requests(url)
        return self.extract_article_data_soup(soup, url, source, company_name, search_query) if soup else (
            self.setup_driver() or self.extract_article_data_selenium(url, source, company_name, search_query))

    def extract_article_data_soup(self, soup: BeautifulSoup, url: str, source: str, company_name: str, search_query: str) -> Optional[Dict]:
        try:
            title = next((soup.select_one(selector).get_text().strip() for selector in ['h1', 'title', '.headline', '.entry-title'] if soup.select_one(selector) and soup.select_one(selector).get_text().strip()), "")
            if not title or len(title) < 10:
                return None
            date = next((soup.select_one(selector).get('datetime') or soup.select_one(selector).get('content') or soup.select_one(selector).get_text().strip()
                         for selector in ['time', '[datetime]', '.date', 'meta[property="article:published_time"]'] if soup.select_one(selector)), "")
            content_parts = []
            for selector in ['article p', '.content p', '.entry-content p', 'p']:
                content_parts = [p.get_text().strip() for p in soup.select(selector) if p.get_text().strip() and len(p.get_text().strip()) > 20]
                if content_parts:
                    break
            content = ' '.join(content_parts[:10])
            summary = (soup.select_one('meta[name="description"]').get('content', '').strip() if soup.select_one('meta[name="description"]') else
                       content_parts[0][:200] + "..." if content_parts and len(content_parts[0]) > 200 else content_parts[0] if content_parts else "")
            article_hash = self.generate_article_hash(title, url)
            relevance_score = self.nlp.calculate_relevance(f"{title} {summary} {content}", company_name)
            if relevance_score < 0.2:
                return None
            return {
                'Company_Name': company_name, 'Search_Query': search_query, 'Article_ID': article_hash, 'Title': title, 'Date': date,
                'Summary': summary, 'Content': content, 'URL': url, 'Source': source, 'Sentiment_Score': self.nlp.calculate_sentiment(f"{title} {summary}"),
                'Key_Entities': self.nlp.extract_entities(f"{title} {content}"), 'Article_Hash': article_hash,
                'Scrape_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Relevance_Score': relevance_score, 'Word_Count': len(content.split()) if content else 0, 'Language': 'en'
            }
        except Exception as e:
            logging.error(f"Error extracting article data: {e}")
            return None

    def extract_article_data_selenium(self, url: str, source: str, company_name: str, search_query: str) -> Optional[Dict]:
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
            title = next((self.driver.find_element(By.CSS_SELECTOR, selector).text.strip() for selector in ['h1', '.headline', '.title', '.story-headline', '.entry-title', '.article-title', 'title']
                          if self.driver.find_elements(By.CSS_SELECTOR, selector) and self.driver.find_element(By.CSS_SELECTOR, selector).text.strip()), self.driver.title.strip())
            if not title or len(title) < 10:
                return None
            date = next((self.driver.find_element(By.CSS_SELECTOR, selector).get_attribute('content') or self.driver.find_element(By.CSS_SELECTOR, selector).text.strip() or self.driver.find_element(By.CSS_SELECTOR, selector).get_attribute('datetime') or ""
                         for selector in ['time', '[datetime]', '.date', '.publish-date', '.story-date', '[class*="date"]', 'meta[property="article:published_time"]']
                         if self.driver.find_elements(By.CSS_SELECTOR, selector)), "")
            content_parts = []
            for selector in ['article p', '.story-content p', '.entry-content p', '.article-content p', '.content p', '[class*="article"] p', '[class*="story"] p', 'p']:
                content_parts = [elem.text.strip() for elem in self.driver.find_elements(By.CSS_SELECTOR, selector) if elem.text.strip() and len(elem.text.strip()) > 20]
                if content_parts:
                    break
            content = ' '.join(content_parts[:10])
            summary = self.driver.find_element(By.CSS_SELECTOR, 'meta[name="description"]').get_attribute('content').strip() if self.driver.find_elements(By.CSS_SELECTOR, 'meta[name="description"]') else ""
            if not summary and content_parts:
                summary = content_parts[0][:200] + "..." if len(content_parts[0]) > 200 else content_parts[0]
            article_hash = self.generate_article_hash(title, url)
            relevance_score = self.nlp.calculate_relevance(f"{title} {summary} {content}", company_name)
            if relevance_score < 0.2:
                return None
            return {
                'Company_Name': company_name, 'Search_Query': search_query, 'Article_ID': article_hash, 'Title': title, 'Date': date,
                'Summary': summary, 'Content': content, 'URL': url, 'Source': source, 'Sentiment_Score': self.nlp.calculate_sentiment(f"{title} {summary}"),
                'Key_Entities': self.nlp.extract_entities(f"{title} {content}"), 'Article_Hash': article_hash,
                'Scrape_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Relevance_Score': relevance_score, 'Word_Count': len(content.split()) if content else 0, 'Language': 'en'
            }
        except Exception as e:
            logging.error(f"Error extracting Selenium article data: {e}")
            return None

    def is_valid_news_url(self, url: str) -> bool:
        if not url or not url.startswith('http'):
            return False
        url_lower = url.lower()
        skip_patterns = ['javascript:', 'mailto:', '#', 'login', 'register', 'subscribe', 'advertisement', 'popup', 'social', 'share', 'video', 'photo', 'gallery', 'poll', 'quiz', 'widget', 'redirect', 'api.', 'rss', 'feed', 'sitemap', 'robots.txt']
        if any(pattern in url_lower for pattern in skip_patterns):
            return False
        news_patterns = ['news', 'article', 'story', 'business', 'companies', 'market', 'ipo', 'financial', 'economy', 'stock', 'share', 'earnings']
        return any(pattern in f"{urlparse(url).netloc} {urlparse(url).path}".lower() for pattern in news_patterns)

    def filter_and_deduplicate(self, articles: List[Dict], company_name: str) -> List[Dict]:
        seen_hashes = set()
        unique_articles = []
        for article in articles:
            if article.get('Article_Hash') and article.get('Article_Hash') not in seen_hashes and not self.is_duplicate_article(article['Article_Hash']):
                if len(article.get('Title', '')) >= 10 and len(article.get('Content', '')) >= 50 and article.get('Relevance_Score', 0) >= 0.2 and self.is_quality_article(article):
                    unique_articles.append(article)
                    seen_hashes.add(article['Article_Hash'])
        return sorted(unique_articles, key=lambda x: (x.get('Relevance_Score', 0), x.get('Date', '')), reverse=True)

    def is_quality_article(self, article: Dict) -> bool:
        text = f"{article.get('Title', '').lower()} {article.get('Content', '').lower()}"
        spam_indicators = ['click here', 'subscribe now', 'advertisement', 'sponsored', 'buy now', 'limited time', 'exclusive offer', 'download app']
        if any(indicator in text for indicator in spam_indicators) or article.get('Word_Count', 0) < 50:
            return False
        business_keywords = ['ipo', 'stock', 'share', 'market', 'financial', 'earnings', 'revenue', 'profit', 'investment', 'business', 'company']
        return any(keyword in text for keyword in business_keywords)

    def save_articles_to_database(self, articles: List[Dict]):
        if not articles:
            logging.info("No articles to save")
            return
        try:
            new_df = pd.DataFrame(articles).fillna('')
            new_df = new_df.replace([float('inf'), float('-inf')], 0)
            combined_df = pd.concat([self.news_db, new_df], ignore_index=True) if not self.news_db.empty else new_df
            combined_df = combined_df.drop_duplicates(subset=['Article_Hash'], keep='last')
            backup_path = f"{self.news_csv_path}.backup"
            if os.path.exists(self.news_csv_path):
                import shutil
                shutil.copy2(self.news_csv_path, backup_path)
            combined_df.to_csv(self.news_csv_path, index=False)
            self.news_db = combined_df
            logging.info(f"Saved {len(articles)} articles. Total: {len(combined_df)}")
        except Exception as e:
            logging.error(f"Error saving articles: {e}")
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, self.news_csv_path)
                logging.info("Restored from backup")

    def query_company_news(self, company_query: str, max_articles: int = 10) -> bool:
        logging.info(f"Querying news for: {company_query}")
        try:
            company_info = self.find_company(company_query)
            if not company_info:
                print(f"❌ Company '{company_query}' not found")
                return False
            company_name = self.get_company_name_from_info(company_info)
            print(f"✅ Found: {company_name} (match score: {company_info.get('match_score', 1.0):.2f})")
            print("🔍 Scraping...")
            articles = self.scrape_company_news(company_name, max_articles)
            if articles:
                self.save_articles_to_database(articles)
                print(f"🎉 Scraped {len(articles)} articles for {company_name}")
                print(f"📄 Saved to: {self.news_csv_path}")
                self.display_articles_summary(articles, company_name)
                return True
            print(f"❌ No articles found for {company_name}")
            return False
        except Exception as e:
            logging.error(f"Error querying news: {e}")
            print(f"❌ Error: {e}")
            return False

    def get_company_name_from_info(self, company_info: Dict) -> str:
        for name_col in ['Company Name', 'Company', 'Name', 'company_name', 'COMPANY', 'COMPANY NAME']:
            if name_col in company_info and company_info[name_col]:
                return str(company_info[name_col]).strip()
        return next((value.strip() for value in company_info.values() if isinstance(value, str) and value.strip()), "Unknown Company")

    def display_articles_summary(self, articles: List[Dict], company_name: str):
        print(f"\n{'='*80}\nSCRAPED ARTICLES SUMMARY FOR {company_name.upper()}\n{'='*80}")
        total_articles = len(articles)
        sources = list(set(article.get('Source', 'Unknown') for article in articles))
        avg_sentiment = sum(article.get('Sentiment_Score', 0) for article in articles) / total_articles if total_articles else 0
        avg_relevance = sum(article.get('Relevance_Score', 0) for article in articles) / total_articles if total_articles else 0
        print(f"📊 Articles: {total_articles}\n📰 Sources: {', '.join(sources)}\n💭 Avg Sentiment: {avg_sentiment:.2f}\n🎯 Avg Relevance: {avg_relevance:.2f}\n{'='*80}")
        for i, article in enumerate(articles, 1):
            print(f"\n--- Article {i} ---\n📰 Title: {article.get('Title', 'No Title')[:100]}...\n📅 Date: {article.get('Date', 'Unknown')}\n🏢 Source: {article.get('Source', 'Unknown')}\n💭 Sentiment: {article.get('Sentiment_Score', 0):.2f}\n🎯 Relevance: {article.get('Relevance_Score', 0):.2f}\n📝 Words: {article.get('Word_Count', 0)}")
            if article.get('Key_Entities'):
                print(f"🔍 Entities: {article.get('Key_Entities')}")
            print(f"🔗 URL: {article.get('URL', 'No URL')}")
            if summary := article.get('Summary', ''):
                print(f"📄 Summary: {summary[:200]}...")

    def list_companies(self, limit: int = 20, search_term: str = ""):
        if self.companies_db.empty:
            print("❌ No companies in IPO database")
            return
        filtered_db = self.companies_db[
            self.companies_db.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)
        ] if search_term else self.companies_db
        print(f"\n{'='*80}\n{'COMPANIES MATCHING' if search_term else 'AVAILABLE IPO COMPANIES'} {'“' + search_term.upper() + '”' if search_term else ''} (showing first {limit})\n{'='*80}")
        if filtered_db.empty:
            print(f"❌ No companies found matching '{search_term}'")
            return
        for i, (_, row) in enumerate(filtered_db.head(limit).iterrows(), 1):
            company_name = self.get_company_name_from_info(row.to_dict())
            print(f"{i:2d}. {company_name}")
            if len(row) > 1:
                other_info = [f"{col}: {val}" for col, val in row.items() if col != company_name and pd.notna(val) and str(val).strip()]
                if other_info:
                    print(f"    {' | '.join(other_info[:3])}")
        if len(filtered_db) > limit:
            print(f"\n... and {len(filtered_db) - limit} more companies")

    def get_news_stats(self):
        if self.news_db.empty:
            print("❌ No news articles in database")
            return
        print(f"\n{'='*80}\nNEWS DATABASE STATISTICS\n{'='*80}")
        total_articles = len(self.news_db)
        unique_companies = self.news_db['Company_Name'].nunique()
        sources = self.news_db['Source'].unique()
        print(f"📊 Articles: {total_articles}\n🏢 Companies: {unique_companies}\n📰 Sources: {', '.join(sources)}")
        if 'Date' in self.news_db.columns and (dates := self.news_db['Date'].dropna()).any():
            print(f"📅 Date range: {dates.min()} to {dates.max()}")
        if 'Sentiment_Score' in self.news_db.columns and (scores := pd.to_numeric(self.news_db['Sentiment_Score'], errors='coerce').dropna()).any():
            avg_sentiment = scores.mean()
            print(f"💭 Avg sentiment: {avg_sentiment:.2f}\n😊 Positive: {len(scores[scores > 0.1])} ({len(scores[scores > 0.1])/total_articles*100:.1f}%)\n😐 Neutral: {len(scores[(scores >= -0.1) & (scores <= 0.1)])} ({len(scores[(scores >= -0.1) & (scores <= 0.1)])/total_articles*100:.1f}%)\n😞 Negative: {len(scores[scores < -0.1])} ({len(scores[scores < -0.1])/total_articles*100:.1f}%)")
        print(f"\n📈 Top companies:\n" + '\n'.join(f"  {company}: {count} articles" for company, count in self.news_db['Company_Name'].value_counts().head(10).items()))
        if 'Scrape_Date' in self.news_db.columns:
            print(f"\n🕒 Recent articles:\n" + '\n'.join(f"  {article.get('Title', 'No Title')[:60]}... ({article.get('Company_Name', 'Unknown')})" for _, article in self.news_db.sort_values('Scrape_Date', ascending=False).head(5).iterrows()))

    def export_company_news(self, company_name: str, output_file: str = None):
        if self.news_db.empty:
            print("❌ No news articles in database")
            return
        company_articles = self.news_db[self.news_db['Company_Name'].str.contains(company_name, case=False, na=False)]
        if company_articles.empty:
            print(f"❌ No articles for: {company_name}")
            return
        output_file = output_file or f"{re.sub(r'[^a-zA-Z0-9_]', '_', company_name.lower())}_news_export.csv"
        try:
            company_articles.to_csv(output_file, index=False)
            print(f"✅ Exported {len(company_articles)} articles to {output_file}")
        except Exception as e:
            print(f"❌ Error exporting: {e}")

    def cleanup(self):
        if self.driver:
            try:
                self.driver.quit()
                logging.info("WebDriver closed")
            except Exception as e:
                logging.error(f"Error closing WebDriver: {e}")
        try:
            self.session.close()
            logging.info("Requests session closed")
        except Exception as e:
            logging.error(f"Error closing session: {e}")

class EnhancedNLP:
    def __init__(self):
        self.positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'success', 'strong', 'rise', 'gain', 'bullish', 'optimistic', 'promising', 'robust', 'surge', 'boom', 'rally', 'upward', 'upgrade', 'outperform', 'beat', 'exceed', 'impressive', 'solid', 'healthy', 'momentum']
        self.negative_words = ['bad', 'poor', 'negative', 'decline', 'loss', 'weak', 'fall', 'drop', 'concern', 'risk', 'bearish', 'pessimistic', 'disappointing', 'vulnerable', 'plunge', 'crash', 'slump', 'downward', 'downgrade', 'underperform', 'miss', 'below', 'worse', 'trouble', 'challenge', 'pressure']
        self.business_keywords = ['ipo', 'shares', 'stock', 'market', 'business', 'company', 'financial', 'earnings', 'revenue', 'profit', 'investment', 'listing', 'trading', 'valuation', 'dividend', 'acquisition', 'merger', 'partnership', 'expansion', 'growth', 'performance', 'results', 'quarter']
        self.entity_patterns = {
            'money': r'₹\s*\d+(?:,\d{2,3})*(?:\.\d{2})?\s*(?:crore|lakh|billion|million|thousand)?',
            'percentage': r'\d+(?:\.\d+)?%',
            'dates': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'numbers': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        }

    def calculate_relevance(self, text: str, company_name: str) -> float:
        if not text or not company_name:
            return 0.0
        text_lower, company_lower = text.lower(), company_name.lower()
        score = 0.6 if company_lower in text_lower else 0.0
        company_words = [word for word in company_lower.split() if len(word) > 2]
        if company_words:
            score += (sum(1 for word in company_words if word in text_lower) / len(company_words)) * 0.3
        score += min(sum(1 for keyword in self.business_keywords if keyword in text_lower) * 0.02, 0.1)
        return min(score, 1.0)

    def calculate_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        positive_score = sum(2 if word in ['excellent', 'outstanding', 'exceptional'] else 1 for word in self.positive_words for _ in range(text_lower.count(word)))
        negative_score = sum(2 if word in ['terrible', 'awful', 'disastrous'] else 1 for word in self.negative_words for _ in range(text_lower.count(word)))
        total = positive_score + negative_score
        return (positive_score - negative_score) / total if total else 0.0

    def extract_entities(self, text: str) -> str:
        if not text:
            return ""
        entities = []
        for pattern in self.entity_patterns.values():
            if matches := re.findall(pattern, text, re.IGNORECASE):
                entities.extend(list(set(matches))[:3])
        return '; '.join(entities)

def signal_handler(sig, frame):
    print("\nCleaning up...")
    scraper.cleanup()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Company News Scraper")
    parser.add_argument("--company", help="Company name to scrape")
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--list-companies", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--export", help="Export news for a company")
    parser.add_argument("--output-file", help="Output file for export")
    args = parser.parse_args()
    global scraper
    scraper = CompanyNewsScraperAgent()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        if args.company:
            scraper.query_company_news(args.company, args.max_articles)
        elif args.list_companies:
            scraper.list_companies()
        elif args.stats:
            scraper.get_news_stats()
        elif args.export:
            scraper.export_company_news(args.export, args.output_file)
        else:
            print("No action specified. Use --help for usage.")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Error in main: {e}")
    finally:
        print("🔚 Program ended")