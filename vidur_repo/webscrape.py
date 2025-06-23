import pandas as pd
import logging
from datetime import datetime
import time
import os
import re
import hashlib
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib.parse import quote_plus, urljoin
from dotenv import load_dotenv
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CompactNewsScraper:
    def __init__(self):
        load_dotenv()
        self.ipo_csv = os.getenv("IPO_CSV_PATH", "/Users/shriya/Documents/GitHub/logo_detect/dsmlprojects222/List_IPO.csv")
        self.news_csv = os.getenv("NEWS_CSV_PATH", "company_news.csv")
        self.max_articles = int(os.getenv("MAX_ARTICLES", 10))
        self.deepseek_api_key = os.getenv("sk-2afc266809a24d1689cfb1e7c2abeb31")  # Add your DeepSeek API key
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'})
        self.companies_db = self.load_companies()
        self.news_db = self.load_news_db()
        self.failed_urls = set()
        
        # Simplified sources
        self.sources = {
            'moneycontrol': {'base': 'https://www.moneycontrol.com', 'search': 'https://www.moneycontrol.com/news/tags/{}.html'},
            'livemint': {'base': 'https://www.livemint.com', 'search': 'https://www.livemint.com/search?q={}'},
            'economictimes': {'base': 'https://economictimes.indiatimes.com', 'search': 'https://economictimes.indiatimes.com/topic/{}'}
        }

    def get_sentiment_label(self, score):
        """Convert sentiment score to descriptive label"""
        if score >= 0.6:
            return "Very Positive"
        elif score >= 0.2:
            return "Positive"
        elif score >= -0.2:
            return "Neutral"
        elif score >= -0.6:
            return "Negative"
        else:
            return "Very Negative"

    def load_companies(self):
        try:
            df = pd.read_csv(self.ipo_csv)
            col = next((c for c in ['Company Name', 'Company', 'Name'] if c in df.columns), df.columns[0])
            df[col] = df[col].astype(str).str.strip()
            return df[df[col].notna() & (df[col] != '')]
        except Exception as e:
            logging.error(f"Error loading companies: {e}")
            return pd.DataFrame()

    def load_news_db(self):
        if os.path.exists(self.news_csv):
            try:
                return pd.read_csv(self.news_csv).fillna('')
            except Exception:
                pass
        df = pd.DataFrame(columns=['Company', 'Title', 'Date', 'Content', 'URL', 'Source', 'Hash', 'Sentiment_Score', 'Sentiment_Label', 'Sentiment_Reason', 'Relevance'])
        df.to_csv(self.news_csv, index=False)
        return df

    def deepseek_sentiment_analysis(self, text):
        """Analyze sentiment using DeepSeek API with reasoning"""
        if not self.deepseek_api_key or not text:
            score = self.calculate_sentiment(text)
            label = self.get_sentiment_label(score)
            reason = "Basic keyword-based analysis (DeepSeek API not available)"
            return score, label, reason
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            # Truncate text to avoid token limits
            text_sample = text[:800] if len(text) > 800 else text
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a financial news sentiment analyst. Analyze the sentiment and provide: 1) A score between -1 (very negative) and 1 (very positive), 2) A brief reason for your analysis. Format: 'Score: X.XX | Reason: brief explanation'"
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze this financial news sentiment: {text_sample}"
                    }
                ],
                "max_tokens": 80,
                "temperature": 0.1
            }
            
            response = requests.post(self.deepseek_api_url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                # Parse score and reason
                score_match = re.search(r'Score:\s*(-?\d*\.?\d+)', response_text, re.IGNORECASE)
                reason_match = re.search(r'Reason:\s*(.+)', response_text, re.IGNORECASE)
                
                if score_match:
                    sentiment_score = max(-1, min(1, float(score_match.group(1))))
                    sentiment_label = self.get_sentiment_label(sentiment_score)
                    sentiment_reason = reason_match.group(1).strip() if reason_match else "Positive business indicators detected"
                    return sentiment_score, sentiment_label, sentiment_reason
            
            logging.warning(f"DeepSeek API error: {response.status_code}")
            
        except Exception as e:
            logging.error(f"DeepSeek sentiment analysis failed: {e}")
        
        # Fallback to basic sentiment analysis
        score = self.calculate_sentiment(text)
        label = self.get_sentiment_label(score)
        reason = "Fallback analysis due to API error"
        return score, label, reason

    def setup_driver(self):
        if self.driver:
            return
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-images")
            options.add_argument("--disable-javascript")
            options.add_argument("--disable-plugins")
            options.add_argument("--disable-java")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--disable-translate")
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument(f"--user-agent={self.session.headers['User-Agent']}")
            
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.page_load_strategy = 'eager'
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_page_load_timeout(15)
            self.driver.implicitly_wait(5)
            logging.info("WebDriver initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize WebDriver: {e}")
            self.driver = None

    def restart_driver(self):
        try:
            if self.driver:
                self.driver.quit()
                logging.info("WebDriver restarted")
        except Exception:
            pass
        finally:
            self.driver = None
            self.setup_driver()

    def find_company(self, query):
        if self.companies_db.empty:
            return None
        query = query.lower().strip()
        for _, row in self.companies_db.iterrows():
            if any(query in str(val).lower() for val in row.values if pd.notna(val)):
                return row.to_dict()
        return None

    def scrape_articles(self, company_name, max_articles=10):
        all_articles = []
        search_terms = [company_name, company_name.replace(' Ltd', '').replace(' Limited', '')]
        
        for source_name, config in self.sources.items():
            if len(all_articles) >= max_articles:
                break
                
            for term in search_terms[:2]:
                try:
                    url = config['search'].format(term.lower().replace(' ', '-'))
                    articles = self.scrape_source(url, source_name, company_name)
                    all_articles.extend(articles)
                    time.sleep(2)
                    if articles:
                        break
                except Exception as e:
                    logging.error(f"Error scraping {source_name}: {e}")
        
        return self.filter_articles(all_articles, company_name)[:max_articles]

    def scrape_source(self, url, source, company):
        articles = []
        soup = self.get_soup_requests(url)
        if not soup:
            soup = self.get_soup_selenium(url)
        
        if soup:
            links = self.extract_article_links(soup, source)
            for article_url in list(links)[:3]:
                if article_url in self.failed_urls:
                    continue
                    
                if article := self.scrape_article_with_retry(article_url, source, company):
                    articles.append(article)
                else:
                    self.failed_urls.add(article_url)
                
                time.sleep(1)
                    
        return articles

    def get_soup_requests(self, url):
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logging.debug(f"Requests failed for {url}: {e}")
        return None

    def get_soup_selenium(self, url):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    self.setup_driver()
                
                if not self.driver:
                    return None
                
                self.driver.get(url)
                WebDriverWait(self.driver, 8).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(1)
                return BeautifulSoup(self.driver.page_source, 'html.parser')
                
            except TimeoutException:
                logging.warning(f"Timeout on attempt {attempt + 1} for {url}")
                if attempt == max_retries - 1:
                    self.restart_driver()
                    return None
            except WebDriverException as e:
                logging.error(f"WebDriver error on attempt {attempt + 1} for {url}: {e}")
                self.restart_driver()
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                logging.error(f"Selenium failed for {url}: {e}")
                return None
        
        return None

    def scrape_article_with_retry(self, url, source, company, max_retries=2):
        for attempt in range(max_retries):
            try:
                article = self.scrape_article_requests(url, source, company)
                if article:
                    return article
                
                if attempt == max_retries - 1:
                    if 'm.economictimes.com' in url:
                        logging.warning(f"Skipping mobile ET URL: {url}")
                        return None
                    
                    article = self.scrape_article_selenium(url, source, company)
                    if article:
                        return article
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                time.sleep(2)
        
        return None

    def extract_article_links(self, soup, source):
        links = set()
        selectors = ['a[href*="/news/"]', 'a[href*="/article/"]', 'a[href*="/story/"]', 'a[href*="/business/"]', 'a[href*="/companies/"]']
        
        for selector in selectors:
            for link in soup.select(selector):
                if href := link.get('href'):
                    full_url = urljoin(self.sources[source]['base'], href)
                    if self.is_valid_url(full_url):
                        if 'm.economictimes.com' not in full_url:
                            links.add(full_url)
        return links

    def scrape_article(self, url, source, company):
        article = self.scrape_article_requests(url, source, company)
        if not article:
            article = self.scrape_article_selenium(url, source, company)
        return article

    def scrape_article_requests(self, url, source, company):
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                return self.extract_article_data(soup, url, source, company)
        except Exception as e:
            logging.debug(f"Requests failed for article {url}: {e}")
        return None

    def scrape_article_selenium(self, url, source, company):
        try:
            if not self.driver:
                self.setup_driver()
            
            if not self.driver:
                return None
            
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            return self.extract_article_data(soup, url, source, company)
            
        except TimeoutException:
            logging.error(f"Timeout scraping article: {url}")
            self.restart_driver()
            return None
        except WebDriverException as e:
            logging.error(f"WebDriver error scraping article {url}: {e}")
            self.restart_driver()
            return None
        except Exception as e:
            logging.error(f"Selenium failed for article {url}: {e}")
            return None

    def extract_article_data(self, soup, url, source, company):
        try:
            title = self.extract_text(soup, ['h1', 'title', '.headline', '.story-headline', '.entry-title'])
            if not title or len(title) < 10:
                return None
                
            date = self.extract_text(soup, ['time', '.date', '[datetime]', '.publish-date', '.story-date'])
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
                
            content_parts = []
            for p in soup.select('p'):
                text = p.get_text().strip()
                if len(text) > 20 and not any(skip in text.lower() for skip in ['subscribe', 'advertisement', 'click here']):
                    content_parts.append(text)
            content = ' '.join(content_parts[:10])
            
            if len(content) < 50:
                return None
            
            article_hash = hashlib.md5(f"{title}{url}".encode()).hexdigest()
            relevance = self.calculate_relevance(f"{title} {content}", company)
            
            if self.is_duplicate(article_hash) or relevance < 0.2:
                return None
            
            # Use DeepSeek for sentiment analysis
            sentiment_score, sentiment_label, sentiment_reason = self.deepseek_sentiment_analysis(f"{title} {content}")
            
            return {
                'Company': company,
                'Title': title,
                'Date': date,
                'Content': content,
                'URL': url,
                'Source': source,
                'Hash': article_hash,
                'Sentiment_Score': sentiment_score,
                'Sentiment_Label': sentiment_label,
                'Sentiment_Reason': sentiment_reason,
                'Relevance': relevance
            }
            
        except Exception as e:
            logging.error(f"Error extracting article data from {url}: {e}")
            return None

    def extract_text(self, soup, selectors):
        for selector in selectors:
            try:
                if elem := soup.select_one(selector):
                    if selector == 'time':
                        datetime_val = elem.get('datetime') or elem.get('content')
                        if datetime_val:
                            return datetime_val
                    if text := elem.get_text().strip():
                        return text
            except Exception:
                continue
        return ""

    def is_valid_url(self, url):
        return (url.startswith('http') and 
                any(pattern in url.lower() for pattern in ['news', 'article', 'story', 'business']) and
                not any(skip in url.lower() for skip in ['login', 'register', 'subscribe', 'video', 'live-blog']))

    def is_duplicate(self, article_hash):
        return article_hash in self.news_db['Hash'].values if not self.news_db.empty else False

    def filter_articles(self, articles, company):
        seen = set()
        filtered = []
        for article in articles:
            hash_val = article.get('Hash')
            if hash_val and hash_val not in seen and len(article.get('Content', '')) > 50:
                filtered.append(article)
                seen.add(hash_val)
        return sorted(filtered, key=lambda x: x.get('Relevance', 0), reverse=True)

    def calculate_relevance(self, text, company):
        if not text or not company:
            return 0.0
        text_lower = text.lower()
        company_lower = company.lower()
        score = 0.6 if company_lower in text_lower else 0.0
        business_words = ['ipo', 'stock', 'share', 'market', 'business', 'earnings', 'financial']
        score += min(sum(1 for word in business_words if word in text_lower) * 0.05, 0.4)
        return min(score, 1.0)

    def calculate_sentiment(self, text):
        """Fallback basic sentiment analysis"""
        if not text:
            return 0.0
        text_lower = text.lower()
        positive = ['good', 'great', 'growth', 'profit', 'strong', 'rise', 'gain', 'positive', 'up', 'high', 'success', 'premium', 'increase']
        negative = ['bad', 'poor', 'decline', 'loss', 'weak', 'fall', 'drop', 'negative', 'down', 'low', 'fail', 'decrease']
        pos_score = sum(text_lower.count(word) for word in positive)
        neg_score = sum(text_lower.count(word) for word in negative)
        total = pos_score + neg_score
        return (pos_score - neg_score) / total if total else 0.0

    def save_articles(self, articles):
        if not articles:
            return
        try:
            new_df = pd.DataFrame(articles)
            combined = pd.concat([self.news_db, new_df], ignore_index=True) if not self.news_db.empty else new_df
            combined = combined.drop_duplicates(subset=['Hash'])
            combined.to_csv(self.news_csv, index=False)
            self.news_db = combined
            logging.info(f"Saved {len(articles)} articles")
        except Exception as e:
            logging.error(f"Error saving: {e}")

    def query_news(self, company_query, max_articles=10):
        company_info = self.find_company(company_query)
        if not company_info:
            print(f"❌ Company '{company_query}' not found")
            return False
        
        company_name = next((str(v).strip() for v in company_info.values() if isinstance(v, str) and v.strip()), "Unknown")
        print(f"✅ Found: {company_name}")
        
        articles = self.scrape_articles(company_name, max_articles)
        if articles:
            self.save_articles(articles)
            print(f" Scraped {len(articles)} articles")
            self.show_summary(articles)
            return True
        else:
            print(" No articles found")
            return False

    def show_summary(self, articles):
        print(f"\n{'='*60}")
        for i, article in enumerate(articles, 1):
            sentiment_score = article.get('Sentiment_Score', 0)
            sentiment_label = article.get('Sentiment_Label', 'Unknown')
            sentiment_reason = article.get('Sentiment_Reason', 'No reason provided')
            
            sentiment_emoji = "😊" if sentiment_score > 0.2 else "😐" if sentiment_score > -0.2 else "😞"
            print(f"{i}. {article['Title'][:80]}...")
            print(f"   {article['Date']} | 🏢 {article['Source']}")
            print(f"   {sentiment_emoji} {sentiment_label} ({sentiment_score:.2f}) - {sentiment_reason}")
            print(f"   {article['URL']}")
            print()

    def list_companies(self, limit=20):
        if self.companies_db.empty:
            print("No companies found")
            return
        print(f"\n{'='*60}\nAVAILABLE COMPANIES (first {limit})\n{'='*60}")
        for i, (_, row) in enumerate(self.companies_db.head(limit).iterrows(), 1):
            company = next((str(v).strip() for v in row.values if isinstance(v, str) and v.strip()), "Unknown")
            print(f"{i:2d}. {company}")

    def get_stats(self):
        if self.news_db.empty:
            print(" No articles in database")
            return
        print(f"\n{'='*60}\nNEWS DATABASE STATS\n{'='*60}")
        print(f"Total articles: {len(self.news_db)}")
        print(f"Companies: {self.news_db['Company'].nunique()}")
        print(f"Sources: {', '.join(self.news_db['Source'].unique())}")
        
        # Handle both old and new column names
        sentiment_col = 'Sentiment_Score' if 'Sentiment_Score' in self.news_db.columns else 'Sentiment'
        
        if sentiment_col in self.news_db.columns:
            avg_sentiment = pd.to_numeric(self.news_db[sentiment_col], errors='coerce').mean()
            print(f"💭 Average sentiment: {avg_sentiment:.2f}")
            
            # Sentiment distribution
            sentiment_scores = pd.to_numeric(self.news_db[sentiment_col], errors='coerce')
            positive = len(sentiment_scores[sentiment_scores > 0.2])
            neutral = len(sentiment_scores[sentiment_scores.between(-0.2, 0.2)])
            negative = len(sentiment_scores[sentiment_scores < -0.2])
            print(f"Positive: {positive} | Neutral: {neutral} | Negative: {negative}")
            
            # Show sentiment labels if available
            if 'Sentiment_Label' in self.news_db.columns:
                label_counts = self.news_db['Sentiment_Label'].value_counts()
                print(f"Sentiment breakdown: {dict(label_counts)}")

    def cleanup(self):
        try:
            if self.driver:
                self.driver.quit()
                logging.info("WebDriver closed")
        except Exception as e:
            logging.error(f"Error closing WebDriver: {e}")
        
        try:
            self.session.close()
            logging.info("Session closed")
        except Exception as e:
            logging.error(f"Error closing session: {e}")
