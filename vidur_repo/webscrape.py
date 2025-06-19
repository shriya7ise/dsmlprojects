# Alternative 1: Use News APIs instead of web scraping
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import json

class IPONewsAPI:
    def __init__(self):
        # Free news APIs (replace with your API keys)
        self.news_apis = {
            'newsapi': {
                'key': '89d1452cc28b48f091975f77b1a28e90',  # Get from newsapi.org
                'base_url': 'https://newsapi.org/v2/everything',
                'params': {'pageSize': 5, 'sortBy': 'publishedAt'}
            },
            'gnews': {
                'key': 'aa6852910b0bb6186ea46ed339ccaa63',  # Get from gnews.io
                'base_url': 'https://gnews.io/api/v4/search',
                'params': {'max': 5, 'lang': 'en'}
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def search_news_api(self, company: str, api_name: str = 'newsapi') -> List[Dict]:
        """Search for IPO news using news APIs"""
        api_config = self.news_apis.get(api_name)
        if not api_config:
            return []
        
        articles = []
        queries = [
            f"{company} IPO",
            f"{company} initial public offering",
            f"{company} stock market listing"
        ]
        
        for query in queries:
            try:
                params = {
                    'q': query,
                    'apiKey' if api_name == 'newsapi' else 'token': api_config['key'],
                    **api_config['params']
                }
                
                if api_name == 'newsapi':
                    params['from'] = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                response = requests.get(api_config['base_url'], params=params)
                response.raise_for_status()
                data = response.json()
                
                if api_name == 'newsapi':
                    for article in data.get('articles', []):
                        articles.append({
                            'company': company,
                            'title': article.get('title', ''),
                            'publish_date': article.get('publishedAt', ''),
                            'content': article.get('content', ''),
                            'summary': article.get('description', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'scraped_at': datetime.now().isoformat()
                        })
                
                elif api_name == 'gnews':
                    for article in data.get('articles', []):
                        articles.append({
                            'company': company,
                            'title': article.get('title', ''),
                            'publish_date': article.get('publishedAt', ''),
                            'content': article.get('content', ''),
                            'summary': article.get('description', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'scraped_at': datetime.now().isoformat()
                        })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error fetching news for {company} via {api_name}: {e}")
        
        return articles[:5]  # Limit results
    
    def get_company_news(self, companies: List[str]) -> pd.DataFrame:
        """Get news for multiple companies using APIs"""
        all_articles = []
        
        for company in companies:
            self.logger.info(f"Fetching news for: {company}")
            articles = self.search_news_api(company)
            all_articles.extend(articles)
            time.sleep(2)  # Be respectful to APIs
        
        return pd.DataFrame(all_articles)

# Alternative 2: Use RSS feeds from financial websites
import feedparser

class IPORSSFeedReader:
    def __init__(self):
        # RSS feeds from major financial news sites
        self.rss_feeds = [
            'https://economictimes.indiatimes.com/markets/ipo/rssfeeds/13357785.cms',
            'https://www.business-standard.com/rss/markets-ipo-106.rss',
            'https://www.moneycontrol.com/rss/ipo.xml',
            'https://feeds.feedburner.com/ndtvprofit-latest'
        ]
    
    def fetch_rss_news(self, company: str) -> List[Dict]:
        """Fetch IPO news from RSS feeds"""
        articles = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    # Check if company is mentioned
                    if company.lower() in title.lower() or company.lower() in summary.lower():
                        articles.append({
                            'company': company,
                            'title': title,
                            'publish_date': entry.get('published', ''),
                            'summary': summary,
                            'url': entry.get('link', ''),
                            'source': feed.feed.get('title', 'RSS Feed'),
                            'scraped_at': datetime.now().isoformat()
                        })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logging.error(f"Error fetching RSS feed {feed_url}: {e}")
        
        return articles

# Alternative 3: Use Google Custom Search API
class GoogleSearchAPI:
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search_ipo_news(self, company: str) -> List[Dict]:
        """Search for IPO news using Google Custom Search API"""
        articles = []
        queries = [
            f"{company} IPO news",
            f"{company} initial public offering date",
            f"{company} stock market debut"
        ]
        
        for query in queries:
            try:
                params = {
                    'key': self.api_key,
                    'cx': self.search_engine_id,
                    'q': query,
                    'num': 3,
                    'dateRestrict': 'm1'  # Last month
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get('items', []):
                    articles.append({
                        'company': company,
                        'title': item.get('title', ''),
                        'summary': item.get('snippet', ''),
                        'url': item.get('link', ''),
                        'source': item.get('displayLink', ''),
                        'scraped_at': datetime.now().isoformat()
                    })
                
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Google search error for {company}: {e}")
        
        return articles

# Alternative 4: Use Selenium with proxy rotation
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SeleniumScraper:
    def __init__(self, use_proxy: bool = False):
        self.use_proxy = use_proxy
        self.proxies = [
            # Add proxy servers here
            # "http://proxy1:port",
            # "http://proxy2:port"
        ]
    
    def create_driver(self, proxy: str = None):
        """Create a Chrome driver with options"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        if proxy:
            options.add_argument(f'--proxy-server={proxy}')
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    def scrape_with_selenium(self, url: str, company: str) -> Optional[Dict]:
        """Scrape using Selenium with JavaScript rendering"""
        driver = None
        try:
            proxy = None
            if self.use_proxy and self.proxies:
                proxy = self.proxies[0]  # Use first proxy
            
            driver = self.create_driver(proxy)
            driver.get(url)
            
            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            title = driver.title
            content = driver.find_element(By.TAG_NAME, "body").text
            
            return {
                'company': company,
                'title': title,
                'content': content[:1000],  # Limit content
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Selenium scraping error for {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()

# Alternative 5: Manual data entry with GUI
import tkinter as tk
from tkinter import ttk, messagebox
import csv

class ManualDataEntry:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IPO News Manual Entry")
        self.root.geometry("800x600")
        
        self.data = []
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI for manual data entry"""
        # Company dropdown
        tk.Label(self.root, text="Company:").pack(pady=5)
        self.company_var = tk.StringVar()
        self.company_combo = ttk.Combobox(self.root, textvariable=self.company_var, width=50)
        self.company_combo.pack(pady=5)
        
        # Title entry
        tk.Label(self.root, text="News Title:").pack(pady=5)
        self.title_entry = tk.Text(self.root, height=2, width=80)
        self.title_entry.pack(pady=5)
        
        # Summary entry
        tk.Label(self.root, text="Summary:").pack(pady=5)
        self.summary_entry = tk.Text(self.root, height=5, width=80)
        self.summary_entry.pack(pady=5)
        
        # URL entry
        tk.Label(self.root, text="URL:").pack(pady=5)
        self.url_entry = tk.Entry(self.root, width=80)
        self.url_entry.pack(pady=5)
        
        # Source entry
        tk.Label(self.root, text="Source:").pack(pady=5)
        self.source_entry = tk.Entry(self.root, width=80)
        self.source_entry.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Add Entry", command=self.add_entry).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear", command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text=f"Entries: {len(self.data)}")
        self.status_label.pack(pady=10)
    
    def load_companies(self, companies: List[str]):
        """Load company list into dropdown"""
        self.company_combo['values'] = companies
    
    def add_entry(self):
        """Add entry to data list"""
        entry = {
            'company': self.company_var.get(),
            'title': self.title_entry.get("1.0", tk.END).strip(),
            'summary': self.summary_entry.get("1.0", tk.END).strip(),
            'url': self.url_entry.get(),
            'source': self.source_entry.get(),
            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if entry['company'] and entry['title']:
            self.data.append(entry)
            self.clear_form()
            self.status_label.config(text=f"Entries: {len(self.data)}")
            messagebox.showinfo("Success", "Entry added successfully!")
        else:
            messagebox.showwarning("Warning", "Company and Title are required!")
    
    def clear_form(self):
        """Clear all form fields"""
        self.company_var.set("")
        self.title_entry.delete("1.0", tk.END)
        self.summary_entry.delete("1.0", tk.END)
        self.url_entry.delete(0, tk.END)
        self.source_entry.delete(0, tk.END)
    
    def save_csv(self):
        """Save data to CSV file"""
        if not self.data:
            messagebox.showwarning("Warning", "No data to save!")
            return
        
        filename = f"manual_ipo_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            if self.data:
                writer = csv.DictWriter(file, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerows(self.data)
        
        messagebox.showinfo("Success", f"Data saved to {filename}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

# Usage example combining multiple approaches
def main_alternative():
    """Main function using alternative approaches"""
    # Load companies from CSV
    try:
        df_companies = pd.read_csv("List_IPO.csv")
        companies = df_companies['COMPANY NAME'].dropna().tolist()
    except:
        companies = ["Example Company 1", "Example Company 2"]  # Fallback
    
    print("Available alternatives:")
    print("1. News APIs (NewsAPI, GNews)")
    print("2. RSS Feeds")
    print("3. Google Custom Search API")
    print("4. Selenium with proxy")
    print("5. Manual data entry GUI")
    
    choice = input("Choose an alternative (1-5): ")
    
    if choice == "1":
        # Use News APIs
        news_api = IPONewsAPI()
        df = news_api.get_company_news(companies)
        print(f"Found {len(df)} articles using News APIs")
        
    elif choice == "2":
        # Use RSS feeds
        rss_reader = IPORSSFeedReader()
        all_articles = []
        for company in companies:
            articles = rss_reader.fetch_rss_news(company)
            all_articles.extend(articles)
        df = pd.DataFrame(all_articles)
        print(f"Found {len(df)} articles using RSS feeds")
        
    elif choice == "3":
        # Use Google Custom Search
        api_key = input("Enter Google API key: ")
        search_engine_id = input("Enter Custom Search Engine ID: ")
        google_search = GoogleSearchAPI(api_key, search_engine_id)
        all_articles = []
        for company in companies:
            articles = google_search.search_ipo_news(company)
            all_articles.extend(articles)
        df = pd.DataFrame(all_articles)
        print(f"Found {len(df)} articles using Google Search")
        
    elif choice == "4":
        # Use Selenium
        print("Note: You'll need to install Chrome and ChromeDriver")
        selenium_scraper = SeleniumScraper()
        # Implementation would require specific URLs
        print("Selenium scraper initialized")
        
    elif choice == "5":
        # Manual data entry
        manual_entry = ManualDataEntry()
        manual_entry.load_companies(companies)
        manual_entry.run()
        return
    
    else:
        print("Invalid choice")
        return
    
    # Save results
    if 'df' in locals() and not df.empty:
        output_file = f"alternative_ipo_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to: {output_file}")

if __name__ == "__main__":
    main_alternative()