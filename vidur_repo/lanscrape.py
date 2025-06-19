from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import spacy
import logging
from datetime import datetime, timezone, timedelta
import time
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# State definition
class AgentState(TypedDict):
    sources: List[str]
    articles: List[Dict]
    visited_urls: set
    current_source: Optional[str]
    data: List[Dict]
    driver: Optional[webdriver.Chrome]
    next_action: Optional[str]
    has_searched: bool

# Gemini 1.5 Flash integration
from transformers import pipeline

class HuggingFaceLLM:
    def __init__(self):
        self.ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        self.relevance_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def analyze_relevance(self, text: str):
        labels = ["IPO news", "Not related"]
        result = self.relevance_classifier(text, candidate_labels=labels)
        score = result['scores'][0] if result['labels'][0] == "IPO news" else result['scores'][1]
        return {
            "relevance_score": float(score),
            "reason": f"Classified as {result['labels'][0]}"
        }

    def extract_entities(self, text: str):
        entities = self.ner(text[:500])  # limit for speed
        orgs = [ent['word'] for ent in entities if ent['entity_group'] == "ORG"]
        dates = [ent['word'] for ent in entities if ent['entity_group'] == "DATE"]
        money = [ent['word'] for ent in entities if ent['entity_group'] == "MONEY"]
        return {
            "companies": "; ".join(orgs[:3]),
            "ipo_dates": "; ".join(dates[:2]),
            "ipo_amounts": "; ".join(money[:2])
        }


# Selenium setup
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    try:
        driver = webdriver.Chrome(options=chrome_options)
        logging.info("Selenium WebDriver initialized successfully")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize WebDriver: {e}")
        raise

# NLP fallback
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("spaCy model not found, install with: python -m spacy download en_core_web_sm")
    nlp = None

# Search Node
def search_sources(state: AgentState) -> AgentState:
    logging.info("Searching for IPO sources...")
    driver = state["driver"]
    
    # Use a more reliable search approach
    search_queries = [
        "https://www.google.com/search?q=ipo+news+2025+india",
        "https://www.moneycontrol.com/ipo/",
        "https://www.livemint.com/market/ipo"
    ]
    
    sources = []
    
    for search_url in search_queries:
        try:
            if "google.com" in search_url:
                driver.get(search_url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                )
                results = driver.find_elements(By.CSS_SELECTOR, "div.g a")[:3]
                for result in results:
                    href = result.get_attribute("href")
                    if href and "http" in href and "google" not in href and href not in sources:
                        sources.append(href)
            else:
                # Direct sources
                sources.append(search_url)
        except Exception as e:
            logging.error(f"Search error for {search_url}: {e}")
            continue
    
    # Fallback sources if search fails
    if not sources:
        sources = [
            "https://www.moneycontrol.com/ipo/",
            "https://www.livemint.com/market/ipo",
            "https://economictimes.indiatimes.com/markets/ipo"
        ]
    
    state["sources"] = sources
    state["current_source"] = sources[0] if sources else None
    state["next_action"] = "scrape" if sources else "end"
    state["has_searched"] = True
    
    logging.info(f"Found {len(sources)} sources to scrape")
    return state

# Scrape Node
def scrape_articles(state: AgentState) -> AgentState:
    current_source = state["current_source"]
    
    # Validate current_source before proceeding
    if not current_source or not isinstance(current_source, str):
        logging.error(f"Invalid current_source: {current_source}")
        state["next_action"] = "supervisor"
        return state
    
    logging.info(f"Scraping articles from {current_source}")
    driver = state["driver"]
    articles = []
    
    try:
        driver.get(current_source)
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a"))
        )
        
        # Different selectors for different sites
        if "moneycontrol.com" in current_source:
            article_selectors = ["a[href*='/news/business/ipo']", "a[href*='/ipo/']"]
        elif "livemint.com" in current_source:
            article_selectors = ["a[href*='/market/ipo']", "a[href*='/companies']"]
        elif "economictimes.indiatimes.com" in current_source:
            article_selectors = ["a[href*='/markets/ipo']", "a[href*='/news/market']"]
        else:
            article_selectors = ["a"]
        
        article_links = []
        for selector in article_selectors:
            try:
                links = driver.find_elements(By.CSS_SELECTOR, selector)
                for link in links[:5]:  # Limit to avoid too many requests
                    href = link.get_attribute("href")
                    if href and href not in state["visited_urls"] and href not in article_links:
                        article_links.append(href)
                if article_links:
                    break
            except Exception as e:
                logging.warning(f"Selector {selector} failed: {e}")
                continue
        
        # Fallback: get any links with news-related keywords
        if not article_links:
            all_links = driver.find_elements(By.CSS_SELECTOR, "a")
            for link in all_links[:10]:
                href = link.get_attribute("href")
                if href and any(keyword in href.lower() for keyword in ['ipo', 'news', 'market', 'company']):
                    if href not in state["visited_urls"] and href not in article_links:
                        article_links.append(href)
        
        logging.info(f"Found {len(article_links)} article links to process")
        
        for article_url in article_links[:3]:  # Limit to 3 articles per source
            try:
                # Open new tab for each article
                driver.execute_script("window.open('');")
                driver.switch_to.window(driver.window_handles[-1])
                driver.get(article_url)
                
                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                title = driver.title.strip()
                
                # Extract date
                date_text = ""
                date_selectors = ["time", "span.date", "div.date", "[class*='date']", "[class*='time']"]
                for selector in date_selectors:
                    try:
                        date_element = driver.find_element(By.CSS_SELECTOR, selector)
                        date_text = date_element.text.strip() or date_element.get_attribute("datetime") or ""
                        if date_text:
                            break
                    except:
                        continue
                
                # Extract summary
                summary = ""
                try:
                    summary_element = driver.find_element(By.CSS_SELECTOR, "meta[name='description']")
                    summary = summary_element.get_attribute("content").strip()
                except:
                    pass
                
                # Extract content
                content = ""
                content_selectors = ["article p", "div.content p", ".story-content p", "[class*='article'] p", "p"]
                for selector in content_selectors:
                    try:
                        content_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        content_texts = [elem.text.strip() for elem in content_elements if elem.text.strip()]
                        if content_texts:
                            content = " ".join(content_texts[:10])  # Limit to first 10 paragraphs
                            break
                    except:
                        continue
                
                # Only add if we have meaningful content
                if title and (summary or content):
                    articles.append({
                        "title": title,
                        "date": date_text,
                        "summary": summary,
                        "content": content[:1000],  # Limit content length
                        "url": article_url
                    })
                    state["visited_urls"].add(article_url)
                    logging.info(f"Successfully scraped: {title[:50]}...")
                
                # Close current tab
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(2)  # Be respectful to the server
                
            except Exception as e:
                logging.error(f"Scrape error for {article_url}: {e}")
                # Ensure we close any opened tabs
                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                continue
        
    except Exception as e:
        logging.error(f"Scrape error for {current_source}: {e}")
    
    state["articles"] = articles
    state["next_action"] = "analyze" if articles else "supervisor"
    logging.info(f"Scraped {len(articles)} articles from {current_source}")
    return state

# Analyze Node
def analyze_articles(state: AgentState) -> AgentState:
    logging.info(f"Analyzing {len(state['articles'])} articles...")
    llm = HuggingFaceLLM()
    relevance_prompt = ChatPromptTemplate.from_template(
        "Classify the relevance of this article for IPO news (0-1 scale). Provide a score and reason. Article summary: {summary}"
    )
    entity_prompt = ChatPromptTemplate.from_template(
        "Extract IPO-related entities from this text. Return companies, IPO dates, and IPO amounts. Text: {text}"
    )
    
    ist = timezone(timedelta(hours=5, minutes=30))
    
    for article in state["articles"]:
        try:
            # Relevance classification
            text_to_analyze = article["summary"] or article["title"]
            relevance_response = llm.invoke(relevance_prompt.format(summary=text_to_analyze))
            relevance_score = relevance_response.get("relevance_score", 0.5)
            
            # Skip low relevance articles
            if relevance_score < 0.3:
                logging.info(f"Skipping low-relevance article: {article['title'][:50]}... (score: {relevance_score})")
                continue
            
            # Entity extraction
            content_to_extract = article["content"] or article["summary"] or article["title"]
            entity_response = llm.invoke(entity_prompt.format(text=content_to_extract))
            
            # Fallback to spaCy if available and LLM fails
            if "error" in entity_response and nlp:
                doc = nlp(content_to_extract)
                entities = {
                    "companies": ";".join([ent.text for ent in doc.ents if ent.label_ == "ORG"][:5]),
                    "ipo_dates": ";".join([ent.text for ent in doc.ents if ent.label_ == "DATE"][:3]),
                    "ipo_amounts": ";".join([ent.text for ent in doc.ents if ent.label_ == "MONEY"][:3])
                }
            else:
                entities = entity_response
            
            state["data"].append({
                "Title": article["title"],
                "Date": article["date"],
                "Summary": article["summary"],
                "Content": article["content"],
                "URL": article["url"],
                "Companies": entities.get("companies", ""),
                "IPO_Dates": entities.get("ipo_dates", ""),
                "IPO_Amounts": entities.get("ipo_amounts", ""),
                "Relevance_Score": relevance_score,
                "Scrape_Date": datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            logging.error(f"Analysis error for {article['title']}: {e}")
            continue
    
    state["next_action"] = "save"
    logging.info(f"Analyzed articles, {len(state['data'])} entries added to data")
    return state

# Save Node
def save_data(state: AgentState) -> AgentState:
    if state["data"]:
        df = pd.DataFrame(state["data"])
        filename = f"ipo_news_langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        logging.info(f"Saved {len(state['data'])} articles to {filename}")
    else:
        logging.warning("No data to save")
    
    state["next_action"] = "supervisor"
    return state

# Supervisor Node
def supervisor(state: AgentState) -> AgentState:
    logging.info(f"Supervisor evaluating state - Sources remaining: {len(state['sources'])}, Has searched: {state.get('has_searched', False)}")
    
    # Initialize driver if not already set
    if state["driver"] is None:
        try:
            state["driver"] = setup_driver()
        except Exception as e:
            logging.error(f"Failed to initialize driver in supervisor: {e}")
            state["next_action"] = "end"
            return state
    
    # If no sources and haven't searched yet, trigger search
    if not state["sources"] and not state.get("has_searched", False):
        state["next_action"] = "search"
        return state
    
    # If we have sources to process, continue with next source
    if state["sources"]:
        # Remove the current source from the list and set the next one
        if state["current_source"] in state["sources"]:
            state["sources"].remove(state["current_source"])
        
        if state["sources"]:
            state["current_source"] = state["sources"][0]
            state["next_action"] = "scrape"
            logging.info(f"Moving to next source: {state['current_source']}")
        else:
            # No more sources, end the workflow
            state["current_source"] = None
            state["next_action"] = "end"
            logging.info("All sources processed, ending workflow")
    else:
        # No sources left, end the workflow
        state["next_action"] = "end"
        logging.info("No more sources to process, ending workflow")
    
    # Cleanup driver if ending
    if state["next_action"] == "end" and state["driver"]:
        try:
            state["driver"].quit()
            logging.info("WebDriver closed successfully")
        except Exception as e:
            logging.error(f"Error closing WebDriver: {e}")
        state["driver"] = None
    
    return state

# Build LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("search", search_sources)
workflow.add_node("scrape", scrape_articles)
workflow.add_node("analyze", analyze_articles)
workflow.add_node("save", save_data)
workflow.add_node("supervisor", supervisor)

workflow.add_edge("search", "supervisor")
workflow.add_edge("scrape", "analyze")
workflow.add_edge("analyze", "save")
workflow.add_edge("save", "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next_action", "end"),
    {
        "search": "search",
        "scrape": "scrape",
        "end": END
    }
)

workflow.set_entry_point("supervisor")

# Compile and run
app = workflow.compile()

# Initial state
initial_state = {
    "sources": [],
    "articles": [],
    "visited_urls": set(),
    "current_source": None,
    "data": [],
    "driver": None,
    "next_action": "search",
    "has_searched": False
}

# Run the agent
if __name__ == "__main__":
    try:
        logging.info("Starting IPO News Scraper Agent...")
        result = app.invoke(initial_state)
        logging.info("Agent execution completed successfully")
        
        if result and result.get("data"):
            print(f"\n=== SCRAPING SUMMARY ===")
            print(f"Total articles scraped: {len(result['data'])}")
            print(f"Sources processed: {len(result.get('visited_urls', set()))}")
            print(f"\n=== IPO NEWS ===")
            for i, article in enumerate(result["data"], start=1):
                print(f"\n--- Article {i} ---")
                print(f"Title       : {article['Title']}")
                print(f"Date        : {article['Date']}")
                print(f"Companies   : {article['Companies']}")
                print(f"IPO Dates   : {article['IPO_Dates']}")
                print(f"Amount      : {article['IPO_Amounts']}")
                print(f"Summary     : {article['Summary']}")
                print(f"Content     : {article['Content'][:300]}...")  # First 300 chars
                print(f"URL         : {article['URL']}")
                print(f"Relevance   : {article['Relevance_Score']}")
        else:
            print("No data was collected")

            
    except Exception as e:
        logging.error(f"Agent execution failed: {e}")
        # Cleanup driver if it exists
        if initial_state.get("driver"):
            try:
                initial_state["driver"].quit()
                logging.info("WebDriver closed in cleanup")
            except:
                pass