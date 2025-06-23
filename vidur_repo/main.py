import argparse
import sys
from webscrape import CompactNewsScraper

def main():
    parser = argparse.ArgumentParser(description="News Scraper for Company Analysis")
    parser.add_argument('--query', type=str, help="Query news for a specific company")
    parser.add_argument('--list-companies', action='store_true', help="List available companies")
    parser.add_argument('--stats', action='store_true', help="Show database statistics")
    parser.add_argument('--max-articles', type=int, default=10, help="Maximum articles to scrape per company")
    parser.add_argument('--limit', type=int, default=20, help="Limit for listing companies")
    
    args = parser.parse_args()
    
    scraper = CompactNewsScraper()
    
    try:
        if args.list_companies:
            scraper.list_companies(limit=args.limit)
        
        elif args.stats:
            scraper.get_stats()
        
        elif args.query:
            success = scraper.query_news(args.query, max_articles=args.max_articles)
            if not success:
                sys.exit(1)
        
        else:
            parser.print_help()
            sys.exit(0)
            
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()