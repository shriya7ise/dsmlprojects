from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import json
import httpx
from bs4 import BeautifulSoup

load_dotenv()
mcp = FastMCP("docs")

User_agent = "docs-app/1.0"
serper_url = "https://google.serper.dev/search"

docs_urls = {
    "langchain": "python.langchain.com/docs",
    "openai": "platform.openai.com/docs",
    "llama_index": "docs.llamaindex.ai/en/stable",
}

async def search_web(query: str)->dict | None:
    payload = json.dumps ({"q": query, "num":2})
    
    headers = {
    "X-API-KEY": os.getenv ("SERPER_API_KEY"),
    "Content-Type":"application/json",
    }
    
    async with httpx.AsyncClient () as client:
        try:
            response = await client.post(
                serper_url, headers=headers, data=payload, timeout=30.0
            )
            response. raise_for_status ()
            return response. json ()
        except httpx. TimeoutException:
            return {"organic": []}
        
        
async def fetch_url(url: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            soup = BeautifulSoup (response. text, "html.parser")
            text = soup.get_text ()
            return  text 
        except httpx. TimebutException:
            return "Timeout error"
    
@mcp.tool()
async def get_docs(query: str, library: str):
    """
    Search the docs for a given query and library.
    Supports langchain, openai, and llama-index.

    Args:
        query: The query to search for (e.g. "Chroma DB*)
        library: The library to search in (e.g. "Langchain")

    Returns:
        List of dictionaries containing source URLs and extracted text
    """
    if library not in docs_urls:
        raise ValueError(f"Library {library} not supported. Supported libraries are: {', '.join(docs_urls.keys())}")
    
    query = f"site:{docs_urls[library]} {query}"
    results = await search_web(query)
    print("SERPER RESPONSE:", results)

    if len(results["organic"]) == 0:
        return "No results found"
    
    
    entries = []
    for result in results["organic"]:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        
        entries.append(f"📄 **{title}**\n\n{snippet}\n\n🔗 [Link]({link})\n\n---")
    
    return "\n".join(entries)

    text =""
    for result in results["organic"]:
        url = result["link"]
        text += await fetch_url(url)
    return text



if __name__ == "__main__":
    mcp.run(transport = "stdio")
