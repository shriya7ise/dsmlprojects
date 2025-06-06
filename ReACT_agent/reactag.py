from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import os
import datetime

load_dotenv
print("Tavily API Key:", os.getenv("TAVILY_API_KEY"))

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
search_tool= TavilySearchResults(search_depth = "basic")

@tool
def get_current_date(format: str = "%Y-%m-%d") -> str:
    "Returns the current date in the specified format."
    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime(format)
    #return datetime.datetime.now().strftime(format)
    return formatted_date

tools= [search_tool, get_current_date]
agent  = initialize_agent(tools=tools, llm=llm, agent  = "zero-shot-react-description", verbose =True) #zero shot means(the agent have not been given any prior(knowledge) examples upon which it can answer the prompt or query)

agent.invoke("When was SpaceX last launch and how many days ago was it?")
#For this search the agent will need current system time because it will need to know the current date to answer the question.
# If such tool is not provided the agent will hallucinate a random method and respond that no such method found.


#agent.invoke("Give me a funny tweet about today's weather in mumbai")
#For this only the Tavily search tool will be used.