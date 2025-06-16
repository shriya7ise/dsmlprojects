from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub

llm= ChatGoogleGenerativeAI(model="gemini-1.5-flash")
search_tool= TavilySearchResults(search_depth = "basic")

@tool
def get_current_date(format: str = "%Y-%m-%d") -> str:
    "Returns the current date in the specified format."
    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime(format)
    #return datetime.datetime.now().strftime(format)
    return formatted_date

tools= [search_tool, get_current_date]

react_prompt = hub.pull("hwchase17/react")
react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
