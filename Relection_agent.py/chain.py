from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a twitter techie influencer tasked with writing excellent twitter posts. "
     "Generate the best Twitter post possible for the user's request. "
     "If the user provides critique, respond with a revised version of your previous attempts."
    ),
    MessagesPlaceholder(variable_name="messages")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a viral twitter influencer grading a tweet.Generate critique and recommendations for the user's tweet. "
     "Always provide detailed recommendations , including requests for length , virality, style, etc. "
    ),
    MessagesPlaceholder(variable_name="messages")
])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

