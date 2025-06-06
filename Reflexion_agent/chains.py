from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(
    tools =[AnswerQuestion]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)
# std actor agent prompt template 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AL researcher.
            Current time: (time)
            1. (figst_instruction)
            2. Reflect and critique your answer. Be severe to maximize improvenent.
            3. After the reflection, **list 1-3 search queries separately** for researching Improvements. Do not include them inside the reflection."""
        ),
        MessagesPlaceholder(variable_name="messages"), 
        ("system","Answer the user's question above using the required format.")
    
    ]
).partial(
    time =lambda: datetime.datetime.now().isoformat(),
)
#.partial() here creates a prompt template with the current time already filled in (as per mentioned in the chat prompt template).

#RESPONDER SECTION

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a ~250 word detailed answer to the question."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools = [AnswerQuestion], tool_choice='AnswerQuestion') | pydantic_parser


#REVISOR SECTION
revise_instructions = """"Revise your previous answer using the new
                        information.
                        - You should use the previous critique to agp important information to your answer.
                        - You MUST include numerical citations in your revised answer to ensure it can be verified.
                        - Add a "References" section to the bottom of your answer (which does not count towards the word limit).
                        In form of:
                        - [1) https://example.com
                        - 2) https://example.com
                        - You should use the previous critique to remove superfluous information from your answer and make SURE 
                        it is not more than 250 words."""
                        
revisor_prompt_template = actor_prompt_template.partial(
    first_instruction= revise_instructions
) | llm.bind_tools(
    tools = [ReviseAnswer], tool_choice='ReviseAnswer')


response = first_responder_chain.invoke(
    {
        "messages": [
            HumanMessage(content="Write me a blog post about AI agents taking over content creation!"),
        ]
    }
)

print(response)