from typing import List,Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END , MessageGraph
from chain import generation_chain, reflection_chain

load_dotenv()

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state):
    return generation_chain.invoke({"messages":state})

def reflect_node(state):
    response = reflection_chain.invoke({"messages":state})
    return [HumanMessage(content=response.content)]

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    print(f"State length: {len(state)}")
    if (len(state)>4):
        return END
    return REFLECT

graph.add_conditional_edges(
    GENERATE, 
    should_continue,
    {
        "reflect": REFLECT, 
        "__end__": END 
    }
)
# LangGraph needs explicit mapping because your function returns string keys ("reflect", "__end__") but LangGraph doesn't 
# automatically know which nodes those strings should route to - the dictionary tells it 
# "reflect" → REFLECT node and "__end__" → END state.

#Because REFLECT = "reflect" just creates a variable - 
#it doesn't tell LangGraph "when a function returns this value, go to this node" - that's what the mapping dictionary does.


graph.add_edge(REFLECT,GENERATE)

app = graph.compile()
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI Agents taking over content creation!"))
print(response)