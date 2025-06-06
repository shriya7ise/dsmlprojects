Advantages: 
ReACT agents are flexible, i.e any state is possible   -  Usees tools according to the problem,

Disadvantages:
Flexibility also means less reliability
[Infinite loop causes:
    1. tools are defined incorrectly
    2. LLM is not capable enough
    3. prompting doesnt define clear end condition] 

**These disadvantages are cleared by using Langgraph**


**LANGGRAPH**
A framework for building controllable  persistent agent workflow with built in support for human interaction, streaming and state management.
* Key features of langgraph
1) Looping and Branching Capabilities
2) State Persistance
3) human machine interaction support
4) streaming processing
5) seamless integration with langchain

* Core components of Langgraph
1) Nodes 
2) Edges
3) Conditional edges
4) state

**Reflection Agents**
It is an AI system pattern that can loook at it's own outputs and thinkabout making them better.
This system consist of :
* A Generator agent
* A reflector agent

Types of Reflection agents
1) Baisc RA:
* It is a class that langgrpah provides that we can use to orchestrate the flow of messages between the nodes; 
(If you just want to pass the messages along between nodes, htne go for message Graph) 


2) Reflexion A
* Reflexion agent, similar to reflection agent, not only critiquesit's own responses but also crosschecks facts using external/internet search(api calls)

- Actor: Main component of ag. which drives everything , it reflexes its own responses and re-executes
- main sub components include: 
- 1) Tools/tool execution
- 2) Initial responder 
- 3) Revisor

* It has episodic memory  whihc refers to n agents ability to ecall specific past events, interactions, rather than generalized logic.
* This memory makes agent context aware and human like over time.


3) Language agent tree search (LATS)

