from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START,END
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command
from langchain_core.messages import HumanMessage, BaseMessage,AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 

def chat_node(state: ChatState) -> str:
    decision=interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question.",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })
    if decision == "no":
        return {"messages": [AIMessage(content="Question was rejected by the user.")] }
    else:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response]
        }   

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

checkpointer=MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"1234"}}
response=workflow.invoke({
    "messages": [HumanMessage(content="What is the capital of France?")]    
},config=config)

message = response['__interrupt__'][0].value

user_input=input(f"\nBackend message - {message} \n Approve this question? (y/n): ")

# Convert y/n to yes/no for matching
decision = "yes" if user_input.lower().strip() == "y" else "no"

final_result = workflow.invoke(
    Command(resume=decision),
    config=config,
)
print(final_result["messages"][-1].content)