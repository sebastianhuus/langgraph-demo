from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

# load model from ollama
llm = init_chat_model(
    "ollama:gemma3:12b",
)

# set up graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# define graph builder for the state
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

user_input = input("Enter a message: ")
messages = [{"role": "system", "content": "You are a helpful assistant called Tim"}]
state = graph.invoke({"messages": messages + [{"role": "user", "content": user_input}]})
response = state["messages"][-1]
print(response.content)
print(state) 