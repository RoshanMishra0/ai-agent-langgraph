from langchain.tools import tool
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph

import os

# Tools
@tool
def calculator(expression: str) -> str:
    """Use ONLY for math calculations"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Use ONLY for weather queries"""
    return f"{city}: Sunny, 30°C"

# Model
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
model = init_chat_model("groq:llama-3.1-8b-instant").bind_tools(
    [calculator, get_weather]
)

# State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# LLM Node
def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Tool Node
tool_node = ToolNode([calculator, get_weather])

# Control Logic
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return END

# Graph
builder = StateGraph(AgentState)

builder.add_node("llm", call_model)
builder.add_node("tools", tool_node)

builder.set_entry_point("llm")

builder.add_conditional_edges(
    "llm",
    should_continue,
    {"tools": "tools", END: END},
)

builder.add_edge("tools", "llm")

graph = builder.compile()

# Run
response = graph.invoke({
    "messages": [
        {"role": "user", "content": "What is 5+5 and weather in Delhi?"}
    ]
})

print(response["messages"][-1].content)
