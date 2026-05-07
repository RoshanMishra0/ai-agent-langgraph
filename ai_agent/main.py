from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated, List
import os

from config import GROQ_API_KEY, GROQ_MODEL
from tools.calculator import calculator
from tools.weather import get_weather
from router.semantic_router import get_routes

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm_model = init_chat_model(GROQ_MODEL)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    routes: List[str]


def route_query(state):
    query = state["messages"][-1].content
    routes = get_routes(query)

    return {"routes": routes}


SYSTEM_PROMPT = """
You are a strict tool-using assistant.
Only use available tools.
"""


def call_model(state):
    routes = state["routes"]
    messages = state["messages"]

    selected_tools = []

    if "math" in routes:
        selected_tools.append(calculator)

    if "weather" in routes:
        selected_tools.append(get_weather)

    if selected_tools:
        model = llm_model.bind_tools(selected_tools)
    else:
        model = llm_model

    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages
    ])

    return {"messages": [response]}


tool_node = ToolNode([calculator, get_weather])


def should_continue(state):
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return END


builder = StateGraph(AgentState)

builder.add_node("router", route_query)
builder.add_node("llm", call_model)
builder.add_node("tools", tool_node)

builder.set_entry_point("router")
builder.add_edge("router", "llm")

builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

builder.add_edge("tools", "llm")

graph = builder.compile()


response = graph.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is weather in Delhi and 5+5?"
        }
    ]
})

print(response["messages"][-1].content)
