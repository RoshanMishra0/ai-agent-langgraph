from langchain.tools import tool
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.graph import StateGraph

import os

@tool
def calculator(expression: str) -> str:
    """Use ONLY for math calculations like 5+5, 10*3"""
    return str(eval(expression))


@tool
def get_weather(city: str) -> str:
    """Use ONLY for weather queries"""
    return f"{city}: Sunny, 30°C"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

router_model = init_chat_model("groq:llama-3.1-8b-instant")
llm_model = init_chat_model("groq:llama-3.1-8b-instant")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
def route_query(state: AgentState):
    query = state["messages"][-1].content

    router_prompt = f"""
Classify the user query into ONE of these categories:
- math
- weather
- general

Only return the category name.

Query: {query}
"""

    response = router_model.invoke([
        {"role": "user", "content": router_prompt}
    ])

    route = response.content.strip().lower()

    # safety fallback
    if route not in ["math", "weather", "general"]:
        route = "general"

    return {"route": route}

SYSTEM_PROMPT = """
You are a strict assistant.

Rules:
- Use tools when required
- ALWAYS use tool outputs
- Be concise
"""

def call_model(state: AgentState):
    route = state["route"]
    messages = state["messages"]

    # Restrict tools based on route
    if route == "math":
        model = llm_model.bind_tools([calculator])

    elif route == "weather":
        model = llm_model.bind_tools([get_weather])

    else:
        model = llm_model  # no tools

    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages
    ])

    return {"messages": [response]}


tool_node = ToolNode([calculator, get_weather])


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return END


builder = StateGraph(AgentState)

builder.add_node("router", route_query)
builder.add_node("llm", call_model)
builder.add_node("tools", tool_node)

# Flow
builder.set_entry_point("router")
builder.add_edge("router", "llm")

builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "llm")

graph = builder.compile()

response = graph.invoke({
    "messages": [
        {"role": "user", "content": "What is 5+5?"}
    ]
})

print(response["messages"][-1].content)
