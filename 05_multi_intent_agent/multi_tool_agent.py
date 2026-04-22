from langchain.tools import tool
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.graph import StateGraph
from typing import List
import os
import json

@tool
def calculator(expression: str) -> str:
    """Use ONLY for math calculations like 5+5, 10*3"""
    return str(eval(expression))


@tool
def get_weather(city: str) -> str:
    """Get real-time weather using OpenWeather API"""

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("cod") != 200:
            return f"[WEATHER ERROR] {data.get('message')}"

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return (
            f"[WEATHER RESULT] {city}: {description}, "
            f"{temp}°C (feels like {feels_like}°C), "
            f"humidity {humidity}%"
        )

    except Exception as e:
        return f"[WEATHER ERROR] {str(e)}"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

router_model = init_chat_model("groq:llama-3.1-8b-instant")
llm_model = init_chat_model("groq:llama-3.1-8b-instant")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    routes: List[str]
def route_query(state: AgentState):
    query = state["messages"][-1].content

    router_prompt = f"""
You are an intent classifier.

Return ONLY a valid JSON list. No explanation. No text.

Allowed categories:
- math
- weather
- general

Examples:
"5+5" → ["math"]
"weather in Delhi" → ["weather"]
"5+5 and weather in Delhi" → ["math", "weather"]
"Tell me a joke" → ["general"]

Query: {query}

Output:
"""
    response = router_model.invoke([
        {"role":"user","content":router_prompt}
    ])
    try:
        routes = json.loads(response.content)
        if not isinstance(routes, list):
            routes = ["general"]
    except:
        routes = ["general"]

    return {"routes": routes}

SYSTEM_PROMPT = """
You are a strict tool-using assistant.

AVAILABLE TOOLS:
- calculator → for math expressions
- get_weather → for weather queries

STRICT RULES:
- You MUST ONLY use the provided tools
- You MUST NOT invent tools (e.g., brave_search)
- You MUST call tools when needed
- You MUST use tool outputs in final answer
- NEVER hallucinate APIs or websites

Return final answer ONLY after tool results.
"""

def call_model(state: AgentState):
    routes = state["routes"]
    messages = state["messages"]

    selected_tools = []

    # 🔥 Restrict tools based on route
    if "math" in routes:
        selected_tools.append(calculator)

    if "weather" in routes:
        selected_tools.append(get_weather)


    if selected_tools:
      model = llm_model.bind_tools(selected_tools)
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
        {"role": "user", "content": "What is 5+5 and weather in Delhi?"}
    ]
})

print(response["messages"][-1].content)
