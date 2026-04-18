from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
import os

# Tools
@tool
def calculator(expression: str) -> str:
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    return f"{city}: Sunny, 30°C"

# API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 🔹 Model
model = init_chat_model("groq:llama-3.1-8b-instant").bind_tools(
    [calculator, get_weather]
)

SYSTEM_PROMPT = "You must use tools and use their outputs. Do not guess."

query = "What is 5+5 and weather in Delhi?"

# Step 1: LLM call
ai_msg = model.invoke([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query}
])

# Step 2: Manual tool execution
tool_results = []

for call in ai_msg.tool_calls:
    if call["name"] == "calculator":
        result = calculator.invoke(call["args"])
    elif call["name"] == "get_weather":
        result = get_weather.invoke(call["args"])

    tool_results.append(
        ToolMessage(
            content=result,
            tool_call_id=call["id"],
            name=call["name"]
        )
    )

# Step 3: Final LLM call
final_response = model.invoke([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query},
    ai_msg,
    *tool_results
])

# 🔹 Output
print(final_response.content)
