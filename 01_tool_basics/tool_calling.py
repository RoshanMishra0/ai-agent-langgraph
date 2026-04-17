from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os

@tool
def calculator(expression: str) -> str:
    return str(eval(expression))

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = init_chat_model("groq:llama-3.1-8b-instant").bind_tools([calculator])

response = model.invoke("What is 5+5?")
print(response)
