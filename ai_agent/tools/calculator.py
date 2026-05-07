import numexpr as ne
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Use ONLY for math calculations like 5+5, 10*3"""
    try:
        result = ne.evaluate(expression)
        return f"[CALC RESULT] {result}"
    except Exception as e:
        return f"[CALC ERROR] {str(e)}"
