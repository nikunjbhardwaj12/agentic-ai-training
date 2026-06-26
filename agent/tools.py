from langchain_core.tools import tool
from duckduckgo_search import DDGS
import datetime

@tool
def search_web(query: str) -> str:
    """Searches the web for the given query to find real-time information."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No results found."
            
            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nLink: {r.get('href')}\n---")
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing search: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Safely evaluates a basic mathematical expression."""
    try:
        # Restrict globals and locals for a basic layer of safety
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "Error: Invalid characters in expression."
        
        result = eval(expression, {"__builtins__": None}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

@tool
def get_current_date() -> str:
    """Returns today's current date."""
    return f"Today's date is {datetime.date.today().strftime('%B %d, %Y')}."

# Export list of tools for easy binding
all_tools = [search_web, calculate, get_current_date]