# ~/ai_agent_system/search_tool.py
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool("web_search")
def web_search(query: str) -> str:
    """
    Performs a comprehensive internet search to retrieve up-to-date information.
    Input should be a clear, concise search query relevant to the information needed.
    Returns the search results summary.
    """
    logger.warning("The web_search tool's Python function is a placeholder and should not be directly executed.")
    return "This tool is handled directly by the LLM's underlying platform (OpenRouter)."

ALL_TOOLS = [web_search]

if __name__ == '__main__':
    print(web_search.name)
    print(web_search.description)
    print(web_search.args)
