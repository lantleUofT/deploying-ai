import os
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv('../.env')
load_dotenv('../.secrets')


@tool
def search_market_news(query: str, max_results: int = 5) -> str:
    """Searches the web for current financial and market information: recent
    market news, company developments, earnings, economic events, or anything
    that requires up-to-date information beyond stored data. Use this when the
    user asks what's happening now, why a stock moved, or for recent news about
    a company, sector, or the market."""
    results = TavilySearch(max_results=max_results).invoke({"query": query})
    return str(results)