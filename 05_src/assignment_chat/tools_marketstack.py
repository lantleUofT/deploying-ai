import os
import json

import requests
from langchain.tools import tool

MARKETSTACK_BASE_URL = "https://api.marketstack.com/v2"


@tool
def get_end_of_day(symbol: str) -> str:
    """Returns the latest end-of-day price data for a stock ticker symbol
    (e.g. 'AAPL', 'MSFT', 'TSLA'). Use this whenever the user asks how a
    specific stock or company is doing, what it closed at, or its latest price."""
    symbol = symbol.upper()

    url = f"{MARKETSTACK_BASE_URL}/eod/latest"
    params = {
        "access_key": os.environ["MARKET_STACK_API_KEY"],
        "symbols": symbol,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        return f"The data desk returned an error for {symbol}: {payload['error'].get('message', 'unknown error')}."

    data = payload.get("data", [])
    if not data:
        return f"No end-of-day data found for '{symbol}'. Double-check the ticker."

    return json.dumps(data[0])