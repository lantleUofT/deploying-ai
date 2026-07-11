# Assignment 2: 

This is a finance focused conversational agent built with LangChain and Gradio. The chat
client takes on the persona of a seasoned stock broker and answers questions by
drawing on three back end services: a live stock quote API, a semantic search
over a database of album reviews, and a web search for current market news.

## The Chat Client

The interface is a Gradio `ChatInterface` using a single `simple_chat`
function. The model is OpenAI `gpt-4o-mini`, accessed through the course API
gateway. Tools are made available to the model with LangChain's `bind_tools`,
and the model decides on its own which tool (if any) a given message requires.


### Personality

The client is styled as "Sal Moreno," a stock broker with thirty years on the
trading floor: plain spoken, direct, fond of market slang, and a lifelong music
obsessive. The persona is created via a structured system prompt.

### Memory

Conversation memory is handled through Gradio's `history` parameter, passing the full prior conversation
into `simple_chat` so the model therefore sees the entire session on each call and can
resolve follow up references such as "how does that compare to Microsoft." It has
full within session memory for the length of the conversation (not the optional memory management system).

## The Three Services

### Service 1: Stock Quotes (API)

`tools_marketstack.py` is a `get_end_of_day` tool that calls the Marketstack
end of day API for a given ticker symbol. The raw response (open, high, low,
close, volume) is returned to the model, which briefs the user on it in the
broker's voice.

Requires a `MARKET_STACK_API_KEY` in the secrets file.

### Service 2: Album Review Semantic Search (ChromaDB)

`tools_pitchfork.py` is a `search_album_reviews` tool that performs semantic
search over a persistent ChromaDB collection of Pitchfork album reviews. The tool
connects to an existing database, and embeds the user's query with
`text-embedding-3-small` through the gateway, and returns the most similar
reviews along with their metadata (album, artist, score, genre, year). 

Embedding process: the embeddings were provided pre computed by the instructional team in the class file
`chroma_inputs.jsonl`, which contains, for each review, an id, the review text,
a 1536 dimensional embedding from `text-embedding-3-small`, and metadata. The
script `build_chroma.py` documents how I created my version of the persistent database
This build step was run once and is not part of running the app.

Database location: the application expects a persistent ChromaDB at
`05_src/documents/pitchfork/chroma_db/`, containing a collection named
`pitchfork_reviews`. It assumes the grader's copy of the database was placed at that path before
running the app.

### Service 3: Market News Web Search (Tavily)

`tools_websearch.py` is a `search_market_news` tool that performs a web
search with Tavily for current financial information: recent market news, company
developments, earnings, and economic events. This complements Service 1 by
providing current context and news, which the stock quote API does not offer.

This is a simple web search (a single Tavily call returning results), not an
agentic search.

Requires a `TAVILY_API_KEY` in the secrets file.

## Guardrails

The guardrails are implemented in the system prompt (the developer prompt),
placed at the start and end of the prompt where instruction adherence tends to be
strongest. They cover:

- Prompt extraction: the client refuses to reveal, repeat, summarize, or
  translate its instructions.
- Prompt modification: the client refuses instructions that attempt to change or
  override its rules or role.
- Restricted topics: the client refuses to discuss cats or dogs, horoscopes or
  zodiac signs, and Taylor Swift, deflecting back to markets.

This approach was tested against extraction and topic attempts. 

## Project Structure

```
assignment_chat/
  assignment_2_interface.py   
  tools_marketstack.py        
  tools_pitchfork.py          
  tools_websearch.py          
  build_chroma.py             
```


## Setup and Running

1. Ensure the standard course environment is active.

2. Provide the following keys in `05_src/.secrets`:
   - `API_GATEWAY_KEY` for the model gateway
   - `OPENAI_API_KEY` (any non empty value; the gateway requires the variable to
     be present but does not use it)
   - `MARKET_STACK_API_KEY` for Service 1
   - `TAVILY_API_KEY` for Service 3

3. Ensure the persistent ChromaDB is present at
   `05_src/documents/pitchfork/chroma_db/` with the `pitchfork_reviews`
   collection.

4. From the `assignment_chat/` directory, run: python assignment_2_interface.py


5. Open the local URL that Gradio prints.
