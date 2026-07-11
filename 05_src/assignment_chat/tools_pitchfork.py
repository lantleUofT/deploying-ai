import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.tools import tool

CHROMA_PATH = "../documents/pitchfork/chroma_db"
COLLECTION_NAME = "pitchfork_reviews"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

_embedding_function = OpenAIEmbeddingFunction(
    api_key="any value",
    api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_type="openai",
    model_name=EMBEDDING_MODEL,
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_function,
)


@tool
def search_album_reviews(query: str, top_n: int = 3) -> str:
    """Searches a database of Pitchfork album reviews using semantic similarity.
    Use this whenever the user asks for album or music recommendations, or asks
    what reviews say about a style, mood, genre, or kind of record (e.g. 'dreamy
    shoegaze albums', 'highly rated jazz', 'moody electronic records')."""
    results = _collection.query(query_texts=[query], n_results=top_n)

    context_data = []
    for idx in range(len(results["ids"][0])):
        details = dict(results["metadatas"][0][idx])
        details["text"] = results["documents"][0][idx]
        context_data.append(details)

    if not context_data:
        return "No matching album reviews found."

    lines = []
    for c in context_data:
        lines.append(
            f"Album: {c.get('album', 'N/A')} | Artist: {c.get('artist', 'N/A')} | "
            f"Score: {c.get('score', 'N/A')} | Genre: {c.get('genre', 'N/A')} | "
            f"Year: {c.get('year', 'N/A')}\nReview excerpt: {c.get('text', 'N/A')}"
        )
    return "\n\n".join(lines)