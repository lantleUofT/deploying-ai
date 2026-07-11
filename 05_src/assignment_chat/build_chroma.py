import os
import json

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv('../.env')
load_dotenv('../.secrets')

DOC_DIR = "../documents/pitchfork/"
CHROMA_PATH = "../documents/pitchfork/chroma_db"
COLLECTION_NAME = "pitchfork_reviews"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

ID_KEY = "id"
EMBEDDING_KEY = "embedding"
DOCUMENT_KEY = "text"
METADATA_KEY = "metadata"


def load_chroma_inputs(path):
    inputs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))
    return inputs


def setup_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddingFunction(
            api_key="any value",
            api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
            api_type="openai",
            model_name=EMBEDDING_MODEL,
            default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
        ),
    )
    return collection


def load_embeddings_to_db(collection, chroma_inputs, batch_size=1000):
    for start in range(0, len(chroma_inputs), batch_size):
        batch = chroma_inputs[start:start + batch_size]
        collection.add(
            ids=[item[ID_KEY] for item in batch],
            embeddings=[item[EMBEDDING_KEY] for item in batch],
            documents=[item[DOCUMENT_KEY] for item in batch],
            metadatas=[item[METADATA_KEY] for item in batch],
        )
        print(f"Loaded {min(start + batch_size, len(chroma_inputs))} / {len(chroma_inputs)}")


if __name__ == "__main__":
    chroma_inputs = load_chroma_inputs(os.path.join(DOC_DIR, "chroma_inputs.jsonl"))
    print(f"Loaded {len(chroma_inputs)} records from file")
    collection = setup_collection()
    load_embeddings_to_db(collection, chroma_inputs)
    print(f"Done. Collection count: {collection.count()}")