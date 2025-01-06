import os
from pinecone import Pinecone, ServerlessSpec
import configparser
from sentence_transformers import SentenceTransformer

config = configparser.ConfigParser()

# Get the absolute path to the `config.ini` file
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")
config.read(config_path)

# Initialize Pinecone client
pc = Pinecone(api_key=config["pinecone"]["api_key"])

index_name = config["pinecone"]["index_name"]
model_name = "sentence-t5-large"  # Ensure the same 1024-dimensional model is used

# Initialize Sentence Transformer model
model = SentenceTransformer(model_name)
dimension = model.get_sentence_embedding_dimension()  # Dynamically check dimension

# Connect to the Pinecone index
if index_name not in pc.list_indexes().names():  # Use 'pc' here instead of 'pinecone'
    raise ValueError(f"Index '{index_name}' does not exist. Please create it first.")

index = pc.Index(index_name)  # Use 'pc' instead of 'pinecone'

def search_description(description_query, top_k=10):
    """
    Searches for the top_k most similar items in Pinecone based on a description query.

    :param description_query: The text description to search for.
    :param top_k: Number of top results to retrieve.
    :return: List of search results with metadata.
    """
    query_embedding = model.encode(description_query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

