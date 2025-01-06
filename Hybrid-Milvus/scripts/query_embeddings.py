import os
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import configparser

# Load configuration
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")
config.read(config_path)

# Connect to Milvus
connections.connect(
    alias="default",
    host=config["milvus"]["host"],
    port=config["milvus"]["port"]
)

collection_name = "hybrid_search"
model_name = "sentence-t5-large"

# Initialize Sentence Transformer model
model = SentenceTransformer(model_name)

# Load collection
collection = Collection(name=collection_name)

def search_description(description_query, top_k=10):
    """
    Searches for the top_k most similar items in Milvus based on a description query.

    :param description_query: The text description to search for.
    :param top_k: Number of top results to retrieve.
    :return: List of search results with metadata.
    """
    query_embedding = model.encode(description_query).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "name", "description"]
    )
    return results
