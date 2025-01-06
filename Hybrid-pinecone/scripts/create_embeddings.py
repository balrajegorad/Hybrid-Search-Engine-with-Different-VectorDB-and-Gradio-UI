import os
import configparser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from db import get_mysql_conn

# Load configuration
config = configparser.ConfigParser()

# Get the absolute path to the `config.ini` file
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")
config.read(config_path)

# Initialize Pinecone
pc = Pinecone(
    api_key=config["pinecone"]["api_key"]
)

# Retrieve Pinecone settings
index_name = "hybrid-search-1024"  # Change index name to a new one for 1024 dimensions
model_name = "sentence-t5-large"  # 1024-dimensional model

# Initialize Sentence Transformer model
model = SentenceTransformer(model_name)
dimension = model.get_sentence_embedding_dimension()

# Check if the index exists; create if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",  # Or "euclidean", depending on your use case
        spec=ServerlessSpec(cloud="aws", region=config["pinecone"]["environment"])
    )
    print(f"Created new index: {index_name}")
else:
    print(f"Using existing index: {index_name}")

# Connect to the Pinecone index
index = pc.Index(index_name)

# Get data from MySQL and create embeddings
conn = get_mysql_conn()

with conn.cursor() as cursor:
    cursor.execute("SELECT id, name, description FROM products")
    products = cursor.fetchall()

    for product in products:
        product_id, name, description = product
        embedding = model.encode(description).tolist()
        index.upsert([
            (str(product_id), embedding, {"name": name, "description": description})
        ])
        print(f"Product {product_id} embedding added to Pinecone.")

conn.close()
print("All embeddings created and uploaded to Pinecone.")
