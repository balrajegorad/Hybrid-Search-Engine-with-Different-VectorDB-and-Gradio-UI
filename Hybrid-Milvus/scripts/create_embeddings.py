import os
import configparser
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from db import get_mysql_conn

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

# Milvus collection settings
collection_name = "hybrid_search"
model_name = "sentence-t5-large"

# Initialize Sentence Transformer model
model = SentenceTransformer(model_name)
dimension = model.get_sentence_embedding_dimension()

# Create collection if it doesn't exist
if not utility.has_collection(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]
    schema = CollectionSchema(fields, description="Hybrid search collection")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 100}})
    print(f"Created new collection: {collection_name}")
else:
    collection = Collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")

# Get data from MySQL and create embeddings
conn = get_mysql_conn()
with conn.cursor() as cursor:
    cursor.execute("SELECT id, name, description FROM products")
    products = cursor.fetchall()

    ids, names, descriptions, embeddings = [], [], [], []
    for product in products:
        product_id, name, description = product
        embedding = model.encode(description).tolist()
        ids.append(product_id)
        names.append(name)
        descriptions.append(description)
        embeddings.append(embedding)

    # Insert data into Milvus
    data = [ids, names, descriptions, embeddings]
    collection.insert(data)
    collection.load()
    print("All embeddings created and uploaded to Milvus.")

conn.close()
