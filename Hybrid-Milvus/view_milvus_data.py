from pymilvus import connections, Collection, utility

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# List collections in Milvus
collections = utility.list_collections()
print("Collections in Milvus:")
print(collections)

# Specify the correct collection name
collection_name = "hybrid_search"  # Correct collection name
collection = Collection(name=collection_name)

# Check if the schema is ready
print(f"\nSchema of the collection '{collection_name}':")
print(collection.describe())

# Count the number of entities (rows) in the collection
print(f"\nNumber of entities in '{collection_name}': {collection.num_entities}")

# Query for all records (embeddings) - adjust the output fields as necessary
print("\nAll embeddings and metadata in the collection:")

# You can limit the results or remove the limit to fetch all
results = collection.query(
    expr="",  # No filter; retrieve all records
    output_fields=["name", "description", "embedding"],  # Adjust according to your collection schema
    limit=20  # Adjust this to the number of records you want to see
)

# Display the results
for record in results:
    print(record)
