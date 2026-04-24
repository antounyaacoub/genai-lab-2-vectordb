from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# The User's Query
user_query = "What hardware or supply chain issues did Apple face in 2022?"

# Convert the query to a vector
query_vector = embedder.encode(user_query).tolist()

print(f"USER QUERY: '{user_query}'\n")
print("Executing Naive Semantic Search...\n")

# CHANGED: Using the modern query_points API
response = client.query_points(
    collection_name="financial_reports",
    query=query_vector,
    limit=2
)

# Display results
for i, hit in enumerate(response.points):
    print(f"--- Result {i+1} (Score: {hit.score:.4f}) ---")
    print(f"Company: {hit.payload['company']} | Year: {hit.payload['year']}")
    print(f"Text: {hit.payload['text']}\n")

print("OBSERVATION: Semantic search is fuzzy. It might return 2022 data, or Microsoft data, because the text concepts match, ignoring the strict noun 'Apple' and integer '2022'.")