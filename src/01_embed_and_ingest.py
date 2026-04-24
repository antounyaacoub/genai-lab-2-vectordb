from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# 1. Initialize the local Vector Database
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "financial_reports"

# 2. Initialize a local, free embedding model (runs on CPU)
print("Loading embedding model... (This takes a few seconds the first time)")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection (dropping it if it already exists to keep tests clean)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384, # The exact output dimension of all-MiniLM-L6-v2
        distance=models.Distance.COSINE
    )
)

# 3. Mock Enterprise Dataset (Notice the structured metadata)
documents = [
    {"text": "We saw record iPhone sales this quarter, driven by strong demand in Asia.", "company": "Apple", "year": 2023},
    {"text": "Supply chain constraints severely impacted our Mac production timelines.", "company": "Apple", "year": 2022},
    {"text": "Cloud revenue grew 30%, but hardware sales declined due to chip shortages.", "company": "Microsoft", "year": 2023},
    {"text": "We are pivoting heavily into AI with our new Copilot integration.", "company": "Microsoft", "year": 2024},
    {"text": "Our ad revenue dropped due to new privacy policies on iOS.", "company": "Meta", "year": 2022},
]

print("\nEmbedding and ingesting documents into Qdrant...")
points = []
for i, doc in enumerate(documents):
    # Convert text to a dense vector array
    vector = embedder.encode(doc["text"]).tolist()
    
    # Create the Vector DB Point
    points.append(
        models.PointStruct(
            id=i,
            vector=vector,
            payload={"company": doc["company"], "year": doc["year"], "text": doc["text"]} # The Metadata!
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Successfully ingested {len(points)} documents into '{COLLECTION_NAME}'!")