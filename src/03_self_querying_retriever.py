import os
import instructor
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize LLM (Query Router)
llm_client = instructor.from_openai(
    OpenAI(api_key=os.getenv("API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
    mode=instructor.Mode.TOOLS
)

# Initialize Vector DB & Embedder
qdrant = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- 1. Define the LLM Extraction Schema ---
class SearchIntent(BaseModel):
    company_filter: Optional[str] = Field(description="The specific company mentioned, if any. Otherwise null.")
    year_filter: Optional[int] = Field(description="The specific year mentioned, if any. Otherwise null.")
    semantic_query: str = Field(description="The remaining core question to search for, optimized for a vector database.")

user_query = "What hardware or supply chain issues did Apple face in 2022?"
print(f"USER QUERY: '{user_query}'\n")

# TODO: Step 1 - Use the LLM to extract the SearchIntent from the user_query
intent = llm_client.chat.completions.create(
    model="gemini-1.5-flash", # Or gemini-3-flash-preview
    response_model=SearchIntent,
    messages=[{"role": "user", "content": f"Extract filters from this query: {user_query}"}]
)

print(f"LLM ROUTER DECISION:")
print(f"Filter Company: {intent.company_filter}")
print(f"Filter Year: {intent.year_filter}")
print(f"Optimized Vector Query: {intent.semantic_query}\n")

# TODO: Step 2 - Convert the Pydantic filters into Qdrant Metadata Filters
# Hint: Use models.Filter(must=[...]) and models.FieldCondition()
qdrant_filters = []

if intent.company_filter:
    pass # Add FieldCondition for "company"
    
if intent.year_filter:
    pass # Add FieldCondition for "year"

# Combine conditions into a Qdrant Filter object
strict_filter = models.Filter(must=qdrant_filters) if qdrant_filters else None

# TODO: Step 3 - Embed the optimized semantic_query
query_vector = [] # Replace with embedder.encode(intent.semantic_query).tolist()

# TODO: Step 4 - Execute the search on Qdrant using the new query_points API
response = client.query_points(
    collection_name="financial_reports",
    query=query_vector,
    query_filter=strict_filter, # This injects the strict NoSQL metadata filter!
    limit=2
)

print("--- HYBRID SEARCH RESULTS ---")
for i, hit in enumerate(response.points):
    print(f"Score: {hit.score:.4f} | Company: {hit.payload['company']} | Year: {hit.payload['year']}")
    print(f"Text: {hit.payload['text']}\n")