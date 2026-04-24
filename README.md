# Lab 2: Enterprise Vector Engineering & Metadata Filtering

## Objective
Basic RAG (Retrieval-Augmented Generation) relies on pure semantic search (Cosine Similarity). In an enterprise setting, this fails. If a user asks "What were Apple's earnings in 2023?", a pure vector search might return Microsoft's 2023 earnings, or Apple's 2022 earnings, simply because the text is semantically similar.

Enterprise Vector Databases require **Hybrid Filtering**: combining fuzzy semantic search with strict, deterministic NoSQL metadata filters.

In this lab, you will:
1. Spin up a local Qdrant Vector Database using Docker.
2. Build an ingestion pipeline that embeds financial reports and attaches structured JSON metadata.
3. Observe the failure of naive semantic search.
4. Build a **Self-Querying Retriever**: An LLM agent that intercepts the user's question, extracts the metadata into a Pydantic model, and converts it into a strict Vector DB filter.

## Setup
1. Clone this repository.
2. Start the Vector Database: `docker-compose up -d`
3. Create a virtual environment: `python -m venv venv` and activate it.
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your Gemini/OpenAI API Key: `API_KEY=your_api_key_here`

---

## Part 1: Ingestion & Metadata Tagging
Look at `src/01_embed_and_ingest.py`. This script simulates an ETL pipeline processing corporate financial reports. It uses a local, free HuggingFace model (`all-MiniLM-L6-v2`) to convert text into 384-dimensional vectors.
**Your Task:**
1. Run the script: `python src/01_embed_and_ingest.py`
2. Look at the terminal output. Notice how we are pushing both the `vector` AND a JSON `payload` (metadata) to Qdrant.

## Part 2: The Failure of Naive Search
Look at `src/02_naive_semantic_search.py`.
**Your Task:**
1. Run the script: `python src/02_naive_semantic_search.py`
2. **Observe the failure:** The user specifically asked about "Apple" in "2022". However, the top result might be Microsoft, or Apple in 2023. Vector math does not understand strict nouns and numbers, only general concepts.

## Part 3: The Self-Querying Retriever (Engineering Challenge)
Open `src/03_self_querying_retriever.py`. We need to fix the pipeline.
Instead of sending the user's raw query directly to the Vector DB, we will send it to an LLM first. The LLM will act as a Query Router, extracting the target `company` and `year` into a Pydantic model.

**Your Engineering Tasks:**
1. Use `Instructor` to extract the `company` and `year` from the user's query into the provided Pydantic model.
2. Convert that Pydantic model into Qdrant `models.Filter` objects.
3. Execute the Qdrant search using BOTH the semantic query vector AND the strict metadata filter. 
4. Run the script and prove that the retrieval is now 100% accurate.