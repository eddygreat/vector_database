# vector_database

# Semantic Document Search Engine using FAISS

This project demonstrates how to build a semantic search engine using Vector Databases and Large Language Model (LLM) embeddings. Instead of relying on exact keyword matching, this script retrieves documents based on their underlying meaning and context.

It is designed to run locally or in Google Colab and indexes a small dataset of engineering and technical documents to demonstrate top-k semantic retrieval.

## 🚀 Features

* **Dense Vector Embeddings:** Utilizes the `sentence-transformers` library (`all-MiniLM-L6-v2` model) to convert natural language text into high-dimensional numerical vectors.
* **Lightning-Fast Retrieval:** Uses **FAISS** (Facebook AI Similarity Search) to index vectors and perform efficient similarity searches using Euclidean distance (L2).
* **Semantic Similarity:** Capable of matching user queries to relevant documents even when no exact words are shared (e.g., matching "massive factory cooling" to "Centrifugal chillers").
* **Top-K Results:** Returns the most mathematically relevant documents ranked by distance.

## 🛠️ Prerequisites & Installation

This project requires Python 3.7+ and the following libraries. If you are running this in **Google Colab**, simply run the installation command in your first cell.

```bash
pip install sentence-transformers faiss-cpu numpy

```

*(Note: Use `faiss-gpu` instead of `faiss-cpu` if you are running this on a machine with a dedicated NVIDIA GPU for faster indexing of massive datasets).*

## 🧠 How It Works

1. **Embedding Generation:** The text documents are passed through a pre-trained Transformer model. The model outputs a dense vector (an array of numbers) that encapsulates the semantic meaning of the text.
2. **Vector Indexing:** These vectors are loaded into a FAISS `IndexFlatL2`. This index maps the vectors in a high-dimensional space.
3. **Query Processing:** When a user asks a question, the text query is embedded using the *exact same model* to create a query vector.
4. **Similarity Search:** FAISS calculates the distance between the query vector and all stored document vectors. The system retrieves the `k` documents with the shortest distance, representing the highest semantic similarity.

## 💻 Usage

Run the Python script to initialize the database, generate the embeddings, and execute test queries.

```python
# Example Usage from the script
search_documents("How do I cool down a massive factory?", k=2)

```

### Example Output

```text
Loading model...
Generating embeddings...
Successfully indexed 5 documents.

Search Query: 'How do I cool down a massive factory?'
--------------------------------------------------
Rank 1 (Distance: 1.0532): Centrifugal chillers are heavily utilized in large-scale industrial cooling and centralized air conditioning plants.
Rank 2 (Distance: 1.4021): Variable Refrigerant Flow (VRF) systems provide highly efficient, zoned temperature control for modern commercial buildings.

```

*Notice how the system successfully identified the centrifugal chiller document without the query containing the words "centrifugal", "chiller", or "industrial".*

## 🗂️ Project Structure

* `main.py` (or your Colab Notebook): Contains the core logic for loading the model, indexing the vectors, and the `search_documents()` function.
* **Dataset:** Currently hardcoded as a small python list within the script for demonstration purposes.

## 🔮 Future Improvements

* Swap the local FAISS index for a managed cloud vector database like **Pinecone** for production-scale deployment.
* Implement a function to save the FAISS index to disk (`faiss.write_index`) to avoid re-computing embeddings on every run.
* Connect the script to a PDF parser to index real-world engineering manuals or building codes.
