import os

# Disable optional Chroma telemetry before importing Chroma or creating clients.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(PROJECT_DIR, "db", "chroma_db")
collection_name = "langchain"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Use the same embedding model and cosine-configured collection as injection.py.
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
client = chromadb.PersistentClient(
    path=persistent_directory,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection(name=collection_name)
if (collection.metadata or {}).get("hnsw:space") != "cosine":
    raise RuntimeError(
        f"Collection '{collection_name}' is not configured for cosine similarity. "
        "Recreate the vector store by running injection.py."
    )

db = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)

# ===== Build BM25 index from all chunks currently in the vector store =====
all_data = collection.get(include=["documents", "metadatas"])
all_chunks = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(all_data["documents"], all_data["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = 6  # candidates to pull from keyword search

# ===== Load the cross-encoder reranker once =====
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Replace the text between the quotes with any question you want to ask.
query = "why water conservation matter?"

# ===== Step 1: Vector search (semantic) =====
vector_results = db.similarity_search_with_relevance_scores(query, k=6)
vector_docs = [doc for doc, _score in vector_results]

# ===== Step 2: BM25 search (keyword) =====
bm25_docs = bm25_retriever.invoke(query)

# ===== Step 3: Combine candidates, removing duplicates =====
combined_docs = {}
for doc in vector_docs + bm25_docs:
    combined_docs[doc.page_content] = doc  # dedupe by content
candidates = list(combined_docs.values())

if not candidates:
    print("I don't have enough information to answer that question based on the provided documents.")
    exit()

# ===== Step 4: Rerank combined candidates with the cross-encoder =====
pairs = [[query, doc.page_content] for doc in candidates]
scores = reranker.predict(pairs)

scored_docs = list(zip(candidates, scores))
scored_docs.sort(key=lambda x: x[1], reverse=True)

# Keep only the top 4 after reranking
top_docs = [doc for doc, _score in scored_docs[:4]]

context = "\n\n".join(
    f"Document {index}: {document.page_content}"
    for index, document in enumerate(top_docs, 1)
)

combined_input = f"""Answer the question using only the context below.
Question: {query}
Context:
{context}

Give a concise, accurate answer based only on the context. Include the important details needed to answer the question. If the answer is not present in the context, say: "I don't have enough information to answer that question based on the provided documents." Do not invent facts.
"""

# Run the answer model locally so retrieval does not depend on the hosted
# Hugging Face inference endpoint or its DNS/network connection.
local_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(local_model_id)
local_model = AutoModelForCausalLM.from_pretrained(local_model_id)
local_model.generation_config.do_sample = False
local_model.generation_config.temperature = None
local_model.generation_config.top_p = None
local_model.generation_config.top_k = None
local_pipeline = pipeline(
    "text-generation",
    model=local_model,
    tokenizer=tokenizer,
    max_new_tokens=384,
    do_sample=False,
    return_full_text=False,
)
model = HuggingFacePipeline(pipeline=local_pipeline)
result = model.invoke(combined_input)

print(f"You: {query}")
print(f"Bot: {result.strip()}")