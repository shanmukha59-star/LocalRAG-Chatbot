# rag_engine.py
import os
import sys
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download

os.environ["ANONYMIZED_TELEMETRY"] = "False"

if getattr(sys, "frozen", False):
    PROJECT_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = getattr(sys, "_MEIPASS", PROJECT_DIR)
else:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = PROJECT_DIR

bundled_database = os.path.join(RESOURCE_DIR, "db", "chroma_db")
project_database = os.path.join(PROJECT_DIR, "db", "chroma_db")
persistent_directory = (
    bundled_database if os.path.isdir(bundled_database) else project_database
)
collection_name = "langchain"

# ---- Fixed local folder for model weights (client machine friendly) ----
MODEL_BASE_DIR = os.path.join(os.path.expanduser("~"), ".rag_chatbot_models")
LLM_DIR = os.path.join(MODEL_BASE_DIR, "qwen2.5-1.5b-instruct")
EMBEDDING_DIR = os.path.join(MODEL_BASE_DIR, "all-MiniLM-L6-v2")

# Filled in by load_models()
db = None
chat_model = None


def ensure_models_downloaded(progress_callback=None):
    """Download models to a fixed local folder on first run only."""
    if not os.path.exists(os.path.join(LLM_DIR, "config.json")):
        if progress_callback:
            progress_callback("Downloading language model (this happens once)...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            local_dir=LLM_DIR,
        )

    if not os.path.exists(os.path.join(EMBEDDING_DIR, "config.json")):
        if progress_callback:
            progress_callback("Downloading embedding model (this happens once)...")
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=EMBEDDING_DIR,
        )


def load_models(progress_callback=None):
    """Download (if needed) and load everything once. Safe to call multiple times."""
    global db, chat_model
    if chat_model is not None:
        return

    ensure_models_downloaded(progress_callback)

    if progress_callback:
        progress_callback("Loading models into memory...")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_DIR)

    client = chromadb.PersistentClient(
        path=persistent_directory,
        settings=Settings(anonymized_telemetry=False),
    )
    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
    local_model = AutoModelForCausalLM.from_pretrained(LLM_DIR)
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
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    chat_model = ChatHuggingFace(llm=llm)


def retrieve_chunks(query: str, k: int = 4):
    load_models()
    results = db.similarity_search_with_relevance_scores(query, k=k)
    return [doc for doc, _score in results]


def build_context(chunks):
    return "\n\n".join(
        f"Document {i}: {doc.page_content}" for i, doc in enumerate(chunks, 1)
    )


def ask_rag(query: str, history: list[dict] | None = None) -> str:
    load_models()

    chunks = retrieve_chunks(query)
    if not chunks:
        return "I don't have enough information to answer that question based on the provided documents."

    context = build_context(chunks)

    system_prompt = (
        "Answer the question using only the context below. "
        "If the answer is not present in the context, say so clearly. Do not invent facts.\n\n"
        f"Context:\n{context}"
    )

    messages = [SystemMessage(content=system_prompt)]

    if history:
        for turn in history:
            messages.append(HumanMessage(content=turn["user"]))
            messages.append(AIMessage(content=turn["assistant"]))

    messages.append(HumanMessage(content=query))

    response = chat_model.invoke(messages)
    return response.content.strip()
