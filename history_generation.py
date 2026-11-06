import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Disable optional Chroma telemetry before creating any Chroma clients.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Load environment variables
load_dotenv()

# Connect to your document database
project_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(project_directory, "db", "chroma_db")
collection_name = "langchain"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(
    path=persistent_directory,
    settings=Settings(anonymized_telemetry=False),
)
db = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)

# Set up AI model
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
llm = HuggingFacePipeline(pipeline=local_pipeline)
model = ChatHuggingFace(llm=llm)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = getattr(result, "content", str(result)).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    documents_text = "\n".join(f"- {doc.page_content}" for doc in docs)
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {documents_text}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = getattr(result, "content", str(result)).strip()
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()