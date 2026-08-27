from __future__  import annotations

import os
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings 


load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER","Ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain-docs")
CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma" / "data"          
TOP_K = int(os.getenv("TOP_K", "4"))


llm = init_chat_model(
    model_provider = MODEL_PROVIDER,
    model = LLM_MODEL, 
    temperature = 0) 

embeddings = OpenAIEmbeddings(
    model = EMBEDDINGS_MODEL
    )

vectorstore = Chroma(
    collection_name = COLLECTION_NAME,
    embedding_function = embeddings,
    persist_directory = str(CHROMA_DIR),
)


# response_format "context and artifact" returns 2 values, "context" returns only one
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to hepl ansewer user queries about LangChain"""
    
    # Retrieve 4 most similar documents
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": TOP_K}).invoke(query)
    
    # Serializing for the model
    serialized = "\n\n".join(
        (f"Source {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return serialized and raw docs
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, any]: 
    """_summary_

    Run the RAG Pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question 

    Returns:
        Dict[str, any]: 
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    
    # Create the agent with retrieval tool
    system_prompt = (
        "You are helpful AI assistant that answers question about LangChain documentation."
        "You have access to a tool that rerieves relevant documentation."
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so"
    )
    
    # Create an Agent
    agent = create_agent(llm, tools = [retrieve_context], system_prompt = system_prompt)
    
    # Build Message list[dict]
    message = [
        {"role": "user", "content": query}
        ]
    
    # Invoke the Agent with dict
    response = agent.invoke({"messages":message})
    
    # Extract the answer from last AI Message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                # extend for docs in one layer
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs
    }
    
    
if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== CONTEXT DOCS ===")
    print(f"count: {len(result['context'])}")
    for i, doc in enumerate(result["context"], start=1):
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n[{i}] {source}")
        print(f"    {preview}...")
