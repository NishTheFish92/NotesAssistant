"""
Utility functions for initializing LLM and embedding models for NotesAssistant.
"""

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
def init_llm():
    """
    Initializes and returns a Google Generative AI chat model.

    Loads the API key from environment variables and sets up the model
    with specific parameters.

    Returns:
        ChatGoogleGenerativeAI: The initialized LLM instance.
    """
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        convert_system_message_to_human=True,
        temperature=0.0,
    )

def init_embeddings():
    """
    Initializes and returns HuggingFace BGE embeddings.

    Sets up the embedding model with CUDA device and normalization.

    Returns:
        HuggingFaceBgeEmbeddings: The initialized embeddings instance.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for answering questions:"
    )