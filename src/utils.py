"""
Utility functions for initializing LLM and embedding models for NotesAssistant.
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch

load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_kwargs = {'device': device}

def init_llm():
    """
    Initializes and returns a chat model served by FreeLLMAPI.

    FreeLLMAPI exposes an OpenAI-compatible endpoint, so ChatOpenAI is pointed
    at it via FREELLM_BASE_URL and authenticated with FREELLM_API_KEY.

    Returns:
        ChatOpenAI: The initialized LLM instance.
    """
    return ChatOpenAI(
        model="auto",
        api_key=os.getenv("FREELLM_API_KEY"),
        base_url=os.getenv("FREELLM_BASE_URL", "http://localhost:3001/v1"),
        temperature=0,
        max_tokens=None,
        max_retries=2,
    )

def init_embeddings():
    """
    Initializes and returns HuggingFace BGE embeddings.

    Sets up the embedding model with CUDA device and normalization.

    Returns:
        HuggingFaceBgeEmbeddings: The initialized embeddings instance.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for answering questions:"
    )