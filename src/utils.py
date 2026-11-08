"""
Utility functions for initializing LLM and embedding models for NotesAssistant.
"""

import os

import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from settings import (
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_INSTRUCTION,
    LLM_BASE_URL_DEFAULT,
    LLM_MODEL,
)

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_llm(max_retries=2):
    """
    Initializes and returns a chat model served by FreeLLMAPI.

    FreeLLMAPI exposes an OpenAI-compatible endpoint, so ChatOpenAI is pointed
    at it via FREELLM_BASE_URL and authenticated with FREELLM_API_KEY.

    Args:
        max_retries (int): Retries done by the OpenAI client itself. Ingestion
            passes 0 because img_to_llm already has its own retry policy.

    Returns:
        ChatOpenAI: The initialized LLM instance.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.getenv("FREELLM_API_KEY"),
        base_url=os.getenv("FREELLM_BASE_URL", LLM_BASE_URL_DEFAULT),
        temperature=0,
        max_tokens=None,
        max_retries=max_retries,
    )


def init_embeddings():
    """
    Initializes and returns BGE embeddings on GPU if available.

    Documents are embedded as-is; queries get BGE's retrieval instruction
    prepended, which is how BGE v1.5 is meant to be used for search.

    Returns:
        HuggingFaceEmbeddings: The initialized embeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": EMBEDDING_QUERY_INSTRUCTION,
        },
    )
