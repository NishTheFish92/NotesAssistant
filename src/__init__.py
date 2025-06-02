"""
NotesAssistant package initialization.
Provides functionality for processing lecture notes and slides.
"""

from .config import init_llm, init_embeddings
from .create_chains import chainer, print_response
from .document_processor import process_documents
from .llm_handler import img2txt
from .parse_to_image import toimages

__all__ = [
    'init_llm',
    'init_embeddings',
    'chainer',
    'print_response',
    'process_documents',
    'img2txt',
    'toimages',
]