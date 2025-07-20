"""
NotesAssistant package initialization.
Provides functionality for processing lecture notes and slides.
"""

from .utils import init_llm, init_embeddings
from .create_chains import chainer, print_response
from .document_embedder import embed_documents
from .img_extract import img_to_txt
from .parse_to_image import pdf_to_images

__all__ = [
    'init_llm',
    'init_embeddings',
    'chainer',
    'print_response',
    'embed_documents',
    'img_to_txt',
    'pdf_to_images',
]