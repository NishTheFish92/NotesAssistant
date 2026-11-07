"""
Document embedding utilities for NotesAssistant.
Splits and embeds text documents for retrieval.
"""

import math
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_OVERLAP, CHUNK_SIZE


def embed_documents(output_path, embeddings, batch_size=32):
    """
    Splits and embeds all text documents in the output_text folder.

    Args:
        output_path (str): Path to the folder containing text files.
        embeddings: Embedding model instance.
        batch_size (int): Number of documents to process in each batch.

    Returns:
        FAISS: Vector store containing embedded documents.
    """
    output_path = os.path.join(output_path, "output_text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    file_names = sorted([f for f in os.listdir(output_path) if f.endswith('.txt')])
    documents = []

    for file_name in file_names:
        with open(os.path.join(output_path, file_name), 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": file_name, "chunk_id": i}
            ))

    vectorstore = None
    num_batches = math.ceil(len(documents) / batch_size)

    for i in range(num_batches):
        batch_docs = documents[i*batch_size:(i+1)*batch_size]
        batch_vs = FAISS.from_documents(batch_docs, embeddings)
        if vectorstore is None:
            vectorstore = batch_vs
        else:
            vectorstore.merge_from(batch_vs)

    return vectorstore


def load_vectorstore(index_path, embeddings):
    """
    Loads a FAISS vector store saved by index.py.

    FAISS stores its docstore with pickle, so loading needs explicit opt-in.
    Only load indexes you created yourself.

    Args:
        index_path (str): Folder the index was saved to.
        embeddings: Embedding model instance (must match the one used to build it).

    Returns:
        FAISS: The loaded vector store.
    """
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
