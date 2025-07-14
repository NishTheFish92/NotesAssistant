import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS

#This function splits, and embeds all the documents present in the corresponding text files.
#Returns a vector store
def embed_documents(output_path,embeddings):
    splitter = RecursiveCharacterTextSplitter()
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
        vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore