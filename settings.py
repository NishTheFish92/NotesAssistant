"""
Central configuration for NotesAssistant. Change values here, not in the modules.
"""

# Paths
OUTPUT_PATH = "./output_folder"   # page images and extracted text
INDEX_PATH = "./faiss_index"      # saved FAISS vector store

# Ingestion
PDF_DPI = 200                     # render resolution for page images

# Models
LLM_MODEL = "auto"                # FreeLLMAPI model name; pin a specific model for evals
LLM_BASE_URL_DEFAULT = "http://localhost:3001/v1"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Chunking and retrieval
CHUNK_SIZE = 4000                 # characters (RecursiveCharacterTextSplitter default)
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 64             # lower this if you run out of GPU memory
TOP_K = 4                         # chunks retrieved per question
