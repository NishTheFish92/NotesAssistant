"""
Central configuration for NotesAssistant. Change values here, not in the modules.
"""

# Paths
OUTPUT_PATH = "./output_folder"   # page images and extracted text
INDEX_PATH = "./faiss_index"      # saved FAISS vector store
CACHE_PATH = "./cache/pages"      # extracted page text, reused across runs

# Ingestion
PDF_DPI = 200                     # render resolution for page images

# Models
LLM_MODEL = "auto:notesassistant"                # FreeLLMAPI model name; pin a specific model for evals
LLM_BASE_URL_DEFAULT = "http://localhost:3001/v1"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Chunking and retrieval
CHUNK_IN_TOKENS = True            # False = old character split (use 4000 / 200 below), kept for the comparison
CHUNK_SIZE = 450                  # BGE tokens; BGE truncates anything past 512
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 64             # lower this if you run out of GPU memory
TOP_K = 4                         # chunks retrieved per question
