# config.py

"""
Central configuration file for the Multimodal RAG Assistant.
Stores model names, API limits, file paths, and other constants.
"""

# --- Language and Embedding Model Configurations ---
LLM_MODEL = "gpt-4o-mini"
HIGH_PERF_LLM_MODEL = "gpt-4"
LLM_TEMP = 0.0
EMBEDDING_MODEL = "llama3.2:1b"
RERANKER_MODEL = "BAAI/bge-reranker-large"
STT_MODEL_SIZE = "small" # For speech-to-text: tiny, base, small, medium, large-v3

# --- Text Processing Configurations ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- API Rate Limiting Configurations ---
TPM_LIMIT = 180_000  # Tokens Per Minute
RPM_LIMIT = 400      # Requests Per Minute
MAX_WORKERS_SUMMARY = 20
ESTIMATED_SUMMARY_TOKENS = 1000

# --- Retriever Configurations ---
RERANKER_TOP_N = 5
ENSEMBLE_WEIGHTS = [0.8, 0.2]  # [text_weight, image_weight]
TEXT_RETRIEVER_K = 8
IMAGE_RETRIEVER_K = 2

# --- File and Directory Paths ---
PDF_PATH = "docs/1910013384_Archer C50(EU)_UG_V1.pdf"
OUTPUT_DOCS_PATH = "processed_docs.json"
OUTPUT_IMAGE_DIR = "all_images"
FAISS_INDEX_PATH = "faiss_index_text"
FAISS_IMG_INDEX_PATH = "faiss_index_img"

# --- LangSmith Configuration ---
LANGCHAIN_PROJECT = "M_RAG_PROJ"