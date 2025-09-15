# utils.py

"""
Utility functions and classes for environment setup, rate limiting,
token estimation, and file handling.
"""

import os
import time
import getpass
import base64
import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import tiktoken
from PIL import Image
from tqdm import tqdm
from langchain_core.documents import Document

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup ---
def setup_environment():
    """Load environment variables from .env file and prompt if not set."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY","LANGSMITH_TRACING"]
    for var in required_vars:
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"Please enter your {var}: ")
    

# --- Rate Limiter Class (from ingest_pipeline.ipynb) ---
class ThreadSafeHybridRateLimiter:
    """
    A thread-safe rate limiter that manages both token and request rates per minute.
    """
    # Refactored from ingest_pipeline.ipynb, cell 5.
    # No major logic changes, just added type hints and docstrings.
    def __init__(self, max_tokens_per_minute: int, max_requests_per_minute: int):
        self.max_tokens = max_tokens_per_minute
        self.max_requests = max_requests_per_minute
        self.token_bucket = deque()
        self.request_bucket = deque()
        self.lock = threading.Lock()

    def _cleanup_buckets(self, now: float):
        while self.token_bucket and now - self.token_bucket[0][0] > 60:
            self.token_bucket.popleft()
        while self.request_bucket and now - self.request_bucket[0] > 60:
            self.request_bucket.popleft()

    def check_and_consume(self, estimated_tokens: int):
        if estimated_tokens > self.max_tokens:
            raise ValueError(
                f"Task requires {estimated_tokens} tokens but max per minute is {self.max_tokens}"
            )
        while True:
            with self.lock:
                now = time.time()
                self._cleanup_buckets(now)
                total_tokens = sum(t for _, t in self.token_bucket)
                if (total_tokens + estimated_tokens <= self.max_tokens and
                        len(self.request_bucket) < self.max_requests):
                    self.token_bucket.append((now, estimated_tokens))
                    self.request_bucket.append(now)
                    return
            time.sleep(0.1)

    def update_actual_usage(self, estimated_tokens, actual_tokens):
        with self.lock:
            now = time.time()
            self._cleanup_buckets(now)
            # Remove last estimated token usage if needed
            for i in reversed(range(len(self.token_bucket))):
                if self.token_bucket[i][1] == estimated_tokens:
                    self.token_bucket.remove(self.token_bucket[i])
                    break
            # Add corrected token usage
            self.token_bucket.append((now, actual_tokens))

    def get_stats(self):
        with self.lock:
            now = time.time()
            self._cleanup_buckets(now)
            total_tokens = sum(t for ts, t in self.token_bucket)
            total_requests = len(self.request_bucket)
            return {
                "tokens_used_last_60s": total_tokens,
                "requests_used_last_60s": total_requests,
            }

# --- Tokenizer and Utilities ---
ENC = tiktoken.encoding_for_model("gpt-4o")

def estimate_tokens_from_messages(messages, model_name="gpt-4o"):
    total = 0
    for msg in messages:
        if hasattr(msg, "content"):   # HumanMessage, AIMessage, etc.
            text = msg.content
        elif isinstance(msg, dict):   # fallback for raw dicts
            text = msg.get("content", "")
        else:
            text = str(msg)
        total += len(ENC.encode(text))
    return total

# --- Document Handling ---
def save_docs_to_json(docs: List[Document], filename: str):
    """Saves a list of Document objects to a JSON file."""
    docs_list = [
        {"metadata": doc.metadata, "page_content": doc.page_content}
        for doc in docs
    ]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(docs_list, f, indent=2, ensure_ascii=False)
    logging.info(f"Successfully saved {len(docs)} documents to {filename}")

def load_docs_from_json(filename: str) -> List[Document]:
    """Loads a list of Document objects from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        docs_list = json.load(f)
    logging.info(f"Loading {len(docs_list)} documents from {filename}")
    return [Document(**item) for item in docs_list]

# --- Image Handling ---
def save_images_from_store(image_store: Dict[str, str], out_dir: str):
    """Decodes and saves base64 encoded images from a dictionary to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    for img_id, img_base64 in image_store.items():
        try:
            img_bytes = base64.b64decode(img_base64)
            file_path = os.path.join(out_dir, f"{img_id}.png")
            with open(file_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            logging.warning(f"Failed to save image {img_id}: {e}")
    logging.info(f"Saved {len(image_store)} images to '{out_dir}/'")
    
def encode_image_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_all_images(folder_path: str, extensions=("jpg", "jpeg", "png")) -> Dict[str, Image.Image]:
    """Reads all images from a folder into a dictionary of PIL Image objects."""
    images = {}
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    for f in image_files:
        try:
            key = os.path.splitext(os.path.basename(f))[0]
            images[key] = Image.open(os.path.join(folder_path, f))
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")
    logging.info(f"Loaded {len(images)} images from '{folder_path}/'")
    return images