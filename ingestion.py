# ingestion.py

"""
Data ingestion pipeline for processing PDFs into text and image summaries,
and creating vector stores for retrieval.
"""

import base64
import io
import logging
import re
from typing import List, Dict, Tuple, Any

import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_community.callbacks.manager import get_openai_callback

# Local imports
import config
from utils import (
    setup_environment, save_docs_to_json, save_images_from_store, 
    ThreadSafeHybridRateLimiter, estimate_tokens_from_messages
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_and_images(pdf_path: str) -> Tuple[List[Document], Dict[str, str]]:
    """
    Extracts text chunks and base64 encoded images from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        A tuple containing a list of text documents and a dictionary of image data.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    doc = fitz.open(pdf_path)
    text_docs, image_data_store = [], {}
    logging.info(f"Processing PDF: {pdf_path}")

    for i, page in enumerate(doc):
        # Extract and chunk text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_docs.extend(text_splitter.split_documents([temp_doc]))

        # Extract and encode images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                image_id = f"page_{i}_img_{img_index}"
                image_data_store[image_id] = img_base64
            except Exception as e:
                logging.error(f"Error processing image {img_index} on page {i}: {e}")

    doc.close()
    logging.info(f"Extracted {len(text_docs)} text chunks and {len(image_data_store)} images.")
    return text_docs, image_data_store


def summarize_images(
    image_store: Dict[str, str],
    llm: ChatOpenAI,
    prompt: str
) -> List[Document]:
    """
    Summarizes images in parallel using an LLM, respecting API rate limits.
    """
    limiter = ThreadSafeHybridRateLimiter(
        max_tokens_per_minute=config.TPM_LIMIT,
        max_requests_per_minute=config.RPM_LIMIT
    )
    image_docs = []

    def process_image(img_id: str, img_b64: str) -> Dict[str, Any]:
        """Wrapper function for concurrent image processing."""
        try:
            # Token estimation for rate limiting
            # GPT-4o cost is complex. A rough estimation: text tokens + fixed cost per image.

            estimated_response_tokens = 1000  # A rough estimate per image
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
            messages = HumanMessage(content=content)
            prompt_tokens = estimate_tokens_from_messages(messages)
            total_estimated = prompt_tokens + estimated_response_tokens

            # ⏳ Rate limit check
            limiter.check_and_consume(total_estimated)

            with get_openai_callback() as cb:
                response = llm.invoke([messages])

            limiter.update_actual_usage(total_estimated,cb.total_tokens)
            return {"id": img_id, "summary": response.content.strip()}
        
        except Exception as e:
            logging.error(f"Skipping image {img_id} due to error: {e}")
            return {"id": img_id, "summary": "Error during summarization."}

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_SUMMARY) as executor:
        futures = [executor.submit(process_image, img_id, img_b64) for img_id, img_b64 in image_store.items()]
        
        with tqdm(total=len(futures), desc="Summarizing images") as pbar:
            for future in as_completed(futures):
                result = future.result()

                if "Error" not in result["summary"]:
                    page_num_match = re.search(r"page_(\d+)_", result["id"])
                    page_num = int(page_num_match.group(1)) if page_num_match else -1
                    
                    img_doc = Document(
                        page_content=result["summary"],
                        metadata={"page": page_num, "type": "image", "image_id": result["id"]}
                    )
                    image_docs.append(img_doc)

                stats = limiter.get_stats()
                pbar.set_description(
                    f"TOK {stats['tokens_used_last_60s']:,}/{limiter.max_tokens:,} | "
                    f"REQ {stats['requests_used_last_60s']:,}/{limiter.max_requests:,}"
                )
                pbar.update(1)
                

    logging.info(f"Successfully summarized {len(image_docs)} images.")
    return image_docs

def create_and_save_vector_stores(
    text_docs: List[Document],
    image_docs: List[Document]
):
    """Creates and saves FAISS vector stores for text and image documents."""
    logging.info("Initializing embedding model...")
    embedding_model = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    logging.info("Creating text vector store...")
    vectorstore_text = FAISS.from_documents(text_docs, embedding_model)
    vectorstore_text.save_local(config.FAISS_INDEX_PATH)
    logging.info(f"Text vector store saved to '{config.FAISS_INDEX_PATH}'")

    logging.info("Creating image vector store...")
    vectorstore_img = FAISS.from_documents(image_docs, embedding_model)
    vectorstore_img.save_local(config.FAISS_IMG_INDEX_PATH)
    logging.info(f"Image vector store saved to '{config.FAISS_IMG_INDEX_PATH}'")


def run_ingestion_pipeline(pdf_path: str):
    """
    Main function to run the complete data ingestion and processing pipeline.
    """
    # Step 1: Initialize environment and models
    setup_environment()
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

    # Step 2: Extract data from PDF
    text_docs, image_data_store = extract_text_and_images(pdf_path)

    # Step 3: Summarize images
    summarization_prompt = (
        "You are an assistant tasked with summarizing images for optimal retrieval. "
        "These summaries will be embedded and used to retrieve the raw image. "
        "Write a clear and concise summary that captures all important information, "
        "including any text, statistics, or key points present in the image."
    )
    
    # image_data_store_partial={}
    # for key, value in image_data_store.items():
    #     if len(image_data_store_partial) < 30:
    #         image_data_store_partial[key] = value
    #     else:
    #         break # Stop after we've collected 30 items

    image_docs = summarize_images(image_data_store, llm, summarization_prompt)

    # Step 4: Consolidate and save artifacts
    all_docs = text_docs + image_docs
    save_docs_to_json(all_docs, config.OUTPUT_DOCS_PATH)
    save_images_from_store(image_data_store, config.OUTPUT_IMAGE_DIR)

    # Step 5: Create and save vector stores
    create_and_save_vector_stores(text_docs, image_docs)
    logging.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    run_ingestion_pipeline(pdf_path=config.PDF_PATH)