# MRAG Project

MRAG (Multi-Modal Retrieval Augmented Generation) is a Python-based application designed for document ingestion, processing, and retrieval augmented generation (RAG) using both text and image data. The project leverages FAISS for efficient similarity search and supports handling large collections of documents and associated images.

## Features

- **Document Ingestion:** Import and preprocess documents for downstream tasks.
- **Image Handling:** Store and manage images associated with document pages.
- **FAISS Indexing:** Fast similarity search for both text and image embeddings.
- **Processed Data Storage:** Store processed document metadata in JSON format.
- **Utility Functions:** Common utilities for file and data management.

## Project Structure

```
config.py              # Configuration settings for the project
ingestion.py           # Document ingestion and preprocessing logic
mrag_app.py            # Main application entry point
processed_docs.json    # Stores processed document metadata
utils.py               # Utility functions
all_images/            # Directory containing all page images
faiss_index_img/       # FAISS index for image embeddings
faiss_index_text/      # FAISS index for text embeddings
docs/                  # Documentation and additional resources
```

## Getting Started

### Prerequisites

- Python 3.10+
- Recommended: Create a virtual environment

### Installation

1. Clone the repository:
   ```powershell
   git clone <repo-url>
   cd MRAG
   ```
2. Install required packages

### Usage

- **Ingest Documents:**
  Run the ingestion script to process documents:
  ```powershell
  python ingestion.py
  ```
- **Run Main Application:**
  ```powershell
  python mrag_app.py
  ```

## Directory Details

- `all_images/`: Contains PNG images for each document page.
- `faiss_index_img/` and `faiss_index_text/`: Store FAISS indices for fast retrieval.
- `processed_docs.json`: Metadata and processed information about ingested documents.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

[MIT License](LICENSE)
