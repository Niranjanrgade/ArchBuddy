#!/usr/bin/env python3
"""
Azure Documentation Embedding Script
Embeds Azure docs from local repo into ChromaDB using Ollama (nomic-embed-text).
"""

import os
import sys
import glob
import time
import yaml
import logging
import argparse
import gc
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Connect to existing ChromaDB
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("embed_azure_docs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Domain -> Directory mapping
DOMAIN_MAP = {
    "compute": [
        "virtual-machines", "app-service", "azure-functions", "container-apps",
        "aks", "container-instances", "virtual-machine-scale-sets", "batch"
    ],
    "network": [
        "virtual-network", "load-balancer", "firewall", "frontdoor", 
        "traffic-manager", "application-gateway", "private-link", 
        "web-application-firewall", "networking", "virtual-network-manager"
    ],
    "storage": [
        "storage", "backup", "data-lake-store"
    ],
    "database": [
        "azure-sql", "cosmos-db", "postgresql", "mysql", 
        "azure-cache-for-redis", "redis"
    ]
}

# Directories to always skip
SKIP_DIRS = {
    "includes", "media", "bread"
}

# Files to always skip
SKIP_FILES = {
    "TOC.yml", "toc.yml", "index.yml", "zone-pivot-groups.yml"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_markdown_with_frontmatter(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse a markdown file, extracting YAML frontmatter and content.
    Returns dict with 'content', 'title', 'service' or None if invalid.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        # Check if file has frontmatter
        if not raw_content.startswith("---"):
            # No frontmatter, just return content with filename as title
            return {
                "content": raw_content,
                "title": os.path.basename(file_path).replace(".md", "").replace("-", " ").title(),
                "service": "unknown"
            }
        
        # Split frontmatter and content
        parts = raw_content.split("---", 2)
        if len(parts) < 3:
            return None  # Malformed
            
        frontmatter_str = parts[1]
        content = parts[2].strip()
        
        if len(content) < 100:
            return None  # Skip empty/stub files
            
        try:
            metadata = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError:
            metadata = {}
            
        return {
            "content": content,
            "title": metadata.get("title", os.path.basename(file_path)),
            "service": metadata.get("ms.service", "azure")
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return None

def get_splitter():
    """Create markdown-aware text splitter"""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n## ",     # H2 headers
            "\n### ",    # H3 headers
            "\n#### ",   # H4 headers
            "\n\n",      # Paragraphs
            "\n",        # Line breaks
            ". ",        # Sentences
            " ",         # Words
            ""
        ]
    )

# ============================================================================
# MAIN INGESTION LOGIC
# ============================================================================

def process_directory(
    base_path: str,
    service_dir: str, 
    domain: str,
    db: Chroma,
    batch_size: int = 20
):
    """
    Process a single service directory.
    """
    full_path = os.path.join(base_path, service_dir)
    if not os.path.exists(full_path):
        logger.warning(f"Directory not found (skipping): {full_path}")
        return

    # Find all .md files
    files = glob.glob(os.path.join(full_path, "**/*.md"), recursive=True)
    if not files:
        logger.info(f"No markdown files in {service_dir}")
        return

    logger.info(f"Processing {service_dir} ({len(files)} files)...")
    
    splitter = get_splitter()
    documents_to_embed = []
    
    # Check if we should skip this directory if already fully ingested?
    # Hard to know "fully", so we'll check individual chunks or just resume.
    # For now, simplistic check: if DB has > 0 docs for this service, user might want to skip.
    # But let's assume valid upsert or just append.
    
    # 1. Parse and Chunk
    for file_path in tqdm(files, desc=f"Parsing {service_dir}", unit="file"):
        if any(skip in file_path for skip in SKIP_DIRS):
            continue
        if os.path.basename(file_path) in SKIP_FILES:
            continue
            
        data = parse_markdown_with_frontmatter(file_path)
        if not data:
            continue
            
        # Create base doc
        doc = Document(
            page_content=data["content"],
            metadata={
                "source": file_path,
                "domain": domain,
                "service": service_dir, # Use dir name as consistent service tag
                "title": data["title"],
                "ms_service": data["service"]
            }
        )
        
        # Split
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            documents_to_embed.append(chunk)

    if not documents_to_embed:
        return

    # 2. Embed and Store in Batches
    total_batches = (len(documents_to_embed) + batch_size - 1) // batch_size
    logger.info(f"Embedding {len(documents_to_embed)} chunks in {total_batches} batches...")
    
    for i in tqdm(range(0, len(documents_to_embed), batch_size), desc=f"Embedding {service_dir}", unit="batch"):
        batch = documents_to_embed[i:i+batch_size]
        
        retry_count = 0
        max_retries = 3
        while retry_count <= max_retries:
            try:
                db.add_documents(batch)
                break
            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count
                logger.warning(f"Batch failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time}s. Error: {e}")
                time.sleep(wait_time)
                if retry_count > max_retries:
                    logger.error(f"Failed to ingest batch starting at index {i} for {service_dir}. Skipping.")
        
        # Memory optimization: excessive sleep + GC
        time.sleep(0.5) 

    # Force cleanup after each directory
    del documents_to_embed
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Embed Azure Docs")
    parser.add_argument("--docs-path", default=os.path.expanduser("~/Desktop/Projects/MTech/azure-docs/articles"), help="Path to articles")
    parser.add_argument("--output-dir", default="./chroma_db_AzureDocs", help="ChromaDB persist directory")
    parser.add_argument("--domains", default="all", help="Comma-separated domains (compute,network,storage,database) or 'all'")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for embedding")
    
    args = parser.parse_args()
    
    # 1. Select Domains
    if args.domains == "all":
        target_domains = DOMAIN_MAP.keys()
    else:
        target_domains = args.domains.split(",")

    # 2. Initialize Chroma + Ollama
    logger.info("Initializing OllamaEmbeddings (model='nomic-embed-text')...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    logger.info(f"Initializing ChromaDB at {args.output_dir}...")
    db = Chroma(
        collection_name="AzureDocs",
        persist_directory=args.output_dir,
        embedding_function=embeddings
    )

    # 3. Process
    for domain in target_domains:
        if domain not in DOMAIN_MAP:
            logger.warning(f"Unknown domain: {domain}")
            continue
            
        logger.info(f"=== PROCESSING DOMAIN: {domain.upper()} ===")
        directories = DOMAIN_MAP[domain]
        
        for service_dir in directories:
            process_directory(
                base_path=args.docs_path,
                service_dir=service_dir,
                domain=domain,
                db=db,
                batch_size=args.batch_size
            )
            
            # Extra pause between directories
            time.sleep(2.0)
            gc.collect()
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
