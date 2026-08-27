from __future__ import annotations

import asyncio

from pathlib import Path
from typing import List 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# =================================
# EMBEDDING MODEL
# =================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size = 50,
    retry_min_seconds=10
)

# =================================
# VECTOR STORE TODO: Change to ChromaDB
# =================================
# Always use project-root chroma, regardless of cwd
CHROMA_DIR = Path(__file__).resolve().parent / "chroma" / "data"
vectorstore = Chroma(
    collection_name="langchain-docs",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

# =================================
# TAVILY
# =================================
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 5, max_breadth = 20, max_pages = 1000)
tavily_crawl = TavilyCrawl()



async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOKUMENTATION INGESTION PIPELINE")
    log_info(
        "   TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE
    )
    
    # Crawl documentation site 
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,                                                     # How far tavily can explore
        "max_breadth": 50,
        "limit": 150,                                                          
        "extract_depth": "advanced",
        "categories": ["Documentation"],
    })
    skipped = [r.get("url") for r in res["results"] if not r.get("raw_content")]
    if skipped:
        log_warning(
            f"TavilyCrawl: Skipped {len(skipped)} pages with empty raw_content"
        )

    all_docs = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]},
        )
        for result in res["results"]
        if result.get("raw_content")
    ]

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URL from documentation site."
    )
    
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 200,
        )
    
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )
    
    # =================================
    # DOCUMENT INDEXING IN BATCHES
    # =================================    
    async def index_documents_async(documents: List[Document], batch_size: int = 50):
        """Process documents in batches asynchronously"""
        log_header("VECTOR STORAGE PHASE")
        log_info(
            f" Vector Store Indexing: Preparing to add {len(documents)} documetns to Vector Store",
            Colors.DARKCYAN
        )
        
        #create batches
        batches = [
            documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        # =================================
        # ADDING BATCHES 
        # =================================  
        async def add_batch(batch: List[Document], batch_num: int):
            try:
                await vectorstore.aadd_documents(batch)
                log_success(f"VecorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(documents)} documents)")
            except Exception as e:
                log_error(f"Vectore Store Indexing: Failed to add a batch {batch_num} - {e}")
                return False
            return True
                
        tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions = True)
        
        successfull = sum(1 for result in results if result is True)
        if successfull == len(batches): 
            log_success(
                f"vectorStore Indexing: All batches processed successfully! ({successfull} / {len(batches)})"
            )    
        else: 
            log_warning(
                f"VectorStore Indexing: Processed {successfull} / {len(batches)} batches successfully"
            )
    
    await index_documents_async(splitted_docs, batch_size = 500)
    
    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished successfully")
    log_info("Summary:", Colors.BOLD)
    # log_info(f"     - URLs mapped: {len(site_map['results'])}" )
    log_info(f"     -Documents extracted: {len(all_docs)}")
    log_info(f"     -Chunks created: {len(splitted_docs)}")

    
if __name__ == "__main__":
    asyncio.run(main())