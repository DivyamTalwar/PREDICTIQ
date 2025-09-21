import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import asyncio

from .vectorstore import PineconeVectorStore
from .embeddings import EmbeddingsManager
from app.parsers import PDFProcessor
from app.db.crud import DocumentCacheCRUD, create_document_cache
from app.db.models import DocumentCache
from app.db.database import db_manager

logger = logging.getLogger(__name__)

class DocumentIndexer:
    """
    Document indexing system for financial documents.

    Handles the complete pipeline from document processing to vector indexing,
    including batch processing, progress tracking, and error recovery.
    """

    def __init__(self):
        self.vectorstore = PineconeVectorStore()
        self.embeddings_manager = EmbeddingsManager()
        self.pdf_processor = PDFProcessor()

        # Batch processing settings
        self.batch_size = 50  # Chunks per batch
        self.max_retries = 3

        # Progress tracking
        self.current_operation = None
        self.total_documents = 0
        self.processed_documents = 0
        self.total_chunks = 0
        self.indexed_chunks = 0

    def index_document_from_file(
        self,
        file_path: Path,
        document_type: str = "quarterly_report",
        company: str = "TCS",
        quarter: Optional[int] = None,
        fiscal_year: Optional[int] = None,
        namespace: Optional[str] = None,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index a single document file into the vector store.

        Args:
            file_path: Path to the document file
            document_type: Type of document
            company: Company name
            quarter: Quarter number (1-4)
            fiscal_year: Fiscal year
            namespace: Vector store namespace (auto-determined if None)
            force_reindex: Whether to reindex if already indexed

        Returns:
            Indexing results dictionary
        """
        logger.info(f"Starting indexing for document: {file_path.name}")

        try:
            # Check if document already indexed
            file_hash = self._calculate_file_hash(file_path)

            with db_manager.get_session_context() as db:
                existing_doc = DocumentCacheCRUD.get_by_hash(db, file_hash)

                if existing_doc and existing_doc.is_processed and not force_reindex:
                    logger.info(f"Document already indexed: {file_path.name}")
                    return {
                        "success": True,
                        "message": "Document already indexed",
                        "document_id": existing_doc.id,
                        "chunks_count": existing_doc.chunks_count
                    }

            # Process document with PDF processor
            processed_data = self.pdf_processor.process_pdf(file_path)
            if not processed_data:
                return {
                    "success": False,
                    "error": "Failed to process PDF"
                }

            # Store in document cache
            with db_manager.get_session_context() as db:
                doc_cache = create_document_cache(
                    db=db,
                    document_type=document_type,
                    file_path=str(file_path),
                    file_hash=file_hash,
                    file_size_bytes=file_path.stat().st_size,
                    original_filename=file_path.name,
                    doc_metadata=processed_data["metadata"],
                    company=company,
                    quarter=quarter,
                    fiscal_year=fiscal_year
                )

            # Generate embeddings for chunks
            chunks = processed_data["chunks"]
            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            enhanced_chunks = self.embeddings_manager.embed_document_chunks(
                chunks,
                text_field="text",
                use_cache=True
            )

            # Determine namespace
            if not namespace:
                namespace = self.vectorstore.get_namespace_for_document_type(document_type)

            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            successful_chunks = 0

            for chunk in enhanced_chunks:
                if "embedding" in chunk:
                    vector_id = self.vectorstore.create_document_id(
                        document_name=file_path.stem,
                        chunk_index=chunk["metadata"]["chunk_index"],
                        document_type=document_type
                    )

                    metadata = self.vectorstore.create_chunk_metadata(
                        document_name=file_path.name,
                        chunk_index=chunk["metadata"]["chunk_index"],
                        chunk_text=chunk["text"],
                        document_type=document_type,
                        quarter=quarter,
                        fiscal_year=fiscal_year,
                        document_id=doc_cache.id,
                        **chunk["metadata"]
                    )

                    vectors_to_upsert.append((
                        vector_id,
                        chunk["embedding"],
                        metadata
                    ))
                    successful_chunks += 1

            if not vectors_to_upsert:
                return {
                    "success": False,
                    "error": "No valid embeddings generated"
                }

            # Upsert to Pinecone
            upsert_result = self.vectorstore.upsert_vectors(
                vectors=vectors_to_upsert,
                namespace=namespace,
                batch_size=self.batch_size
            )

            if not upsert_result.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to upsert vectors: {upsert_result.get('error')}"
                }

            # Update document cache
            with db_manager.get_session_context() as db:
                DocumentCacheCRUD.mark_processed(
                    db=db,
                    document_id=doc_cache.id,
                    chunks_count=successful_chunks
                )

            logger.info(f"Successfully indexed document: {file_path.name} ({successful_chunks} chunks)")

            return {
                "success": True,
                "document_id": doc_cache.id,
                "file_name": file_path.name,
                "total_chunks": len(chunks),
                "indexed_chunks": successful_chunks,
                "namespace": namespace,
                "embeddings_cost": self.embeddings_manager.get_cost_stats()
            }

        except Exception as e:
            logger.error(f"Error indexing document {file_path.name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def index_multiple_documents(
        self,
        file_paths: List[Path],
        document_type: str = "quarterly_report",
        company: str = "TCS",
        force_reindex: bool = False,
        parallel_processing: bool = False
    ) -> Dict[str, Any]:
        """
        Index multiple documents with progress tracking.

        Args:
            file_paths: List of file paths to index
            document_type: Type of documents
            company: Company name
            force_reindex: Whether to reindex existing documents
            parallel_processing: Whether to process documents in parallel

        Returns:
            Batch indexing results
        """
        logger.info(f"Starting batch indexing for {len(file_paths)} documents")

        self.current_operation = "batch_indexing"
        self.total_documents = len(file_paths)
        self.processed_documents = 0

        results = {
            "success": True,
            "total_documents": len(file_paths),
            "processed_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "indexed_chunks": 0,
            "errors": [],
            "embeddings_cost": {},
            "processing_time": 0
        }

        start_time = datetime.now()

        try:
            if parallel_processing and len(file_paths) > 1:
                # Parallel processing (limited concurrency to avoid rate limits)
                results = asyncio.run(self._process_documents_parallel(
                    file_paths, document_type, company, force_reindex, results
                ))
            else:
                # Sequential processing
                for file_path in file_paths:
                    try:
                        # Extract metadata from filename if possible
                        quarter, fiscal_year = self._extract_metadata_from_filename(file_path.name)

                        result = self.index_document_from_file(
                            file_path=file_path,
                            document_type=document_type,
                            company=company,
                            quarter=quarter,
                            fiscal_year=fiscal_year,
                            force_reindex=force_reindex
                        )

                        self.processed_documents += 1
                        results["processed_documents"] = self.processed_documents

                        if result["success"]:
                            results["successful_documents"] += 1
                            results["total_chunks"] += result.get("total_chunks", 0)
                            results["indexed_chunks"] += result.get("indexed_chunks", 0)
                        else:
                            results["failed_documents"] += 1
                            results["errors"].append({
                                "file": file_path.name,
                                "error": result.get("error", "Unknown error")
                            })

                        # Log progress
                        progress = (self.processed_documents / self.total_documents) * 100
                        logger.info(f"Batch indexing progress: {progress:.1f}% ({self.processed_documents}/{self.total_documents})")

                    except Exception as e:
                        logger.error(f"Error processing {file_path.name}: {e}")
                        results["failed_documents"] += 1
                        results["errors"].append({
                            "file": file_path.name,
                            "error": str(e)
                        })

            # Final statistics
            end_time = datetime.now()
            results["processing_time"] = (end_time - start_time).total_seconds()
            results["embeddings_cost"] = self.embeddings_manager.get_cost_stats()

            if results["failed_documents"] > 0:
                results["success"] = False

            logger.info(f"Batch indexing completed: {results['successful_documents']}/{results['total_documents']} successful")

        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
            results["success"] = False
            results["errors"].append({"general": str(e)})

        finally:
            self.current_operation = None

        return results

    async def _process_documents_parallel(
        self,
        file_paths: List[Path],
        document_type: str,
        company: str,
        force_reindex: bool,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process documents in parallel with limited concurrency."""
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent documents

        async def process_single_document(file_path: Path):
            async with semaphore:
                # Run in thread pool since our indexing is synchronous
                loop = asyncio.get_event_loop()
                quarter, fiscal_year = self._extract_metadata_from_filename(file_path.name)

                return await loop.run_in_executor(
                    None,
                    self.index_document_from_file,
                    file_path,
                    document_type,
                    company,
                    quarter,
                    fiscal_year,
                    None,  # namespace
                    force_reindex
                )

        # Create tasks for all documents
        tasks = [process_single_document(fp) for fp in file_paths]

        # Process tasks and collect results
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                self.processed_documents += 1
                results["processed_documents"] = self.processed_documents

                if result["success"]:
                    results["successful_documents"] += 1
                    results["total_chunks"] += result.get("total_chunks", 0)
                    results["indexed_chunks"] += result.get("indexed_chunks", 0)
                else:
                    results["failed_documents"] += 1
                    results["errors"].append({
                        "file": result.get("file_name", f"document_{i}"),
                        "error": result.get("error", "Unknown error")
                    })

            except Exception as e:
                results["failed_documents"] += 1
                results["errors"].append({
                    "file": f"document_{i}",
                    "error": str(e)
                })

        return results

    def _extract_metadata_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract quarter and fiscal year from filename."""
        import re

        quarter = None
        fiscal_year = None

        # Look for Q1, Q2, Q3, Q4
        quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))

        # Look for FY or year patterns
        year_match = re.search(r'(?:FY|20)(\d{2,4})', filename, re.IGNORECASE)
        if year_match:
            year_str = year_match.group(1)
            if len(year_str) == 2:
                fiscal_year = 2000 + int(year_str)
            else:
                fiscal_year = int(year_str)

        return quarter, fiscal_year

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file for duplicate detection."""
        import hashlib

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def reindex_document(self, document_id: int) -> Dict[str, Any]:
        """Reindex a specific document by ID."""
        try:
            with db_manager.get_session_context() as db:
                doc = DocumentCacheCRUD.get_by_id(db, document_id)
                if not doc:
                    return {"success": False, "error": "Document not found"}

                file_path = Path(doc.file_path)
                if not file_path.exists():
                    return {"success": False, "error": "File not found"}

                return self.index_document_from_file(
                    file_path=file_path,
                    document_type=doc.document_type,
                    company=doc.company,
                    quarter=doc.quarter,
                    fiscal_year=doc.fiscal_year,
                    force_reindex=True
                )

        except Exception as e:
            logger.error(f"Error reindexing document {document_id}: {e}")
            return {"success": False, "error": str(e)}

    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents."""
        try:
            with db_manager.get_session_context() as db:
                stats = {
                    "total_documents": 0,
                    "processed_documents": 0,
                    "total_chunks": 0,
                    "by_document_type": {},
                    "by_company": {},
                    "by_fiscal_year": {},
                    "recent_activity": []
                }

                # Get document counts
                all_docs = db.query(DocumentCache).all()
                stats["total_documents"] = len(all_docs)

                processed_docs = [doc for doc in all_docs if doc.is_processed]
                stats["processed_documents"] = len(processed_docs)

                for doc in processed_docs:
                    stats["total_chunks"] += doc.chunks_count or 0

                    # Count by document type
                    doc_type = doc.document_type
                    stats["by_document_type"][doc_type] = stats["by_document_type"].get(doc_type, 0) + 1

                    # Count by company
                    company = doc.company or "Unknown"
                    stats["by_company"][company] = stats["by_company"].get(company, 0) + 1

                    # Count by fiscal year
                    fy = doc.fiscal_year or "Unknown"
                    stats["by_fiscal_year"][str(fy)] = stats["by_fiscal_year"].get(str(fy), 0) + 1

                # Get recent activity (last 10 processed documents)
                recent_docs = db.query(DocumentCache).filter(
                    DocumentCache.is_processed == True
                ).order_by(DocumentCache.download_timestamp.desc()).limit(10).all()

                for doc in recent_docs:
                    stats["recent_activity"].append({
                        "id": doc.id,
                        "filename": doc.original_filename,
                        "document_type": doc.document_type,
                        "chunks_count": doc.chunks_count,
                        "processed_at": doc.download_timestamp.isoformat()
                    })

                return stats

        except Exception as e:
            logger.error(f"Error getting indexing stats: {e}")
            return {"error": str(e)}

    def search_indexed_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        namespace: str = "general"
    ) -> List[Dict[str, Any]]:
        """Search indexed documents using the retriever."""
        from .retriever import DocumentRetriever

        retriever = DocumentRetriever()
        return retriever.retrieve_documents(
            query=query,
            top_k=top_k,
            namespace=namespace
        )

    def get_processing_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        if not self.current_operation:
            return {"status": "idle"}

        progress = {
            "status": "processing",
            "operation": self.current_operation,
            "progress_percentage": 0
        }

        if self.total_documents > 0:
            progress["progress_percentage"] = (self.processed_documents / self.total_documents) * 100
            progress["processed_documents"] = self.processed_documents
            progress["total_documents"] = self.total_documents

        if self.total_chunks > 0:
            progress["chunks_progress"] = (self.indexed_chunks / self.total_chunks) * 100
            progress["indexed_chunks"] = self.indexed_chunks
            progress["total_chunks"] = self.total_chunks

        return progress

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on indexing system."""
        vectorstore_health = self.vectorstore.health_check()
        embeddings_health = self.embeddings_manager.health_check()

        indexer_health = {
            "indexer": {
                "status": "healthy" if (
                    vectorstore_health["pinecone"]["status"] in ["healthy", "not_configured"] and
                    embeddings_health["embeddings"]["status"] in ["healthy", "not_configured"]
                ) else "unhealthy",
                "pdf_processor_available": bool(self.pdf_processor),
                "current_operation": self.current_operation,
                "stats": self.get_indexing_stats()
            }
        }

        return {**vectorstore_health, **embeddings_health, **indexer_health}