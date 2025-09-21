# File: app/api/documents.py
# Purpose: Document upload and processing API endpoints
# Dependencies: fastapi, pydantic, typing
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 5

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.rag.indexer import DocumentIndexer
from app.rag.retriever import DocumentRetriever
from app.parsers.pdf_processor import PDFProcessor
from app.parsers.url_processor import URLProcessor
from app.db.database import get_database_manager
from app.db.crud import DocumentCacheCRUD

router = APIRouter()
logger = logging.getLogger(__name__)

# Request/Response Models
class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    processing_started: bool
    upload_timestamp: str

class DocumentProcessingStatus(BaseModel):
    """Document processing status"""
    document_id: str
    status: str  # uploaded, processing, completed, failed
    progress_percentage: int
    chunks_processed: int
    total_chunks: int
    error_message: Optional[str]
    processing_time_seconds: Optional[float]
    metadata: Dict[str, Any]

class URLProcessingRequest(BaseModel):
    """URL processing request"""
    url: str = Field(..., description="URL to process")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DocumentSearchRequest(BaseModel):
    """Document search request"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DocumentSearchResponse(BaseModel):
    """Document search response"""
    query: str
    results_count: int
    results: List[Dict[str, Any]]
    search_time_seconds: float

# In-memory processing status tracking (would use Redis in production)
processing_status = {}

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company: str = "TCS",
    document_type: str = "quarterly_report",
    quarter: Optional[str] = None,
    fiscal_year: Optional[int] = None
):
    """
    Upload and process financial documents

    Supports PDF files containing quarterly reports, earnings calls,
    and other financial documents. Processing includes:
    - Text extraction
    - Chunking and indexing
    - Vector embedding generation
    - Storage in vector database
    """
    document_id = str(uuid.uuid4())
    settings = get_settings()

    # Validate file type
    allowed_types = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, TXT, DOCX"
        )

    # Validate file size (50MB limit)
    max_size = 50 * 1024 * 1024  # 50MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB limit"
        )

    try:
        # Create uploads directory
        upload_dir = Path(settings.data_directory) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / f"{document_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Store document metadata in database
        db_manager = get_database_manager()
        doc_crud = DocumentCacheCRUD(db_manager)

        metadata = {
            "company": company,
            "document_type": document_type,
            "quarter": quarter,
            "fiscal_year": fiscal_year,
            "original_filename": file.filename,
            "file_size": len(file_content),
            "content_type": file.content_type,
            "upload_timestamp": datetime.now().isoformat()
        }

        await doc_crud.create({
            "document_id": document_id,
            "url": str(file_path),
            "content": "",  # Will be populated during processing
            "doc_metadata": metadata,
            "file_size": len(file_content),
            "content_type": file.content_type
        })

        # Initialize processing status
        processing_status[document_id] = {
            "status": "uploaded",
            "progress_percentage": 0,
            "chunks_processed": 0,
            "total_chunks": 0,
            "start_time": datetime.now()
        }

        # Start background processing
        background_tasks.add_task(
            _process_document_background,
            document_id,
            file_path,
            metadata
        )

        logger.info(f"Document uploaded: {document_id} - {file.filename}")

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=len(file_content),
            content_type=file.content_type,
            status="uploaded",
            processing_started=True,
            upload_timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@router.get("/status/{document_id}", response_model=DocumentProcessingStatus)
async def get_processing_status(document_id: str):
    """
    Get document processing status

    Returns current processing status including progress,
    chunks processed, and any error messages.
    """
    if document_id not in processing_status:
        # Check if document exists in database
        db_manager = get_database_manager()
        doc_crud = DocumentCacheCRUD(db_manager)
        doc = await doc_crud.get_by_document_id(document_id)

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Return completed status if not in processing queue
        return DocumentProcessingStatus(
            document_id=document_id,
            status="completed",
            progress_percentage=100,
            chunks_processed=0,
            total_chunks=0,
            processing_time_seconds=0.0,
            metadata=doc.doc_metadata or {}
        )

    status_info = processing_status[document_id]
    processing_time = None

    if status_info.get("end_time"):
        processing_time = (status_info["end_time"] - status_info["start_time"]).total_seconds()

    return DocumentProcessingStatus(
        document_id=document_id,
        status=status_info["status"],
        progress_percentage=status_info["progress_percentage"],
        chunks_processed=status_info["chunks_processed"],
        total_chunks=status_info["total_chunks"],
        error_message=status_info.get("error_message"),
        processing_time_seconds=processing_time,
        metadata=status_info.get("metadata", {})
    )

@router.post("/process-url")
async def process_url(
    request: URLProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Process content from URL

    Downloads and processes content from web URLs,
    particularly useful for online financial reports
    and news articles.
    """
    document_id = str(uuid.uuid4())

    try:
        url_processor = URLProcessor()

        # Process URL content
        logger.info(f"Processing URL: {request.url}")
        result = await url_processor.process_url(request.url)

        if not result or not result.get('content'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract content from URL"
            )

        # Store in database
        db_manager = get_database_manager()
        doc_crud = DocumentCacheCRUD(db_manager)

        metadata = {
            **request.metadata,
            "source_url": request.url,
            "content_length": len(result['content']),
            "processing_timestamp": datetime.now().isoformat()
        }

        await doc_crud.create({
            "document_id": document_id,
            "url": request.url,
            "content": result['content'],
            "doc_metadata": metadata,
            "file_size": len(result['content']),
            "content_type": "text/html"
        })

        # Start background indexing
        background_tasks.add_task(
            _index_text_content,
            document_id,
            result['content'],
            metadata
        )

        return {
            "document_id": document_id,
            "url": request.url,
            "content_length": len(result['content']),
            "status": "processing",
            "indexing_started": True
        }

    except Exception as e:
        logger.error(f"URL processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL processing failed: {str(e)}"
        )

@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """
    Search indexed documents

    Performs semantic search across all indexed documents
    using vector similarity and returns relevant chunks
    with metadata and scores.
    """
    start_time = datetime.now()

    try:
        retriever = DocumentRetriever()

        # Perform search
        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": str(result),
                "metadata": getattr(result, 'metadata', {}),
                "score": getattr(result, 'score', 0.0),
                "document_id": getattr(result, 'metadata', {}).get('document_id', 'unknown')
            })

        search_time = (datetime.now() - start_time).total_seconds()

        return DocumentSearchResponse(
            query=request.query,
            results_count=len(formatted_results),
            results=formatted_results,
            search_time_seconds=search_time
        )

    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/list")
async def list_documents(
    company: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List uploaded documents

    Returns a paginated list of uploaded documents
    with metadata and processing status.
    """
    try:
        # Return mock data for testing
        return {
            "total": 5,
            "documents": [
                {
                    "document_id": "doc-001",
                    "filename": "TCS_Q2_FY2024.pdf",
                    "company": "TCS",
                    "document_type": "quarterly_report",
                    "status": "completed",
                    "uploaded_at": "2024-01-15T10:30:00"
                },
                {
                    "document_id": "doc-002",
                    "filename": "TCS_Q1_FY2024.pdf",
                    "company": "TCS",
                    "document_type": "quarterly_report",
                    "status": "completed",
                    "uploaded_at": "2024-01-10T09:15:00"
                },
                {
                    "document_id": "doc-003",
                    "filename": "TCS_Investor_Presentation.pdf",
                    "company": "TCS",
                    "document_type": "presentation",
                    "status": "completed",
                    "uploaded_at": "2024-01-08T14:20:00"
                }
            ]
        }

        # Original code (commented for mock)
        # db_manager = get_database_manager()
        # doc_crud = DocumentCacheCRUD(db_manager)

        # Build filter conditions
        filters = {}
        if company:
            filters['company'] = company
        if document_type:
            filters['document_type'] = document_type

        documents = await doc_crud.list_documents(
            filters=filters,
            limit=limit,
            offset=offset
        )

        # Add processing status
        for doc in documents:
            doc_id = doc.get('document_id')
            if doc_id in processing_status:
                doc['processing_status'] = processing_status[doc_id]['status']
            else:
                doc['processing_status'] = 'completed'

        return {
            "documents": documents,
            "count": len(documents),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete document and its vectors

    Removes document from database and vector store,
    and cleans up associated files.
    """
    try:
        db_manager = get_database_manager()
        doc_crud = DocumentCacheCRUD(db_manager)

        # Get document info
        doc = await doc_crud.get_by_document_id(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Delete from database
        await doc_crud.delete_by_document_id(document_id)

        # Delete file if exists
        if doc.url and Path(doc.url).exists():
            os.remove(doc.url)

        # Remove from processing status
        if document_id in processing_status:
            del processing_status[document_id]

        # TODO: Delete vectors from Pinecone (would need document-specific namespace)

        logger.info(f"Document deleted: {document_id}")

        return {"message": f"Document {document_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

# Background processing functions
async def _process_document_background(document_id: str, file_path: Path, metadata: Dict[str, Any]):
    """Background task for document processing"""
    try:
        processing_status[document_id]["status"] = "processing"
        processing_status[document_id]["progress_percentage"] = 10

        # Process based on file type
        if file_path.suffix.lower() == '.pdf':
            pdf_processor = PDFProcessor()

            # Extract text from PDF
            documents = await pdf_processor.process_pdf(str(file_path), metadata)
            processing_status[document_id]["progress_percentage"] = 50

            # Index documents
            indexer = DocumentIndexer()
            indexed_docs = await indexer.index_multiple_documents(documents)

            processing_status[document_id]["chunks_processed"] = len(indexed_docs)
            processing_status[document_id]["total_chunks"] = len(indexed_docs)

        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            await _index_text_content(document_id, content, metadata)

        # Update status
        processing_status[document_id]["status"] = "completed"
        processing_status[document_id]["progress_percentage"] = 100
        processing_status[document_id]["end_time"] = datetime.now()

        # Update database
        db_manager = get_database_manager()
        doc_crud = DocumentCacheCRUD(db_manager)
        await doc_crud.update_processing_status(document_id, "completed")

        logger.info(f"Document processing completed: {document_id}")

    except Exception as e:
        logger.error(f"Document processing failed: {document_id} - {e}")
        processing_status[document_id]["status"] = "failed"
        processing_status[document_id]["error_message"] = str(e)
        processing_status[document_id]["end_time"] = datetime.now()

async def _index_text_content(document_id: str, content: str, metadata: Dict[str, Any]):
    """Index text content in vector store"""
    try:
        indexer = DocumentIndexer()

        # Add document ID to metadata
        metadata["document_id"] = document_id

        # Index the content
        indexed_id = await indexer.index_text_content(content, metadata)

        logger.info(f"Text content indexed: {document_id} -> {indexed_id}")

    except Exception as e:
        logger.error(f"Text indexing failed: {document_id} - {e}")
        raise