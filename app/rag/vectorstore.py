# File: app/rag/vectorstore.py
# Purpose: Pinecone vector store setup and management for TCS financial documents
# Dependencies: pinecone-client, logging, typing
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 3

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone client not available")

from app.config import settings

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    """
    Pinecone vector store manager for TCS financial documents.

    Handles index creation, vector operations, and namespace management
    for different document types and time periods.
    """

    def __init__(self):
        self.pc: Optional[Pinecone] = None
        self.index = None
        self.index_name = settings.pinecone_index_name
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension

        # Namespace strategy for organizing vectors
        self.namespaces = {
            "quarterly_reports": "quarterly_reports",
            "earnings_calls": "earnings_calls",
            "market_data": "market_data",
            "general": "general"
        }

        self._initialize_pinecone()

    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone client and connection."""
        if not PINECONE_AVAILABLE:
            logger.error("Pinecone client not available - install pinecone-client")
            return

        if not settings.pinecone_api_key:
            logger.warning("Pinecone API key not configured")
            return

        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=settings.pinecone_api_key)

            logger.info("Pinecone client initialized successfully")

            # Try to connect to existing index or create new one
            self._setup_index()

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.pc = None

    def _setup_index(self) -> None:
        """Setup or create Pinecone index."""
        if not self.pc:
            return

        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if self.index_name in index_names:
                logger.info(f"Connecting to existing index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
            else:
                logger.info(f"Creating new index: {self.index_name}")
                self._create_index()

        except Exception as e:
            logger.error(f"Error setting up index: {e}")

    def _create_index(self) -> None:
        """Create a new Pinecone index with optimal configuration."""
        if not self.pc:
            return

        try:
            # Create index with serverless specification
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",  # Best for text embeddings
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Default region
                )
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                logger.info("Waiting for index to be ready...")
                time.sleep(1)

            self.index = self.pc.Index(self.index_name)
            logger.info(f"Index {self.index_name} created and ready")

        except Exception as e:
            logger.error(f"Failed to create index: {e}")

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if not self.index:
            return {"error": "Index not available"}

        try:
            stats = self.index.describe_index_stats()

            # Convert to serializable format
            stats_dict = {
                "total_vector_count": stats.get("total_vector_count", 0),
                "namespaces": {}
            }

            if "namespaces" in stats:
                for ns, data in stats["namespaces"].items():
                    stats_dict["namespaces"][ns] = {
                        "vector_count": data.get("vector_count", 0)
                    }

            return stats_dict

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        namespace: str = "general",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone index.

        Args:
            vectors: List of (id, vector, metadata) tuples
            namespace: Namespace to store vectors in
            batch_size: Number of vectors to upsert per batch

        Returns:
            Dictionary with upsert results
        """
        if not self.index:
            return {"error": "Index not available", "upserted_count": 0}

        try:
            total_upserted = 0

            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]

                # Format for Pinecone
                pinecone_vectors = []
                for vector_id, vector, metadata in batch:
                    pinecone_vectors.append({
                        "id": vector_id,
                        "values": vector,
                        "metadata": metadata
                    })

                # Upsert batch
                response = self.index.upsert(
                    vectors=pinecone_vectors,
                    namespace=namespace
                )

                total_upserted += response.get("upserted_count", 0)

                logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")

            logger.info(f"Successfully upserted {total_upserted} vectors to namespace '{namespace}'")

            return {
                "success": True,
                "upserted_count": total_upserted,
                "namespace": namespace
            }

        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            return {"error": str(e), "upserted_count": 0}

    def query_vectors(
        self,
        query_vector: List[float],
        namespace: str = "general",
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> Dict[str, Any]:
        """
        Query vectors from Pinecone index.

        Args:
            query_vector: Query embedding vector
            namespace: Namespace to search in
            top_k: Number of similar vectors to return
            filter_dict: Metadata filters
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values

        Returns:
            Query results with matches
        """
        if not self.index:
            return {"error": "Index not available", "matches": []}

        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_values
            )

            # Process results
            matches = []
            for match in response.get("matches", []):
                match_data = {
                    "id": match.id,
                    "score": match.score
                }

                if include_metadata and hasattr(match, 'metadata'):
                    match_data["metadata"] = match.metadata

                if include_values and hasattr(match, 'values'):
                    match_data["values"] = match.values

                matches.append(match_data)

            logger.debug(f"Query returned {len(matches)} matches from namespace '{namespace}'")

            return {
                "success": True,
                "matches": matches,
                "namespace": namespace
            }

        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            return {"error": str(e), "matches": []}

    def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: str = "general"
    ) -> Dict[str, Any]:
        """Delete vectors from index."""
        if not self.index:
            return {"error": "Index not available"}

        try:
            self.index.delete(ids=vector_ids, namespace=namespace)

            logger.info(f"Deleted {len(vector_ids)} vectors from namespace '{namespace}'")

            return {
                "success": True,
                "deleted_count": len(vector_ids),
                "namespace": namespace
            }

        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return {"error": str(e)}

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """Delete all vectors in a namespace."""
        if not self.index:
            return {"error": "Index not available"}

        try:
            self.index.delete(delete_all=True, namespace=namespace)

            logger.info(f"Deleted all vectors from namespace '{namespace}'")

            return {
                "success": True,
                "namespace": namespace
            }

        except Exception as e:
            logger.error(f"Error deleting namespace: {e}")
            return {"error": str(e)}

    def get_vector(
        self,
        vector_id: str,
        namespace: str = "general"
    ) -> Dict[str, Any]:
        """Fetch a specific vector by ID."""
        if not self.index:
            return {"error": "Index not available"}

        try:
            response = self.index.fetch(ids=[vector_id], namespace=namespace)

            vectors = response.get("vectors", {})
            if vector_id in vectors:
                vector_data = vectors[vector_id]
                return {
                    "success": True,
                    "id": vector_id,
                    "metadata": vector_data.get("metadata", {}),
                    "values": vector_data.get("values", [])
                }
            else:
                return {"error": "Vector not found"}

        except Exception as e:
            logger.error(f"Error fetching vector: {e}")
            return {"error": str(e)}

    def create_document_id(
        self,
        document_name: str,
        chunk_index: int,
        document_type: str = "quarterly_report"
    ) -> str:
        """Create a standardized document ID for vector storage."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{document_type}_{document_name}_{chunk_index}_{timestamp}"

    def create_chunk_metadata(
        self,
        document_name: str,
        chunk_index: int,
        chunk_text: str,
        document_type: str = "quarterly_report",
        quarter: Optional[int] = None,
        fiscal_year: Optional[int] = None,
        **additional_metadata
    ) -> Dict[str, Any]:
        """Create standardized metadata for document chunks."""
        metadata = {
            "document_name": document_name,
            "document_type": document_type,
            "chunk_index": chunk_index,
            "chunk_size": len(chunk_text),
            "created_at": datetime.now().isoformat(),
            "company": "TCS"
        }

        if quarter:
            metadata["quarter"] = quarter
        if fiscal_year:
            metadata["fiscal_year"] = fiscal_year

        # Add any additional metadata
        metadata.update(additional_metadata)

        return metadata

    def search_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        namespace: str = "general",
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """Search vectors by metadata filters only."""
        if not self.index:
            return []

        try:
            # Create a dummy query vector (all zeros) since we're filtering by metadata
            dummy_vector = [0.0] * self.dimension

            response = self.query_vectors(
                query_vector=dummy_vector,
                namespace=namespace,
                top_k=top_k,
                filter_dict=filter_dict,
                include_metadata=True
            )

            return response.get("matches", [])

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    def get_namespace_for_document_type(self, document_type: str) -> str:
        """Get appropriate namespace for document type."""
        if document_type in ["quarterly_report", "annual_report"]:
            return self.namespaces["quarterly_reports"]
        elif document_type in ["earnings_transcript", "earnings_call"]:
            return self.namespaces["earnings_calls"]
        elif document_type in ["market_data", "news", "analysis"]:
            return self.namespaces["market_data"]
        else:
            return self.namespaces["general"]

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Pinecone connection."""
        health_status = {
            "pinecone": {
                "status": "unknown",
                "client_available": PINECONE_AVAILABLE,
                "api_key_configured": bool(settings.pinecone_api_key),
                "index_name": self.index_name,
                "index_available": bool(self.index),
                "stats": {},
                "error": None
            }
        }

        try:
            if not PINECONE_AVAILABLE:
                health_status["pinecone"]["status"] = "unavailable"
                health_status["pinecone"]["error"] = "Pinecone client not installed"
            elif not settings.pinecone_api_key:
                health_status["pinecone"]["status"] = "not_configured"
                health_status["pinecone"]["error"] = "API key not configured"
            elif not self.index:
                health_status["pinecone"]["status"] = "no_index"
                health_status["pinecone"]["error"] = "Index not available"
            else:
                # Try to get index stats
                stats = self.get_index_stats()
                if "error" in stats:
                    health_status["pinecone"]["status"] = "error"
                    health_status["pinecone"]["error"] = stats["error"]
                else:
                    health_status["pinecone"]["status"] = "healthy"
                    health_status["pinecone"]["stats"] = stats

        except Exception as e:
            health_status["pinecone"]["status"] = "error"
            health_status["pinecone"]["error"] = str(e)

        return health_status

    def cleanup(self) -> None:
        """Cleanup Pinecone resources."""
        # Pinecone client doesn't require explicit cleanup
        logger.info("Pinecone vectorstore cleanup completed")