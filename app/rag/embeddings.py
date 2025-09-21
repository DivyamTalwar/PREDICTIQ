# File: app/rag/embeddings.py
# Purpose: OpenAI embeddings pipeline for TCS financial documents
# Dependencies: openai, tiktoken, numpy, logging
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 3

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI client not available")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available for token counting")

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """
    OpenAI embeddings manager for financial documents.

    Handles batch processing, cost optimization, caching, and rate limiting
    for generating embeddings from document chunks.
    """

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.model = "text-embedding-ada-002"
        self.dimension = 1536
        self.max_tokens = 8191  # Max tokens for ada-002
        self.batch_size = 100  # OpenAI recommended batch size

        # Cost tracking
        self.cost_per_1k_tokens = 0.0001  # $0.0001 per 1K tokens for ada-002
        self.tokens_used = 0
        self.total_cost = 0.0

        # Rate limiting
        self.requests_per_minute = 3000  # OpenAI default
        self.last_request_time = 0
        self.request_interval = 60 / self.requests_per_minute

        # Caching
        self.cache_dir = settings.vectorstore_directory / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer for counting
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI client not available - install openai")
            return

        if not settings.openai_api_key:
            logger.warning("OpenAI API key not configured")
            return

        try:
            self.client = OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized successfully")

            # Test the connection with a simple embedding
            self._test_connection()

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def _test_connection(self) -> None:
        """Test OpenAI connection with a simple embedding."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input="test connection"
            )
            logger.info("OpenAI connection test successful")
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")

    def _rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def _create_cache_key(self, text: str) -> str:
        """Create cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        cache_key = self._create_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    logger.debug(f"Using cached embedding for text: {text[:50]}...")
                    return cache_data["embedding"]
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")

        return None

    def _save_cached_embedding(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        cache_key = self._create_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                "text": text[:100],  # Store first 100 chars for reference
                "embedding": embedding,
                "model": self.model,
                "created_at": datetime.now().isoformat()
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)

            logger.debug(f"Cached embedding for text: {text[:50]}...")

        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits."""
        if not self.tokenizer:
            # Simple character-based truncation
            max_chars = self.max_tokens * 4
            return text[:max_chars]

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return text

        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:self.max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)

        logger.warning(f"Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text

    def generate_embedding(self, text: str, use_cache: bool = True) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector or None if failed
        """
        if not self.client:
            logger.error("OpenAI client not available")
            return None

        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Check cache first
        if use_cache:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding:
                return cached_embedding

        try:
            # Truncate text if necessary
            processed_text = self._truncate_text(text.strip())

            # Apply rate limiting
            self._rate_limit()

            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=processed_text
            )

            embedding = response.data[0].embedding

            # Update cost tracking
            tokens_used = response.usage.total_tokens
            self.tokens_used += tokens_used
            self.total_cost += (tokens_used / 1000) * self.cost_per_1k_tokens

            logger.debug(f"Generated embedding: {tokens_used} tokens, ${self.total_cost:.6f} total cost")

            # Cache the embedding
            if use_cache:
                self._save_cached_embedding(text, embedding)

            return embedding

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            time.sleep(60)  # Wait 1 minute
            return self.generate_embedding(text, use_cache)  # Retry

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            show_progress: Whether to log progress

        Returns:
            List of embeddings (same order as input texts)
        """
        if not self.client:
            logger.error("OpenAI client not available")
            return [None] * len(texts)

        embeddings = []
        cache_hits = 0
        api_calls = 0

        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []

            # Check cache for each text in batch
            texts_to_embed = []
            cached_results = {}

            for j, text in enumerate(batch):
                if use_cache:
                    cached_embedding = self._get_cached_embedding(text)
                    if cached_embedding:
                        cached_results[j] = cached_embedding
                        cache_hits += 1
                    else:
                        texts_to_embed.append((j, text))
                else:
                    texts_to_embed.append((j, text))

            # Generate embeddings for non-cached texts
            if texts_to_embed:
                try:
                    # Prepare texts for API call
                    api_texts = [self._truncate_text(text.strip()) for _, text in texts_to_embed]

                    # Apply rate limiting
                    self._rate_limit()

                    # Make API call
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=api_texts
                    )

                    # Update cost tracking
                    tokens_used = response.usage.total_tokens
                    self.tokens_used += tokens_used
                    self.total_cost += (tokens_used / 1000) * self.cost_per_1k_tokens

                    api_calls += 1

                    # Store results
                    for idx, (original_idx, original_text) in enumerate(texts_to_embed):
                        embedding = response.data[idx].embedding
                        cached_results[original_idx] = embedding

                        # Cache the embedding
                        if use_cache:
                            self._save_cached_embedding(original_text, embedding)

                except Exception as e:
                    logger.error(f"Error in batch embedding: {e}")
                    # Fill with None for failed embeddings
                    for original_idx, _ in texts_to_embed:
                        cached_results[original_idx] = None

            # Reconstruct batch results in original order
            for j in range(len(batch)):
                batch_embeddings.append(cached_results.get(j, None))

            embeddings.extend(batch_embeddings)

            if show_progress:
                progress = ((i + len(batch)) / len(texts)) * 100
                logger.info(f"Embedding progress: {progress:.1f}% ({i + len(batch)}/{len(texts)})")

        if show_progress:
            logger.info(f"Batch embedding complete: {cache_hits} cache hits, {api_calls} API calls")
            logger.info(f"Total cost: ${self.total_cost:.6f}, Total tokens: {self.tokens_used}")

        return embeddings

    def embed_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "text",
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks with metadata.

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            List of chunks with embeddings added
        """
        logger.info(f"Embedding {len(chunks)} document chunks")

        # Extract texts
        texts = [chunk.get(text_field, "") for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, use_cache=use_cache)

        # Add embeddings to chunks
        enhanced_chunks = []
        successful_embeddings = 0

        for chunk, embedding in zip(chunks, embeddings):
            enhanced_chunk = chunk.copy()

            if embedding:
                enhanced_chunk["embedding"] = embedding
                enhanced_chunk["embedding_model"] = self.model
                enhanced_chunk["embedding_dimension"] = len(embedding)
                enhanced_chunk["embedding_created_at"] = datetime.now().isoformat()
                successful_embeddings += 1
            else:
                enhanced_chunk["embedding_error"] = "Failed to generate embedding"

            enhanced_chunks.append(enhanced_chunk)

        logger.info(f"Successfully embedded {successful_embeddings}/{len(chunks)} chunks")

        return enhanced_chunks

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def find_similar_chunks(
        self,
        query_embedding: List[float],
        chunk_embeddings: List[Tuple[str, List[float], Dict[str, Any]]],
        top_k: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to query embedding.

        Args:
            query_embedding: Query vector
            chunk_embeddings: List of (id, embedding, metadata) tuples
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar chunks with similarity scores
        """
        similarities = []

        for chunk_id, embedding, metadata in chunk_embeddings:
            similarity = self.cosine_similarity(query_embedding, embedding)

            if similarity >= min_similarity:
                similarities.append({
                    "id": chunk_id,
                    "similarity": similarity,
                    "metadata": metadata
                })

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def get_cost_stats(self) -> Dict[str, Any]:
        """Get cost and usage statistics."""
        return {
            "total_tokens_used": self.tokens_used,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_per_token": self.cost_per_1k_tokens / 1000,
            "model": self.model,
            "dimension": self.dimension
        }

    def clear_cache(self, max_age_days: int = 30) -> int:
        """Clear old cache files."""
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_date:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.error(f"Error removing cache file {cache_file}: {e}")

        logger.info(f"Cleared {removed_count} old embedding cache files")
        return removed_count

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on embeddings system."""
        health_status = {
            "embeddings": {
                "status": "unknown",
                "client_available": OPENAI_AVAILABLE,
                "api_key_configured": bool(settings.openai_api_key),
                "tokenizer_available": TIKTOKEN_AVAILABLE,
                "model": self.model,
                "dimension": self.dimension,
                "cost_stats": self.get_cost_stats(),
                "error": None
            }
        }

        try:
            if not OPENAI_AVAILABLE:
                health_status["embeddings"]["status"] = "unavailable"
                health_status["embeddings"]["error"] = "OpenAI client not installed"
            elif not settings.openai_api_key:
                health_status["embeddings"]["status"] = "not_configured"
                health_status["embeddings"]["error"] = "API key not configured"
            elif not self.client:
                health_status["embeddings"]["status"] = "initialization_failed"
                health_status["embeddings"]["error"] = "Client initialization failed"
            else:
                # Try a simple embedding
                test_embedding = self.generate_embedding("health check test", use_cache=False)
                if test_embedding:
                    health_status["embeddings"]["status"] = "healthy"
                else:
                    health_status["embeddings"]["status"] = "api_error"
                    health_status["embeddings"]["error"] = "Failed to generate test embedding"

        except Exception as e:
            health_status["embeddings"]["status"] = "error"
            health_status["embeddings"]["error"] = str(e)

        return health_status