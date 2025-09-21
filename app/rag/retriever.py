# File: app/rag/retriever.py
# Purpose: Document retrieval system with query enhancement and reranking
# Dependencies: openai, re, typing, logging
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 3

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict

from .vectorstore import PineconeVectorStore
from .embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Advanced document retrieval system for TCS financial documents.

    Implements query enhancement, multi-query generation, reranking,
    and Maximum Marginal Relevance (MMR) for diverse results.
    """

    def __init__(self):
        self.vectorstore = PineconeVectorStore()
        self.embeddings_manager = EmbeddingsManager()

        # Query enhancement patterns
        self.financial_synonyms = {
            "revenue": ["sales", "income", "turnover", "receipts"],
            "profit": ["earnings", "net income", "surplus", "gains"],
            "growth": ["increase", "expansion", "rise", "improvement"],
            "margin": ["profitability", "ratio", "percentage"],
            "quarter": ["Q1", "Q2", "Q3", "Q4", "quarterly"],
            "year": ["annual", "yearly", "FY", "fiscal year"],
            "tcs": ["Tata Consultancy Services", "company"]
        }

        # Financial metrics patterns
        self.metrics_patterns = {
            "revenue_queries": ["revenue", "sales", "income", "turnover"],
            "profitability_queries": ["profit", "margin", "EBITDA", "earnings"],
            "growth_queries": ["growth", "increase", "YoY", "year over year"],
            "segment_queries": ["segment", "vertical", "business unit"],
            "geography_queries": ["region", "geographic", "North America", "Europe", "India"]
        }

    def enhance_query(self, query: str) -> List[str]:
        """
        Enhance query with synonyms and financial terminology.

        Args:
            query: Original user query

        Returns:
            List of enhanced query variations
        """
        enhanced_queries = [query]  # Start with original
        query_lower = query.lower()

        # Add synonym variations
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    enhanced_query = re.sub(
                        r'\b' + re.escape(term) + r'\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if enhanced_query != query:
                        enhanced_queries.append(enhanced_query)

        # Add financial context variations
        context_enhanced = []

        # Add TCS context if not present
        if "tcs" not in query_lower and "company" not in query_lower:
            context_enhanced.append(f"TCS {query}")
            context_enhanced.append(f"{query} for TCS")

        # Add time context if not present
        time_terms = ["quarter", "year", "Q1", "Q2", "Q3", "Q4", "FY", "annual"]
        if not any(term in query_lower for term in time_terms):
            context_enhanced.append(f"{query} quarterly")
            context_enhanced.append(f"{query} annual")

        enhanced_queries.extend(context_enhanced)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in enhanced_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        logger.debug(f"Enhanced query '{query}' into {len(unique_queries)} variations")
        return unique_queries[:5]  # Limit to top 5 variations

    def generate_multi_queries(self, query: str) -> List[str]:
        """
        Generate multiple query variations for comprehensive search.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        queries = []

        # Original query
        queries.append(query)

        # Question variations
        if not query.endswith('?'):
            queries.append(f"What is {query}?")
            queries.append(f"How much {query}?")

        # Specific metric queries
        query_lower = query.lower()

        if any(term in query_lower for term in self.metrics_patterns["revenue_queries"]):
            queries.extend([
                "TCS revenue growth",
                "quarterly revenue performance",
                "revenue by segment"
            ])

        if any(term in query_lower for term in self.metrics_patterns["profitability_queries"]):
            queries.extend([
                "profit margin analysis",
                "EBITDA performance",
                "net income trends"
            ])

        if any(term in query_lower for term in self.metrics_patterns["growth_queries"]):
            queries.extend([
                "year over year growth",
                "quarterly growth trends",
                "business growth drivers"
            ])

        # Remove duplicates
        unique_queries = list(dict.fromkeys(queries))

        logger.debug(f"Generated {len(unique_queries)} query variations")
        return unique_queries[:7]  # Limit to 7 queries

    def identify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Identify the intent and key entities in the query.

        Args:
            query: User query

        Returns:
            Dictionary with intent analysis
        """
        query_lower = query.lower()

        intent = {
            "primary_intent": "general",
            "metrics_requested": [],
            "time_period": None,
            "segments": [],
            "geography": [],
            "company": "TCS",
            "query_type": "factual"
        }

        # Identify metrics
        for metric_type, keywords in self.metrics_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                intent["metrics_requested"].append(metric_type.replace("_queries", ""))

        # Identify time periods
        time_patterns = {
            "Q1": r'\bq1\b|\bfirst quarter\b',
            "Q2": r'\bq2\b|\bsecond quarter\b',
            "Q3": r'\bq3\b|\bthird quarter\b',
            "Q4": r'\bq4\b|\bfourth quarter\b',
            "annual": r'\bannual\b|\byearly\b|\bfy\b',
            "latest": r'\blatest\b|\brecent\b|\bcurrent\b'
        }

        for period, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                intent["time_period"] = period
                break

        # Identify segments/verticals
        segment_keywords = [
            "BFSI", "banking", "financial services",
            "retail", "manufacturing", "healthcare",
            "telecom", "utilities", "segment", "vertical"
        ]

        for keyword in segment_keywords:
            if keyword in query_lower:
                intent["segments"].append(keyword)

        # Identify geography
        geo_keywords = [
            "North America", "Europe", "India", "Asia Pacific",
            "UK", "US", "America", "region", "geographic"
        ]

        for keyword in geo_keywords:
            if keyword in query_lower:
                intent["geography"].append(keyword)

        # Determine query type
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            intent["query_type"] = "comparison"
        elif any(word in query_lower for word in ["trend", "change", "growth", "decline"]):
            intent["query_type"] = "trend_analysis"
        elif any(word in query_lower for word in ["why", "how", "reason", "cause"]):
            intent["query_type"] = "analytical"
        elif any(word in query_lower for word in ["forecast", "predict", "future", "outlook"]):
            intent["query_type"] = "predictive"

        # Set primary intent
        if intent["metrics_requested"]:
            intent["primary_intent"] = intent["metrics_requested"][0]
        elif intent["query_type"] != "factual":
            intent["primary_intent"] = intent["query_type"]

        return intent

    def build_metadata_filter(self, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build metadata filter based on query intent.

        Args:
            intent: Query intent analysis

        Returns:
            Metadata filter dictionary
        """
        filters = {}

        # Company filter
        if intent["company"]:
            filters["company"] = intent["company"]

        # Time period filter
        if intent["time_period"]:
            if intent["time_period"] in ["Q1", "Q2", "Q3", "Q4"]:
                filters["quarter"] = int(intent["time_period"][1])
            elif intent["time_period"] == "latest":
                # This would be handled in the retrieval logic
                pass

        # Document type filter based on metrics
        if intent["metrics_requested"]:
            if any(metric in ["revenue", "profitability"] for metric in intent["metrics_requested"]):
                filters["document_type"] = "quarterly_report"

        return filters if filters else None

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 10,
        namespace: str = "general",
        use_query_enhancement: bool = True,
        rerank: bool = True,
        mmr_diversity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            namespace: Pinecone namespace to search
            use_query_enhancement: Whether to enhance query
            rerank: Whether to apply reranking
            mmr_diversity: Diversity factor for MMR (0-1)

        Returns:
            List of relevant documents with scores
        """
        if not self.vectorstore.index:
            logger.error("Vector store not available")
            return []

        # Analyze query intent
        intent = self.identify_query_intent(query)
        logger.info(f"Query intent: {intent['primary_intent']}, type: {intent['query_type']}")

        # Generate query variations
        if use_query_enhancement:
            enhanced_queries = self.enhance_query(query)
            multi_queries = self.generate_multi_queries(query)
            all_queries = list(set(enhanced_queries + multi_queries))
        else:
            all_queries = [query]

        # Generate embeddings for all queries
        query_embeddings = self.embeddings_manager.generate_embeddings_batch(
            all_queries,
            use_cache=True,
            show_progress=False
        )

        # Filter out failed embeddings
        valid_query_embeddings = [emb for emb in query_embeddings if emb is not None]

        if not valid_query_embeddings:
            logger.error("Failed to generate embeddings for queries")
            return []

        # Build metadata filter
        metadata_filter = self.build_metadata_filter(intent)

        # Retrieve from vector store for each query
        all_results = []

        for i, embedding in enumerate(valid_query_embeddings):
            try:
                results = self.vectorstore.query_vectors(
                    query_vector=embedding,
                    namespace=namespace,
                    top_k=min(top_k * 2, 50),  # Get more results for diversity
                    filter_dict=metadata_filter,
                    include_metadata=True
                )

                if results.get("success") and results.get("matches"):
                    for match in results["matches"]:
                        match["query_index"] = i
                        match["original_query"] = all_queries[i] if i < len(all_queries) else query
                        all_results.append(match)

            except Exception as e:
                logger.error(f"Error retrieving for query {i}: {e}")

        if not all_results:
            logger.warning("No results found for any query variation")
            return []

        # Remove duplicates based on document ID
        unique_results = {}
        for result in all_results:
            doc_id = result["id"]
            if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
                unique_results[doc_id] = result

        unique_results_list = list(unique_results.values())

        # Apply reranking if requested
        if rerank:
            reranked_results = self._rerank_results(query, unique_results_list, intent)
        else:
            reranked_results = sorted(unique_results_list, key=lambda x: x["score"], reverse=True)

        # Apply MMR for diversity
        if mmr_diversity > 0:
            final_results = self._apply_mmr(
                query_embeddings[0],  # Use original query embedding
                reranked_results,
                top_k,
                mmr_diversity
            )
        else:
            final_results = reranked_results[:top_k]

        logger.info(f"Retrieved {len(final_results)} documents for query: '{query}'")
        return final_results

    def _rerank_results(
        self,
        original_query: str,
        results: List[Dict[str, Any]],
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on relevance and intent.

        Args:
            original_query: Original user query
            results: List of retrieved results
            intent: Query intent analysis

        Returns:
            Reranked results
        """
        query_lower = original_query.lower()

        for result in results:
            base_score = result["score"]
            boost_factors = []

            metadata = result.get("metadata", {})

            # Intent-based boosting
            if intent["primary_intent"] == "revenue" and any(
                term in metadata.get("text", "").lower()
                for term in ["revenue", "sales", "income"]
            ):
                boost_factors.append(1.2)

            if intent["primary_intent"] == "profitability" and any(
                term in metadata.get("text", "").lower()
                for term in ["profit", "margin", "EBITDA"]
            ):
                boost_factors.append(1.2)

            # Time relevance boosting
            if intent["time_period"]:
                if intent["time_period"] == "latest":
                    # Boost more recent documents
                    if metadata.get("fiscal_year", 0) >= 2023:
                        boost_factors.append(1.15)
                elif intent["time_period"] in ["Q1", "Q2", "Q3", "Q4"]:
                    quarter_num = int(intent["time_period"][1])
                    if metadata.get("quarter") == quarter_num:
                        boost_factors.append(1.3)

            # Document type relevance
            doc_type = metadata.get("document_type", "")
            if intent["query_type"] == "analytical" and "earnings" in doc_type:
                boost_factors.append(1.1)
            elif intent["metrics_requested"] and "quarterly_report" in doc_type:
                boost_factors.append(1.1)

            # Apply boosts
            final_score = base_score
            for boost in boost_factors:
                final_score *= boost

            result["rerank_score"] = final_score
            result["boost_factors"] = boost_factors

        # Sort by reranked score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

        logger.debug(f"Reranked {len(results)} results")
        return reranked

    def _apply_mmr(
        self,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
        top_k: int,
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximum Marginal Relevance for diverse results.

        Args:
            query_embedding: Query embedding vector
            results: List of results with embeddings
            top_k: Number of results to return
            diversity_factor: Diversity weight (0-1)

        Returns:
            Diversified results using MMR
        """
        if len(results) <= top_k:
            return results

        selected = []
        remaining = results.copy()

        # Always select the top result first
        if remaining:
            best = max(remaining, key=lambda x: x.get("rerank_score", x["score"]))
            selected.append(best)
            remaining.remove(best)

        # Select remaining results using MMR
        while len(selected) < top_k and remaining:
            mmr_scores = []

            for candidate in remaining:
                # Relevance score
                relevance = candidate.get("rerank_score", candidate["score"])

                # Calculate maximum similarity to already selected documents
                max_similarity = 0
                candidate_embedding = candidate.get("embedding")

                if candidate_embedding:
                    for selected_doc in selected:
                        selected_embedding = selected_doc.get("embedding")
                        if selected_embedding:
                            similarity = self.embeddings_manager.cosine_similarity(
                                candidate_embedding, selected_embedding
                            )
                            max_similarity = max(max_similarity, similarity)

                # MMR score: relevance - diversity_factor * max_similarity
                mmr_score = relevance - (diversity_factor * max_similarity)
                mmr_scores.append((candidate, mmr_score))

            if mmr_scores:
                # Select candidate with highest MMR score
                best_candidate, best_score = max(mmr_scores, key=lambda x: x[1])
                best_candidate["mmr_score"] = best_score
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        logger.debug(f"Applied MMR to select {len(selected)} diverse results")
        return selected

    def search_by_filters(
        self,
        filters: Dict[str, Any],
        namespace: str = "general",
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters only.

        Args:
            filters: Metadata filters
            namespace: Pinecone namespace
            top_k: Maximum results to return

        Returns:
            List of matching documents
        """
        if not self.vectorstore.index:
            logger.error("Vector store not available")
            return []

        try:
            results = self.vectorstore.search_by_metadata(
                filter_dict=filters,
                namespace=namespace,
                top_k=top_k
            )

            logger.info(f"Found {len(results)} documents matching filters: {filters}")
            return results

        except Exception as e:
            logger.error(f"Error searching by filters: {e}")
            return []

    def get_document_context(
        self,
        document_id: str,
        namespace: str = "general",
        context_window: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get surrounding context for a document chunk.

        Args:
            document_id: Document ID
            namespace: Pinecone namespace
            context_window: Number of chunks before/after to include

        Returns:
            List of context chunks
        """
        try:
            # Extract document info from ID
            parts = document_id.split("_")
            if len(parts) >= 3:
                doc_name = "_".join(parts[1:-2])  # Reconstruct document name
                chunk_index = int(parts[-2])

                # Get surrounding chunks
                context_chunks = []
                for i in range(max(0, chunk_index - context_window),
                             chunk_index + context_window + 1):
                    context_id = f"{parts[0]}_{doc_name}_{i}_{parts[-1]}"

                    chunk_data = self.vectorstore.get_vector(context_id, namespace)
                    if chunk_data.get("success"):
                        context_chunks.append(chunk_data)

                return context_chunks

        except Exception as e:
            logger.error(f"Error getting document context: {e}")

        return []

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on retrieval system."""
        vectorstore_health = self.vectorstore.health_check()
        embeddings_health = self.embeddings_manager.health_check()

        retriever_health = {
            "retriever": {
                "status": "healthy" if (
                    vectorstore_health["pinecone"]["status"] == "healthy" and
                    embeddings_health["embeddings"]["status"] == "healthy"
                ) else "degraded",
                "vectorstore": vectorstore_health["pinecone"]["status"],
                "embeddings": embeddings_health["embeddings"]["status"],
                "query_enhancement": True,
                "mmr_enabled": True
            }
        }

        return {**vectorstore_health, **embeddings_health, **retriever_health}