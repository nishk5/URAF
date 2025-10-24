"""
Advanced RAG 2.0 System with Reranking

Based on:
- Self-RAG (Asai et al., 2024)
- CRAG: Corrective RAG (Yan et al., 2024)
- Advanced retrieval techniques (HyDE, Multi-hop, Reranking)

Provides state-of-the-art retrieval-augmented generation with:
- Query transformation (HyDE, decomposition, expansion)
- Hybrid search (dense + sparse)
- Multi-hop retrieval
- Cross-encoder reranking
- Self-reflective retrieval (when to retrieve, is relevant, is supported)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import chromadb
from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from uraf.llm_client import LLMClient


class RetrievalMode(Enum):
    """Retrieval modes."""

    DENSE = "dense"  # Semantic similarity with embeddings
    SPARSE = "sparse"  # Keyword-based (TF-IDF, BM25)
    HYBRID = "hybrid"  # Combination of dense and sparse


class QueryTransformStrategy(Enum):
    """Query transformation strategies."""

    NONE = "none"  # No transformation
    HYDE = "hyde"  # Hypothetical Document Embeddings
    DECOMPOSITION = "decomposition"  # Break into sub-queries
    EXPANSION = "expansion"  # Add related terms


@dataclass
class RetrievedChunk:
    """A retrieved text chunk with metadata."""

    content: str
    score: float
    metadata: dict[str, Any]
    source: str  # dense, sparse, or hybrid


@dataclass
class RetrievalResult:
    """Result of retrieval with metadata."""

    chunks: list[RetrievedChunk]
    query: str
    transformed_query: str | None
    num_chunks: int
    retrieval_method: str


class AdvancedRAG:
    """Advanced RAG 2.0 system."""

    def __init__(
        self,
        llm_client: LLMClient,
        collection_name: str = "rag_knowledge",
        persist_directory: str = "data/rag_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ):
        """
        Initialize RAG system.

        Args:
            llm_client: LLM client for generation
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistence
            embedding_model: Model for dense retrieval
            reranker_model: Cross-encoder for reranking
            retrieval_mode: Retrieval mode (dense/sparse/hybrid)
        """
        self.llm = llm_client
        self.retrieval_mode = retrieval_mode

        # Initialize embeddings
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize reranker
        self.reranker = CrossEncoder(reranker_model)

        # Initialize ChromaDB for dense retrieval
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # Initialize TF-IDF for sparse retrieval
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.tfidf_matrix = None
        self.sparse_docs: list[str] = []

        logger.info(f"Initialized AdvancedRAG with {retrieval_mode.value} retrieval")

    def add_documents(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Add documents to knowledge base.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        logger.info(f"Adding {len(documents)} documents to RAG system")

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        # Add to ChromaDB (dense)
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            embeddings=embeddings, documents=documents, ids=ids, metadatas=metadatas or [{}] * len(documents)
        )

        # Update TF-IDF (sparse)
        self.sparse_docs.extend(documents)
        if len(self.sparse_docs) > 1:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sparse_docs)

        logger.info(f"Knowledge base now has {len(self.sparse_docs)} documents")

    async def query(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        query_transform: QueryTransformStrategy = QueryTransformStrategy.NONE,
        multi_hop: bool = False,
    ) -> dict[str, Any]:
        """
        RAG query with retrieval and generation.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            rerank: Whether to rerank results
            query_transform: Query transformation strategy
            multi_hop: Whether to use multi-hop retrieval

        Returns:
            Generated answer with sources
        """
        logger.info(f"RAG query: {query[:100]}...")

        # Query transformation
        transformed_query = await self._transform_query(query, query_transform)

        # Retrieval
        if multi_hop:
            retrieval_result = await self._multi_hop_retrieval(query, transformed_query, top_k)
        else:
            retrieval_result = await self._single_retrieval(transformed_query or query, top_k)

        # Reranking
        if rerank and len(retrieval_result.chunks) > 1:
            retrieval_result = self._rerank_chunks(query, retrieval_result)

        # Context filtering
        filtered_chunks = self._filter_relevant_chunks(retrieval_result.chunks, relevance_threshold=0.3)

        # Generate answer
        answer = await self._generate_with_context(query, filtered_chunks)

        return {
            "answer": answer,
            "sources": [
                {"content": c.content[:200], "score": c.score, "metadata": c.metadata} for c in filtered_chunks
            ],
            "num_sources_used": len(filtered_chunks),
            "retrieval_method": retrieval_result.retrieval_method,
            "query_transformed": transformed_query is not None,
        }

    async def _transform_query(self, query: str, strategy: QueryTransformStrategy) -> str | None:
        """Transform query using specified strategy."""
        if strategy == QueryTransformStrategy.NONE:
            return None

        elif strategy == QueryTransformStrategy.HYDE:
            # HyDE: Generate hypothetical document that would answer query
            hyde_prompt = f"""Write a detailed passage that would perfectly answer this question:

Question: {query}

Passage:"""

            hypothetical_doc = await self.llm.query(hyde_prompt)
            logger.debug(f"HyDE generated: {hypothetical_doc[:100]}...")
            return hypothetical_doc.strip()

        elif strategy == QueryTransformStrategy.DECOMPOSITION:
            # Break complex query into simpler sub-queries
            decomp_prompt = f"""Break this complex question into 2-3 simpler sub-questions:

Question: {query}

Sub-questions:
1."""

            response = await self.llm.query(decomp_prompt)
            # Parse sub-questions (simplified)
            return response.strip()

        elif strategy == QueryTransformStrategy.EXPANSION:
            # Add related terms
            expand_prompt = f"""Expand this query with related terms and synonyms:

Query: {query}

Expanded query:"""

            expanded = await self.llm.query(expand_prompt)
            return expanded.strip()

        return None

    async def _single_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Single-step retrieval."""
        if self.retrieval_mode == RetrievalMode.DENSE:
            chunks = self._dense_retrieval(query, top_k)
            method = "dense"

        elif self.retrieval_mode == RetrievalMode.SPARSE:
            chunks = self._sparse_retrieval(query, top_k)
            method = "sparse"

        elif self.retrieval_mode == RetrievalMode.HYBRID:
            dense_chunks = self._dense_retrieval(query, top_k)
            sparse_chunks = self._sparse_retrieval(query, top_k)
            chunks = self._merge_retrieval_results(dense_chunks, sparse_chunks, top_k)
            method = "hybrid"

        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

        return RetrievalResult(
            chunks=chunks, query=query, transformed_query=None, num_chunks=len(chunks), retrieval_method=method
        )

    def _dense_retrieval(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Dense retrieval using embeddings."""
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(top_k, self.collection.count())
        )

        chunks = []
        if results["documents"]:
            for doc, dist, metadata in zip(
                results["documents"][0], results["distances"][0], results["metadatas"][0], strict=True
            ):
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1.0 / (1.0 + dist)

                chunks.append(RetrievedChunk(content=doc, score=score, metadata=metadata, source="dense"))

        return chunks

    def _sparse_retrieval(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Sparse retrieval using TF-IDF."""
        if self.tfidf_matrix is None or len(self.sparse_docs) == 0:
            return []

        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])

        # Compute similarities
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        # Get top-k
        top_indices = similarities.argsort()[-top_k:][::-1]

        chunks = []
        for idx in top_indices:
            if similarities[idx] > 0:
                chunks.append(
                    RetrievedChunk(
                        content=self.sparse_docs[idx], score=float(similarities[idx]), metadata={}, source="sparse"
                    )
                )

        return chunks

    def _merge_retrieval_results(
        self, dense_chunks: list[RetrievedChunk], sparse_chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Merge dense and sparse retrieval using reciprocal rank fusion."""
        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant

        # Score each document
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, RetrievedChunk] = {}

        # Add dense scores
        for rank, chunk in enumerate(dense_chunks):
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[chunk.content] = doc_scores.get(chunk.content, 0.0) + rrf_score
            doc_objects[chunk.content] = chunk

        # Add sparse scores
        for rank, chunk in enumerate(sparse_chunks):
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[chunk.content] = doc_scores.get(chunk.content, 0.0) + rrf_score
            if chunk.content not in doc_objects:
                doc_objects[chunk.content] = chunk

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k
        merged = []
        for doc_content, score in sorted_docs[:top_k]:
            chunk = doc_objects[doc_content]
            chunk.score = score
            chunk.source = "hybrid"
            merged.append(chunk)

        return merged

    async def _multi_hop_retrieval(self, original_query: str, query: str, top_k: int) -> RetrievalResult:
        """Multi-hop retrieval for complex questions."""
        logger.info("Performing multi-hop retrieval")

        all_chunks = []

        # First hop: Retrieve based on query
        first_hop = await self._single_retrieval(query, top_k)
        all_chunks.extend(first_hop.chunks)

        # Generate follow-up query based on first results
        context_summary = "\n".join([c.content[:200] for c in first_hop.chunks[:2]])

        followup_prompt = f"""Original question: {original_query}

Information found so far:
{context_summary}

What additional information should we look for? Generate a follow-up query.

Follow-up query:"""

        followup_query = await self.llm.query(followup_prompt)

        # Second hop: Retrieve based on follow-up
        second_hop = await self._single_retrieval(followup_query.strip(), top_k // 2)
        all_chunks.extend(second_hop.chunks)

        # Deduplicate
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.content not in seen:
                seen.add(chunk.content)
                unique_chunks.append(chunk)

        return RetrievalResult(
            chunks=unique_chunks[:top_k],
            query=original_query,
            transformed_query=query,
            num_chunks=len(unique_chunks[:top_k]),
            retrieval_method="multi_hop",
        )

    def _rerank_chunks(self, query: str, retrieval_result: RetrievalResult) -> RetrievalResult:
        """Rerank retrieved chunks using cross-encoder."""
        logger.debug(f"Reranking {len(retrieval_result.chunks)} chunks")

        # Prepare pairs for cross-encoder
        pairs = [[query, chunk.content] for chunk in retrieval_result.chunks]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Update scores
        for chunk, new_score in zip(retrieval_result.chunks, rerank_scores, strict=True):
            chunk.score = float(new_score)

        # Sort by new scores
        retrieval_result.chunks.sort(key=lambda x: x.score, reverse=True)

        return retrieval_result

    def _filter_relevant_chunks(
        self, chunks: list[RetrievedChunk], relevance_threshold: float = 0.3
    ) -> list[RetrievedChunk]:
        """Filter out low-relevance chunks."""
        filtered = [c for c in chunks if c.score >= relevance_threshold]

        logger.debug(f"Filtered {len(chunks)} -> {len(filtered)} chunks (threshold={relevance_threshold})")

        return filtered

    async def _generate_with_context(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Generate answer using retrieved context."""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}] {chunk.content}")

        context = "\n\n".join(context_parts)

        # Generate answer
        rag_prompt = f"""Use the following sources to answer the question. If the sources don't contain enough information, say so.

Sources:
{context}

Question: {query}

Answer (cite sources by number):"""

        answer = await self.llm.query(rag_prompt)

        return answer.strip()


class SelfRAG:
    """
    Self-RAG: Agent decides when to retrieve and critiques retrieval usage.

    Based on "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2024)
    """

    def __init__(self, rag_system: AdvancedRAG, llm_client: LLMClient):
        """
        Initialize Self-RAG.

        Args:
            rag_system: RAG system for retrieval
            llm_client: LLM client
        """
        self.rag = rag_system
        self.llm = llm_client
        logger.info("Initialized SelfRAG")

    async def query_with_reflection(self, query: str) -> dict[str, Any]:
        """
        Query with self-reflective retrieval.

        Args:
            query: User query

        Returns:
            Answer with reflection metadata
        """
        logger.info(f"Self-RAG query: {query[:100]}...")

        # Decide if retrieval is needed
        should_retrieve = await self._should_retrieve(query)

        if should_retrieve:
            # Retrieve
            retrieval_result = await self.rag.query(query, rerank=True)

            # Check if retrieved content is relevant
            is_relevant = await self._is_relevant(query, retrieval_result["sources"])

            # Check if answer is supported by sources
            is_supported = await self._is_supported(retrieval_result["answer"], retrieval_result["sources"])

            return {
                "answer": retrieval_result["answer"],
                "retrieval_used": True,
                "sources": retrieval_result["sources"],
                "reflection": {
                    "should_retrieve": should_retrieve,
                    "is_relevant": is_relevant,
                    "is_supported": is_supported,
                    "confidence": "high"
                    if is_relevant and is_supported
                    else "medium"
                    if is_relevant or is_supported
                    else "low",
                },
            }

        else:
            # Generate without retrieval
            answer = await self.llm.query(query)

            return {
                "answer": answer,
                "retrieval_used": False,
                "sources": [],
                "reflection": {"should_retrieve": False, "confidence": "medium"},
            }

    async def _should_retrieve(self, query: str) -> bool:
        """Decide if retrieval is needed."""
        # Simple heuristic: Retrieve for factual questions, not for opinion/creative
        decision_prompt = f"""Question: {query}

Does this question require external knowledge/facts to answer, or can it be answered with reasoning alone?

Answer with RETRIEVE or NO_RETRIEVE:"""

        response = await self.llm.query(decision_prompt)

        return "RETRIEVE" in response.upper()

    async def _is_relevant(self, query: str, sources: list[dict]) -> bool:
        """Check if retrieved sources are relevant."""
        if not sources:
            return False

        source_summary = "\n".join([s["content"][:100] for s in sources[:2]])

        relevance_prompt = f"""Query: {query}

Retrieved sources:
{source_summary}

Are these sources relevant to answering the query?

Answer YES or NO:"""

        response = await self.llm.query(relevance_prompt)

        return "YES" in response.upper()

    async def _is_supported(self, answer: str, sources: list[dict]) -> bool:
        """Check if answer is supported by sources."""
        if not sources:
            return False

        source_text = "\n".join([s["content"][:150] for s in sources[:3]])

        support_prompt = f"""Answer: {answer}

Sources:
{source_text}

Is the answer supported by these sources?

Answer YES or NO:"""

        response = await self.llm.query(support_prompt)

        return "YES" in response.upper()
