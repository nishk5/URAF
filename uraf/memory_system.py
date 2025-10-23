"""
Memory System - Persistent Vector Memory with ChromaDB

Implements long-term memory for agents inspired by MemGPT and Reflexion.
Supports episodic, semantic, and procedural memory types.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from loguru import logger


class MemoryEntry:
    """Single memory entry with metadata."""

    def __init__(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict] = None,
        importance: float = 0.5
    ):
        """
        Initialize memory entry.

        Args:
            content: Memory content text
            memory_type: Type of memory (episodic, semantic, procedural)
            metadata: Additional metadata
            importance: Importance score (0-1)
        """
        self.id = f"{memory_type}_{datetime.now().timestamp()}"
        self.content = content
        self.memory_type = memory_type
        self.timestamp = datetime.now().isoformat()
        self.importance = importance
        self.access_count = 0
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            **self.metadata
        }


class AgentMemory:
    """
    Comprehensive memory system for agents.

    Memory Types:
    - Episodic: Specific experiences and interactions
    - Semantic: General knowledge and facts
    - Procedural: Strategies and methods that worked
    """

    def __init__(
        self,
        persist_directory: str = "data/memory",
        collection_name: str = "agent_memory",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize agent memory system.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_directory)
            ))

            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing memory collection: {collection_name}")
            except:
                self.collection = self.client.create_collection(name=collection_name)
                logger.info(f"Created new memory collection: {collection_name}")
        else:
            logger.warning("ChromaDB not available, using in-memory fallback")
            self.collection = None
            self._fallback_memory = []

        # In-memory caches for fast access
        self.recent_memories: List[MemoryEntry] = []
        self.max_recent_cache = 50

        logger.info(f"Initialized AgentMemory at {persist_directory}")

    async def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a new memory.

        Args:
            content: Memory content
            memory_type: Type (episodic/semantic/procedural)
            importance: Importance score (0-1)
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        entry = MemoryEntry(content, memory_type, metadata, importance)

        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()

        # Store in ChromaDB
        if self.collection:
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[entry.to_dict()],
                ids=[entry.id]
            )
        else:
            # Fallback storage
            self._fallback_memory.append({
                "id": entry.id,
                "content": content,
                "embedding": embedding,
                "metadata": entry.to_dict()
            })

        # Add to recent cache
        self.recent_memories.append(entry)
        if len(self.recent_memories) > self.max_recent_cache:
            self.recent_memories.pop(0)

        logger.debug(f"Stored {memory_type} memory: {entry.id}")
        return entry.id

    async def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        importance_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories by semantic similarity.

        Args:
            query: Search query
            memory_type: Filter by memory type (optional)
            top_k: Number of results
            importance_threshold: Minimum importance score

        Returns:
            List of relevant memories with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        if self.collection:
            # Query ChromaDB
            where_filter = {}
            if memory_type:
                where_filter["memory_type"] = memory_type

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None
            )

            memories = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0.0

                    # Filter by importance
                    if metadata.get('importance', 0) >= importance_threshold:
                        memories.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity": 1 - distance,  # Convert distance to similarity
                            "id": results['ids'][0][i]
                        })

            return memories
        else:
            # Fallback similarity search
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            similarities = []
            for mem in self._fallback_memory:
                sim = cosine_similarity(
                    [query_embedding],
                    [mem['embedding']]
                )[0][0]

                if memory_type and mem['metadata'].get('memory_type') != memory_type:
                    continue

                if mem['metadata'].get('importance', 0) >= importance_threshold:
                    similarities.append({
                        "content": mem['content'],
                        "metadata": mem['metadata'],
                        "similarity": float(sim),
                        "id": mem['id']
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]

    async def retrieve_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve most recent memories.

        Args:
            n: Number of recent memories

        Returns:
            Recent memories list
        """
        recent = self.recent_memories[-n:]
        return [
            {
                "content": mem.content,
                "metadata": mem.to_dict(),
                "id": mem.id
            }
            for mem in reversed(recent)
        ]

    async def consolidate(self, time_window_hours: int = 24):
        """
        Consolidate memories (like human sleep).

        Merges similar memories and adjusts importance scores.

        Args:
            time_window_hours: Time window for consolidation
        """
        logger.info(f"Starting memory consolidation (window: {time_window_hours}h)")

        # Get recent memories
        recent = await self.retrieve_recent(n=100)

        # Find similar memory clusters
        clusters = self._cluster_similar_memories(recent)

        # Merge clusters and boost importance
        for cluster in clusters:
            if len(cluster) > 1:
                # Create consolidated memory
                contents = [m['content'] for m in cluster]
                consolidated_content = self._merge_memories(contents)

                # Boost importance
                avg_importance = sum(m['metadata'].get('importance', 0.5) for m in cluster) / len(cluster)
                boosted_importance = min(1.0, avg_importance * 1.2)

                # Store consolidated memory
                await self.store(
                    content=consolidated_content,
                    memory_type="semantic",  # Consolidated memories become semantic
                    importance=boosted_importance,
                    metadata={"consolidated_from": len(cluster)}
                )

                logger.debug(f"Consolidated {len(cluster)} memories into semantic memory")

        logger.info("Memory consolidation complete")

    def _cluster_similar_memories(self, memories: List[Dict], threshold: float = 0.85) -> List[List[Dict]]:
        """
        Cluster similar memories together.

        Args:
            memories: List of memories
            threshold: Similarity threshold

        Returns:
            List of memory clusters
        """
        if not memories:
            return []

        # Simple clustering by pairwise similarity
        clusters = []
        used = set()

        for i, mem1 in enumerate(memories):
            if i in used:
                continue

            cluster = [mem1]
            used.add(i)

            for j, mem2 in enumerate(memories[i+1:], start=i+1):
                if j in used:
                    continue

                # Calculate similarity
                emb1 = self.embedding_model.encode(mem1['content'])
                emb2 = self.embedding_model.encode(mem2['content'])

                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity([emb1], [emb2])[0][0]

                if sim >= threshold:
                    cluster.append(mem2)
                    used.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _merge_memories(self, contents: List[str]) -> str:
        """
        Merge multiple memory contents into one.

        Args:
            contents: List of memory contents

        Returns:
            Merged content
        """
        # Simple merging - in production, use LLM for better summarization
        unique_sentences = set()
        for content in contents:
            sentences = content.split('. ')
            unique_sentences.update(sentences)

        return '. '.join(sorted(unique_sentences)[:5])  # Top 5 sentences

    async def forget(self, memory_id: str):
        """
        Remove a memory by ID.

        Args:
            memory_id: Memory identifier
        """
        if self.collection:
            self.collection.delete(ids=[memory_id])
        else:
            self._fallback_memory = [m for m in self._fallback_memory if m['id'] != memory_id]

        logger.info(f"Forgot memory: {memory_id}")

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage.

        Returns:
            Memory statistics
        """
        if self.collection:
            count = self.collection.count()
        else:
            count = len(self._fallback_memory)

        # Count by type
        type_counts = {"episodic": 0, "semantic": 0, "procedural": 0}

        if self.collection:
            # Note: ChromaDB doesn't support aggregation, need to fetch all
            pass  # Simplified for now
        else:
            for mem in self._fallback_memory:
                mem_type = mem['metadata'].get('memory_type', 'episodic')
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        return {
            "total_memories": count,
            "memory_types": type_counts,
            "recent_cache_size": len(self.recent_memories),
            "persist_directory": str(self.persist_directory)
        }


class WorkingMemory:
    """
    Short-term working memory for active context.

    Manages conversation context and temporary information.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize working memory.

        Args:
            max_size: Maximum number of items to keep
        """
        self.max_size = max_size
        self.memory: List[Dict[str, Any]] = []
        logger.info(f"Initialized WorkingMemory (max_size={max_size})")

    def add(self, content: str, role: str = "user", metadata: Optional[Dict] = None):
        """
        Add item to working memory.

        Args:
            content: Content text
            role: Role (user/assistant/system)
            metadata: Additional metadata
        """
        entry = {
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.memory.append(entry)

        # Evict oldest if over capacity
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
            logger.debug("Evicted oldest working memory entry")

    def get_context(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent context.

        Args:
            n: Number of recent items (None = all)

        Returns:
            Recent memory entries
        """
        if n is None:
            return self.memory

        return self.memory[-n:]

    def clear(self):
        """Clear all working memory."""
        self.memory = []
        logger.info("Cleared working memory")

    def get_summary(self) -> str:
        """
        Get summary of working memory.

        Returns:
            Summary text
        """
        if not self.memory:
            return "No active context"

        summary = f"Working Memory ({len(self.memory)} entries):\n"
        for entry in self.memory[-5:]:  # Last 5
            summary += f"- [{entry['role']}] {entry['content'][:50]}...\n"

        return summary
