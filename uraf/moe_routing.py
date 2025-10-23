"""
Mixture of Experts (MoE) Routing System

Routes tasks to specialized expert models based on task classification.
"""

import re
from typing import Dict, Optional, Any, List
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ExpertRouter:
    """Routes tasks to specialized expert models."""

    def __init__(self):
        """Initialize expert router with predefined experts."""
        self.experts = {
            "math": {
                "models": ["deepseek-math-7b", "qwen2.5-math-7b"],
                "keywords": ["calculate", "equation", "solve", "mathematical", "arithmetic", "algebra"],
                "description": "Mathematical reasoning and problem solving"
            },
            "code": {
                "models": ["codellama-34b", "deepseek-coder-33b"],
                "keywords": ["code", "function", "programming", "algorithm", "debug", "implement"],
                "description": "Code generation and debugging"
            },
            "reasoning": {
                "models": ["qwen2.5-72b", "claude-3-opus", "gpt-4"],
                "keywords": ["analyze", "reason", "logic", "deduce", "infer", "conclude"],
                "description": "Complex reasoning and analysis"
            },
            "creative": {
                "models": ["claude-3-sonnet", "gpt-4-turbo"],
                "keywords": ["creative", "story", "write", "compose", "generate", "imagine"],
                "description": "Creative writing and content generation"
            },
            "general": {
                "models": ["gpt-4", "claude-3-opus"],
                "keywords": [],
                "description": "General-purpose tasks"
            }
        }

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"Initialized ExpertRouter with {len(self.experts)} expert types")

    def classify_task(self, task: str) -> str:
        """
        Classify task to determine expert type.

        Args:
            task: Task description

        Returns:
            Expert type (math/code/reasoning/creative/general)
        """
        task_lower = task.lower()

        # Keyword matching
        matches = {}
        for expert_type, expert_info in self.experts.items():
            if expert_type == "general":
                continue

            keyword_matches = sum(
                1 for keyword in expert_info["keywords"]
                if keyword in task_lower
            )
            matches[expert_type] = keyword_matches

        # Select expert with most keyword matches
        if max(matches.values()) > 0:
            expert_type = max(matches, key=matches.get)
        else:
            expert_type = "general"

        logger.info(f"Classified task as: {expert_type}")
        return expert_type

    def route(self, task: str, available_models: Optional[List[str]] = None) -> str:
        """
        Route task to best available expert model.

        Args:
            task: Task description
            available_models: List of available models (optional)

        Returns:
            Recommended model name
        """
        expert_type = self.classify_task(task)
        expert_models = self.experts[expert_type]["models"]

        # Filter by available models
        if available_models:
            expert_models = [m for m in expert_models if m in available_models]

        if not expert_models:
            # Fallback to general
            expert_models = self.experts["general"]["models"]

        recommended = expert_models[0]
        logger.info(f"Routed to: {recommended} (type: {expert_type})")

        return recommended

    def get_expert_info(self, expert_type: str) -> Dict[str, Any]:
        """Get information about an expert type."""
        return self.experts.get(expert_type, {})


class EnsembleAggregator:
    """Aggregates responses from multiple expert models."""

    def __init__(self):
        """Initialize ensemble aggregator."""
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Initialized EnsembleAggregator")

    async def aggregate(
        self,
        responses: Dict[str, str],
        method: str = "voting"
    ) -> Dict[str, Any]:
        """
        Aggregate multiple expert responses.

        Args:
            responses: Dict mapping model names to responses
            method: Aggregation method (voting/weighted/semantic)

        Returns:
            Aggregated result
        """
        if not responses:
            return {"error": "No responses to aggregate"}

        if len(responses) == 1:
            return {
                "aggregated_response": list(responses.values())[0],
                "method": "single",
                "num_models": 1
            }

        if method == "voting":
            # Simple majority voting (for classification tasks)
            return self._majority_vote(responses)
        elif method == "weighted":
            # Weighted by model confidence
            return self._weighted_aggregate(responses)
        elif method == "semantic":
            # Semantic clustering and selection
            return self._semantic_aggregate(responses)
        else:
            # Default: concatenate
            combined = "\n\n".join([
                f"[{model}]: {response}"
                for model, response in responses.items()
            ])
            return {
                "aggregated_response": combined,
                "method": "concatenation",
                "num_models": len(responses)
            }

    def _majority_vote(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """Simple majority voting."""
        # Count occurrences of each response
        from collections import Counter
        response_list = list(responses.values())
        counts = Counter(response_list)
        most_common = counts.most_common(1)[0]

        return {
            "aggregated_response": most_common[0],
            "method": "majority_vote",
            "votes": most_common[1],
            "num_models": len(responses)
        }

    def _weighted_aggregate(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """Weighted aggregation (placeholder)."""
        # In production, use model confidence scores
        return self._majority_vote(responses)

    def _semantic_aggregate(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """Aggregate based on semantic similarity."""
        response_texts = list(responses.values())
        embeddings = self.embedding_model.encode(response_texts)

        # Find most central response (highest average similarity to others)
        similarities = cosine_similarity(embeddings)
        avg_similarities = similarities.mean(axis=1)
        most_central_idx = np.argmax(avg_similarities)

        return {
            "aggregated_response": response_texts[most_central_idx],
            "method": "semantic_center",
            "centrality_score": float(avg_similarities[most_central_idx]),
            "num_models": len(responses)
        }
