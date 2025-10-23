"""
Explainability Module - Interpretable AI Decisions

Provides explanations for agent decisions and reasoning chains.
"""

import re
from typing import Any

from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class ExplainabilityModule:
    """Makes agent decisions interpretable and explainable."""

    def __init__(self):
        """Initialize explainability module."""
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tfidf = TfidfVectorizer(max_features=20, stop_words="english")
        logger.info("Initialized ExplainabilityModule")

    def generate_explanation(
        self, decision: str, reasoning_chain: list[str], context: str | None = None
    ) -> dict[str, Any]:
        """
        Generate explanation for a decision.

        Args:
            decision: Final decision made
            reasoning_chain: Steps of reasoning
            context: Original context/problem

        Returns:
            Comprehensive explanation
        """
        logger.info("Generating decision explanation")

        explanation = {
            "decision": decision,
            "key_factors": self._extract_key_factors(reasoning_chain),
            "reasoning_steps": self._summarize_reasoning(reasoning_chain),
            "confidence_indicators": self._identify_confidence_markers(decision),
            "important_terms": self._extract_important_terms(reasoning_chain),
        }

        if context:
            explanation["relevance_to_context"] = self._assess_relevance(decision, context)

        return explanation

    def _extract_key_factors(self, reasoning_chain: list[str]) -> list[str]:
        """Extract key factors from reasoning."""
        key_factors = []

        for step in reasoning_chain:
            # Look for causal language
            if re.search(r"\b(because|since|due to|therefore|thus)\b", step, re.IGNORECASE):
                # Extract the clause after causal marker
                causal_match = re.search(r"\b(?:because|since|due to)\s+([^.]+)", step, re.IGNORECASE)
                if causal_match:
                    key_factors.append(causal_match.group(1).strip())

        return key_factors[:5]  # Top 5

    def _summarize_reasoning(self, reasoning_chain: list[str]) -> list[str]:
        """Summarize reasoning steps."""
        if not reasoning_chain:
            return []

        # Simple summarization: first sentence of each step
        summaries = []
        for step in reasoning_chain:
            first_sentence = step.split(".")[0].strip()
            if first_sentence:
                summaries.append(first_sentence)

        return summaries

    def _identify_confidence_markers(self, text: str) -> dict[str, Any]:
        """Identify confidence markers in text."""
        high_confidence = [r"\bclearly\b", r"\bobviously\b", r"\bcertainly\b", r"\bdefinitely\b", r"\bundoubtedly\b"]

        low_confidence = [
            r"\bmaybe\b",
            r"\bmight\b",
            r"\bpossibly\b",
            r"\bperhaps\b",
            r"\buncertain\b",
            r"\bcould be\b",
        ]

        high_count = sum(1 for pattern in high_confidence if re.search(pattern, text, re.IGNORECASE))
        low_count = sum(1 for pattern in low_confidence if re.search(pattern, text, re.IGNORECASE))

        if high_count > low_count:
            confidence_level = "high"
        elif low_count > high_count:
            confidence_level = "low"
        else:
            confidence_level = "medium"

        return {"level": confidence_level, "high_confidence_markers": high_count, "low_confidence_markers": low_count}

    def _extract_important_terms(self, reasoning_chain: list[str]) -> list[str]:
        """Extract important terms using TF-IDF."""
        if not reasoning_chain:
            return []

        try:
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(reasoning_chain)
            feature_names = self.tfidf.get_feature_names_out()

            # Get top terms
            importance_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = importance_scores.argsort()[-10:][::-1]

            top_terms = [feature_names[i] for i in top_indices]
            return top_terms

        except Exception as e:
            logger.warning(f"Term extraction failed: {e}")
            return []

    def _assess_relevance(self, decision: str, context: str) -> float:
        """Assess how relevant decision is to context."""
        decision_embedding = self.embedding_model.encode([decision])
        context_embedding = self.embedding_model.encode([context])

        from sklearn.metrics.pairwise import cosine_similarity

        relevance = cosine_similarity(decision_embedding, context_embedding)[0][0]

        return float(relevance)

    def generate_counterfactual(
        self, original_decision: str, reasoning_chain: list[str], factor_to_change: str
    ) -> dict[str, str]:
        """
        Generate counterfactual explanation.

        Args:
            original_decision: Original decision
            reasoning_chain: Original reasoning
            factor_to_change: Factor to modify

        Returns:
            Counterfactual explanation
        """
        counterfactual = {
            "original_decision": original_decision,
            "changed_factor": factor_to_change,
            "counterfactual_explanation": f"If {factor_to_change} were different, the decision might change because...",
        }

        # In production, use LLM to generate actual counterfactual
        return counterfactual

    def explain_comparison(self, decision_a: str, decision_b: str, context: str) -> dict[str, Any]:
        """
        Explain why two decisions differ.

        Args:
            decision_a: First decision
            decision_b: Second decision
            context: Shared context

        Returns:
            Comparative explanation
        """
        # Tokenize and find differences
        words_a = set(decision_a.lower().split())
        words_b = set(decision_b.lower().split())

        unique_to_a = words_a - words_b
        unique_to_b = words_b - words_a
        shared = words_a & words_b

        return {
            "unique_to_decision_a": list(unique_to_a)[:10],
            "unique_to_decision_b": list(unique_to_b)[:10],
            "shared_concepts": list(shared)[:10],
            "divergence_score": len(unique_to_a) + len(unique_to_b),
            "explanation": f"Decisions differ primarily in: {', '.join(list(unique_to_a | unique_to_b)[:5])}",
        }


class AttentionVisualizer:
    """Visualizes attention/importance of different reasoning steps."""

    def __init__(self):
        """Initialize attention visualizer."""
        logger.info("Initialized AttentionVisualizer")

    def visualize_step_importance(self, reasoning_steps: list[str], final_decision: str) -> dict[str, Any]:
        """
        Calculate importance of each reasoning step to final decision.

        Args:
            reasoning_steps: List of reasoning steps
            final_decision: Final decision text

        Returns:
            Step importance scores
        """
        if not reasoning_steps:
            return {"error": "No reasoning steps"}

        # Simple heuristic: semantic similarity to final decision
        model = SentenceTransformer("all-MiniLM-L6-v2")

        decision_embedding = model.encode([final_decision])
        step_embeddings = model.encode(reasoning_steps)

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(step_embeddings, decision_embedding).flatten()

        # Normalize to 0-1
        if similarities.max() > 0:
            normalized = similarities / similarities.max()
        else:
            normalized = similarities

        step_importance = [
            {"step_index": i, "step_text": step[:100], "importance": float(score)}
            for i, (step, score) in enumerate(zip(reasoning_steps, normalized))
        ]

        # Sort by importance
        step_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "step_importance": step_importance,
            "most_important_step": step_importance[0] if step_importance else None,
        }
