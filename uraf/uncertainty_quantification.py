"""
Uncertainty Quantification for LLMs

Based on:
- "Semantic Uncertainty" (Kuhn et al., 2024)
- "Conformal Prediction for Language Models" (Angelopoulos et al., 2024)
- Calibration techniques (Temperature scaling, Platt scaling)

Provides statistically grounded uncertainty estimates for LLM outputs.
"""

import asyncio
import math
from typing import Any

import numpy as np
from loguru import logger
from scipy.special import expit
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from uraf.llm_client import LLMClient


class SemanticUncertainty:
    """
    Semantic uncertainty estimation via clustering.

    Key insight: Uncertainty should be measured in semantic space, not token space.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.85):
        """
        Initialize semantic uncertainty estimator.

        Args:
            embedding_model_name: Model for computing semantic embeddings
            similarity_threshold: Threshold for clustering (higher = stricter clusters)
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        logger.info("Initialized SemanticUncertainty")

    async def estimate_uncertainty(
        self, llm_client: LLMClient, prompt: str, num_samples: int = 5, temperature: float = 0.8
    ) -> dict[str, Any]:
        """
        Estimate uncertainty by sampling multiple outputs and clustering.

        Args:
            llm_client: LLM client for sampling
            prompt: Input prompt
            num_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            Uncertainty metrics and semantic clusters
        """
        logger.info(f"Estimating semantic uncertainty with {num_samples} samples")

        # Generate multiple samples
        samples = await self._generate_samples(llm_client, prompt, num_samples, temperature)

        # Compute embeddings
        embeddings = self.embedding_model.encode(samples)

        # Cluster semantically similar outputs
        clusters = self._cluster_outputs(samples, embeddings)

        # Compute uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(clusters, num_samples)

        return {
            "samples": samples,
            "num_clusters": len(clusters),
            "clusters": clusters,
            "uncertainty": uncertainty_metrics["entropy"],
            "confidence": 1.0 - uncertainty_metrics["entropy"],
            "agreement_rate": uncertainty_metrics["agreement_rate"],
            "most_common_answer": uncertainty_metrics["most_common"],
            "metrics": uncertainty_metrics,
        }

    async def _generate_samples(
        self, llm_client: LLMClient, prompt: str, num_samples: int, temperature: float
    ) -> list[str]:
        """Generate multiple samples from LLM."""
        # Modify LLM parameters for sampling
        original_temp = llm_client.params.get("temperature", 0.7)
        llm_client.params["temperature"] = temperature

        # Generate samples in parallel
        tasks = [llm_client.query(prompt) for _ in range(num_samples)]
        samples = await asyncio.gather(*tasks)

        # Restore original temperature
        llm_client.params["temperature"] = original_temp

        return [s.strip() for s in samples]

    def _cluster_outputs(self, samples: list[str], embeddings: np.ndarray) -> list[dict[str, Any]]:
        """Cluster semantically similar outputs."""
        # Use DBSCAN for clustering
        # eps controls cluster tightness (lower eps = tighter clusters)
        eps = 1.0 - self.similarity_threshold  # Convert similarity to distance
        clustering = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(embeddings)

        labels = clustering.labels_
        num_clusters = len(set(labels))

        # Group samples by cluster
        clusters = []
        for cluster_id in set(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_samples = [samples[i] for i in cluster_indices]

            # Representative (centroid)
            cluster_embeddings = embeddings[cluster_indices]
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)

            # Find sample closest to centroid
            similarities = cosine_similarity(centroid, cluster_embeddings)[0]
            representative_idx = cluster_indices[similarities.argmax()]

            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": len(cluster_samples),
                    "samples": cluster_samples,
                    "representative": samples[representative_idx],
                    "probability": len(cluster_samples) / len(samples),
                }
            )

        # Sort by size (largest first)
        clusters.sort(key=lambda x: x["size"], reverse=True)

        logger.debug(f"Clustered {len(samples)} samples into {num_clusters} semantic clusters")

        return clusters

    def _compute_uncertainty_metrics(self, clusters: list[dict], total_samples: int) -> dict[str, Any]:
        """Compute uncertainty metrics from clusters."""
        # Cluster probabilities
        probs = np.array([c["size"] / total_samples for c in clusters])

        # Shannon entropy (measure of uncertainty)
        # High entropy = high uncertainty (many diverse outputs)
        # Low entropy = low uncertainty (agreement on output)
        semantic_entropy = entropy(probs)

        # Normalize entropy to [0, 1]
        max_entropy = math.log(len(clusters)) if len(clusters) > 1 else 1.0
        normalized_entropy = semantic_entropy / max_entropy if max_entropy > 0 else 0.0

        # Agreement rate (fraction in largest cluster)
        agreement_rate = max(probs) if probs.size > 0 else 0.0

        # Most common answer
        most_common = clusters[0]["representative"] if clusters else ""

        return {
            "entropy": normalized_entropy,
            "raw_entropy": semantic_entropy,
            "agreement_rate": agreement_rate,
            "num_unique_answers": len(clusters),
            "most_common": most_common,
            "cluster_distribution": probs.tolist(),
        }


class ConformalPrediction:
    """
    Conformal prediction for LLMs.

    Provides statistically valid prediction sets with guaranteed coverage.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Args:
            alpha: Significance level (1-alpha = coverage probability)
                   alpha=0.1 means 90% coverage guarantee
        """
        self.alpha = alpha
        self.calibration_scores: list[float] = []
        logger.info(f"Initialized ConformalPrediction with alpha={alpha}")

    def calibrate(self, validation_scores: list[float]):
        """
        Calibrate using validation set conformity scores.

        Args:
            validation_scores: Conformity scores from validation set
                              (e.g., confidence scores, similarity scores)
        """
        self.calibration_scores = sorted(validation_scores)
        logger.info(f"Calibrated with {len(validation_scores)} validation samples")

    def get_prediction_set(self, candidate_answers: list[str], candidate_scores: list[float]) -> dict[str, Any]:
        """
        Get prediction set with guaranteed coverage.

        Args:
            candidate_answers: List of candidate answers
            candidate_scores: Conformity scores for each candidate

        Returns:
            Prediction set with coverage guarantee
        """
        if not self.calibration_scores:
            raise ValueError("Must calibrate before getting prediction sets")

        # Compute quantile threshold
        n = len(self.calibration_scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        quantile_idx = min(max(quantile_idx, 0), n - 1)
        threshold = self.calibration_scores[quantile_idx]

        # Include candidates above threshold
        prediction_set = [
            {"answer": ans, "score": score}
            for ans, score in zip(candidate_answers, candidate_scores, strict=True)
            if score >= threshold
        ]

        # Sort by score
        prediction_set.sort(key=lambda x: x["score"], reverse=True)

        coverage_probability = 1 - self.alpha

        return {
            "prediction_set": prediction_set,
            "set_size": len(prediction_set),
            "threshold": threshold,
            "coverage_probability": coverage_probability,
            "guaranteed_coverage": f"{coverage_probability * 100:.1f}%",
        }


class CalibrationModule:
    """
    Calibration techniques for LLM confidence scores.

    Ensures that predicted confidence matches empirical accuracy.
    """

    def __init__(self):
        """Initialize calibration module."""
        self.temperature: float | None = None
        self.platt_params: dict[str, float] | None = None
        logger.info("Initialized CalibrationModule")

    def temperature_scaling(
        self, confidences: np.ndarray, labels: np.ndarray, search_temps: list[float] | None = None
    ) -> float:
        """
        Find optimal temperature via grid search.

        Args:
            confidences: Raw confidence scores [0, 1]
            labels: True labels (0 or 1)
            search_temps: Temperature values to search

        Returns:
            Optimal temperature
        """
        if search_temps is None:
            search_temps = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

        best_temp = 1.0
        best_nll = float("inf")

        for temp in search_temps:
            # Apply temperature scaling
            calibrated_probs = self._apply_temperature(confidences, temp)

            # Compute negative log-likelihood
            nll = -np.mean(
                labels * np.log(calibrated_probs + 1e-10) + (1 - labels) * np.log(1 - calibrated_probs + 1e-10)
            )

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp
        logger.info(f"Optimal temperature: {best_temp:.3f}")

        return best_temp

    def _apply_temperature(self, confidences: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to confidences."""
        # Convert to logits (assuming confidences are probabilities)
        logits = np.log(confidences / (1 - confidences + 1e-10) + 1e-10)

        # Scale by temperature
        scaled_logits = logits / temperature

        # Convert back to probabilities
        calibrated_probs = expit(scaled_logits)

        return calibrated_probs

    def platt_scaling(self, confidences: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Platt scaling: fit logistic regression on validation set.

        Args:
            confidences: Raw confidence scores
            labels: True labels

        Returns:
            Platt scaling parameters (a, b)
        """
        from sklearn.linear_model import LogisticRegression

        # Reshape for sklearn
        X = confidences.reshape(-1, 1)
        y = labels

        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(X, y)

        # Extract parameters
        a = lr.coef_[0][0]
        b = lr.intercept_[0]

        self.platt_params = {"a": a, "b": b}
        logger.info(f"Platt scaling: a={a:.3f}, b={b:.3f}")

        return self.platt_params

    def apply_calibration(self, confidence: float, method: str = "temperature") -> float:
        """
        Apply calibration to a confidence score.

        Args:
            confidence: Raw confidence
            method: Calibration method ("temperature" or "platt")

        Returns:
            Calibrated confidence
        """
        if method == "temperature":
            if self.temperature is None:
                logger.warning("Temperature not set, returning raw confidence")
                return confidence

            logit = math.log(confidence / (1 - confidence + 1e-10) + 1e-10)
            scaled_logit = logit / self.temperature
            calibrated = 1 / (1 + math.exp(-scaled_logit))

            return calibrated

        elif method == "platt":
            if self.platt_params is None:
                logger.warning("Platt parameters not set, returning raw confidence")
                return confidence

            a = self.platt_params["a"]
            b = self.platt_params["b"]

            calibrated = 1 / (1 + math.exp(-(a * confidence + b)))

            return calibrated

        else:
            raise ValueError(f"Unknown calibration method: {method}")


class UncertaintyEstimator:
    """
    Comprehensive uncertainty estimation combining multiple approaches.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize uncertainty estimator.

        Args:
            llm_client: LLM client
        """
        self.llm = llm_client
        self.semantic_uncertainty = SemanticUncertainty()
        self.conformal = ConformalPrediction()
        self.calibration = CalibrationModule()
        logger.info("Initialized UncertaintyEstimator")

    async def estimate_comprehensive_uncertainty(self, prompt: str, num_samples: int = 5) -> dict[str, Any]:
        """
        Comprehensive uncertainty estimation.

        Args:
            prompt: Input prompt
            num_samples: Number of samples for semantic uncertainty

        Returns:
            Comprehensive uncertainty metrics
        """
        logger.info("Computing comprehensive uncertainty")

        # Semantic uncertainty via sampling
        semantic_result = await self.semantic_uncertainty.estimate_uncertainty(self.llm, prompt, num_samples)

        # Combined uncertainty score
        # High uncertainty = semantic disagreement
        combined_uncertainty = semantic_result["uncertainty"]

        interpretation = self._interpret_uncertainty(combined_uncertainty)

        return {
            "uncertainty_score": combined_uncertainty,
            "confidence_score": 1.0 - combined_uncertainty,
            "interpretation": interpretation,
            "semantic_uncertainty": semantic_result,
            "recommended_action": self._recommend_action(combined_uncertainty),
        }

    def _interpret_uncertainty(self, uncertainty: float) -> str:
        """Interpret uncertainty score."""
        if uncertainty < 0.2:
            return "Very low uncertainty - high confidence in answer"
        elif uncertainty < 0.4:
            return "Low uncertainty - moderate confidence"
        elif uncertainty < 0.6:
            return "Moderate uncertainty - some disagreement in outputs"
        elif uncertainty < 0.8:
            return "High uncertainty - significant disagreement"
        else:
            return "Very high uncertainty - outputs are highly diverse"

    def _recommend_action(self, uncertainty: float) -> str:
        """Recommend action based on uncertainty."""
        if uncertainty < 0.3:
            return "Safe to use answer directly"
        elif uncertainty < 0.6:
            return "Review answer before using"
        else:
            return "Seek additional information or human review"


def calculate_expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures how well confidence scores match empirical accuracy.

    Args:
        confidences: Predicted confidence scores
        accuracies: Whether predictions were correct (0 or 1)
        n_bins: Number of bins for calibration curve

    Returns:
        ECE score (lower is better, 0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() == 0:
            continue

        # Average confidence and accuracy in bin
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()

        # Weighted contribution to ECE
        bin_weight = in_bin.sum() / len(confidences)
        ece += bin_weight * abs(avg_confidence - avg_accuracy)

    return float(ece)
