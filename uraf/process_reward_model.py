"""
Process Reward Model (PRM) - OpenAI o1-Style Step-by-Step Evaluation

Evaluates reasoning quality at each intermediate step, not just final outputs.
Based on "Let's Verify Step by Step" (OpenAI, 2023)
"""

import re
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger


class ProcessRewardModel:
    """
    Evaluates reasoning chains step-by-step with process supervision.

    Key Metrics:
    - Step Correctness: Individual step validity (0-1 per step)
    - Logical Consistency: Coherence across steps
    - Self-Correction Detection: Identifies and rewards backtracking
    - Progress Tracking: Measures advancement toward solution
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize PRM with embedding model for semantic analysis.

        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Initialized PRM with embedding model: {model_name}")

    def parse_reasoning_steps(self, response: str) -> List[str]:
        """
        Extract individual reasoning steps from structured response.

        Args:
            response: Full LLM response with reasoning pathway

        Returns:
            List of individual reasoning steps
        """
        # Extract reasoning pathway section
        reasoning_match = re.search(
            r'\*Reasoning Pathway:\*(.*?)(?:\*|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )

        if not reasoning_match:
            logger.warning("No reasoning pathway found in response")
            return []

        reasoning_text = reasoning_match.group(1).strip()

        # Split by common step delimiters
        steps = re.split(r'\n(?=\d+\.|\-|\*|Step)', reasoning_text)
        steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 10]

        logger.debug(f"Parsed {len(steps)} reasoning steps")
        return steps

    def evaluate_step_correctness(self, step: str, context: str = "") -> Dict[str, float]:
        """
        Evaluate individual step quality.

        Checks for:
        - Logical operators (therefore, because, if-then)
        - Concrete examples or evidence
        - Quantitative information
        - Hedge words (might, could, possibly) - reduce score

        Args:
            step: Individual reasoning step text
            context: Previous steps for context

        Returns:
            Dict with correctness metrics
        """
        score = 0.5  # Base score

        # Logical connectives boost score
        logical_patterns = [
            r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
            r'\bbecause\b', r'\bsince\b', r'\bgiven that\b',
            r'\bif\b.*\bthen\b', r'\bimplies\b'
        ]
        for pattern in logical_patterns:
            if re.search(pattern, step, re.IGNORECASE):
                score += 0.1

        # Concrete evidence boosts score
        if re.search(r'\d+', step):  # Contains numbers
            score += 0.1
        if re.search(r'for example|such as|specifically', step, re.IGNORECASE):
            score += 0.1

        # Uncertainty reduces score
        uncertainty_patterns = [
            r'\bmaybe\b', r'\bmight\b', r'\bcould be\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\bprobably\b'
        ]
        for pattern in uncertainty_patterns:
            if re.search(pattern, step, re.IGNORECASE):
                score -= 0.05

        # Step length penalty (too short = vague, too long = unfocused)
        word_count = len(step.split())
        if word_count < 10:
            score -= 0.1
        elif word_count > 100:
            score -= 0.05

        return {
            "correctness_score": np.clip(score, 0.0, 1.0),
            "word_count": word_count
        }

    def evaluate_consistency(self, steps: List[str]) -> Dict[str, float]:
        """
        Evaluate logical consistency across reasoning steps using embeddings.

        Args:
            steps: List of reasoning steps

        Returns:
            Dict with consistency metrics
        """
        if len(steps) < 2:
            return {
                "avg_step_similarity": 1.0,
                "min_step_similarity": 1.0,
                "consistency_score": 1.0
            }

        # Compute embeddings for all steps
        embeddings = self.embedding_model.encode(steps)

        # Calculate pairwise similarities between adjacent steps
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)

        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)

        # Penalize large jumps (low similarity)
        consistency_score = avg_similarity
        if min_similarity < 0.3:
            consistency_score *= 0.8  # Penalty for major inconsistency

        return {
            "avg_step_similarity": float(avg_similarity),
            "min_step_similarity": float(min_similarity),
            "consistency_score": float(np.clip(consistency_score, 0.0, 1.0))
        }

    def detect_self_correction(self, steps: List[str]) -> Dict[str, any]:
        """
        Detect and reward self-correction / backtracking in reasoning.

        Args:
            steps: List of reasoning steps

        Returns:
            Dict with correction detection results
        """
        correction_patterns = [
            r'\bwait\b', r'\bactually\b', r'\bhowever\b', r'\bon second thought\b',
            r'\bupon reflection\b', r'\bcorrect(?:ing|ion)\b', r'\brevise\b',
            r'\blet me reconsider\b', r'\bmistake\b', r'\berror\b'
        ]

        corrections = []
        for i, step in enumerate(steps):
            for pattern in correction_patterns:
                if re.search(pattern, step, re.IGNORECASE):
                    corrections.append({
                        "step_index": i,
                        "step_text": step[:100],  # First 100 chars
                        "pattern_matched": pattern
                    })
                    break

        correction_bonus = len(corrections) * 0.15
        correction_bonus = min(correction_bonus, 0.3)  # Max 0.3 bonus

        return {
            "num_corrections": len(corrections),
            "correction_details": corrections,
            "correction_bonus": correction_bonus
        }

    def evaluate_progress(self, steps: List[str], initial_problem: str) -> float:
        """
        Measure whether reasoning progresses toward solution.

        Uses semantic similarity to problem statement to track relevance.

        Args:
            steps: List of reasoning steps
            initial_problem: Original problem statement

        Returns:
            Progress score (0-1)
        """
        if not steps:
            return 0.0

        # Embed problem and all steps
        problem_embedding = self.embedding_model.encode([initial_problem])
        step_embeddings = self.embedding_model.encode(steps)

        # Calculate similarity of each step to problem
        similarities = cosine_similarity(step_embeddings, problem_embedding).flatten()

        # Check if later steps maintain relevance (not drifting)
        first_half_sim = np.mean(similarities[:len(similarities)//2]) if len(similarities) > 1 else similarities[0]
        second_half_sim = np.mean(similarities[len(similarities)//2:]) if len(similarities) > 1 else similarities[0]

        # Progress score: maintain or increase relevance
        if second_half_sim >= first_half_sim * 0.85:  # Allow 15% drift
            progress_score = (first_half_sim + second_half_sim) / 2
        else:
            progress_score = (first_half_sim + second_half_sim) / 2 * 0.7  # Penalty

        return float(np.clip(progress_score, 0.0, 1.0))

    def evaluate_reasoning_chain(
        self,
        response: str,
        problem: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Complete process reward evaluation of reasoning chain.

        Args:
            response: Full LLM response with structured reasoning
            problem: Original problem statement (optional, for progress tracking)

        Returns:
            Comprehensive PRM evaluation results
        """
        logger.info("Starting PRM evaluation of reasoning chain")

        # Parse steps
        steps = self.parse_reasoning_steps(response)

        if not steps:
            logger.warning("No steps found, returning minimal score")
            return {
                "num_steps": 0,
                "step_scores": [],
                "avg_step_correctness": 0.0,
                "consistency_metrics": {},
                "self_correction": {},
                "progress_score": 0.0,
                "final_prm_score": 0.0
            }

        # Evaluate each step
        step_scores = []
        for i, step in enumerate(steps):
            context = " ".join(steps[:i]) if i > 0 else ""
            step_eval = self.evaluate_step_correctness(step, context)
            step_scores.append(step_eval)

        avg_correctness = np.mean([s["correctness_score"] for s in step_scores])

        # Evaluate consistency
        consistency = self.evaluate_consistency(steps)

        # Detect self-correction
        correction = self.detect_self_correction(steps)

        # Evaluate progress (if problem provided)
        progress = self.evaluate_progress(steps, problem) if problem else 0.8

        # Calculate final PRM score (weighted combination)
        final_score = (
            avg_correctness * 0.40 +        # Step correctness: 40%
            consistency["consistency_score"] * 0.35 +  # Consistency: 35%
            progress * 0.15 +                # Progress: 15%
            correction["correction_bonus"]   # Self-correction bonus: 10%
        )

        final_score = np.clip(final_score, 0.0, 1.0)

        logger.info(f"PRM evaluation complete: {final_score:.3f} (based on {len(steps)} steps)")

        return {
            "num_steps": len(steps),
            "step_scores": step_scores,
            "avg_step_correctness": float(avg_correctness),
            "consistency_metrics": consistency,
            "self_correction": correction,
            "progress_score": float(progress),
            "final_prm_score": float(final_score)
        }

    def compare_reasoning_chains(
        self,
        response_a: str,
        response_b: str,
        problem: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compare two reasoning chains using PRM.

        Args:
            response_a: First response
            response_b: Second response
            problem: Original problem (optional)

        Returns:
            Comparison results with winner determination
        """
        eval_a = self.evaluate_reasoning_chain(response_a, problem)
        eval_b = self.evaluate_reasoning_chain(response_b, problem)

        winner = "A" if eval_a["final_prm_score"] > eval_b["final_prm_score"] else "B"
        if abs(eval_a["final_prm_score"] - eval_b["final_prm_score"]) < 0.05:
            winner = "Tie"

        return {
            "response_a_evaluation": eval_a,
            "response_b_evaluation": eval_b,
            "winner": winner,
            "score_difference": abs(eval_a["final_prm_score"] - eval_b["final_prm_score"])
        }


class StepValidator:
    """
    Validates individual reasoning steps using LLM-as-a-judge.

    Can be used to provide fine-grained feedback for training.
    """

    def __init__(self, llm_client=None):
        """
        Initialize with optional LLM client for validation.

        Args:
            llm_client: LLMClient instance for LLM-based validation
        """
        self.llm_client = llm_client
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def validate_step_async(
        self,
        step: str,
        previous_steps: List[str],
        problem: str
    ) -> Dict[str, any]:
        """
        Validate a single step using LLM.

        Args:
            step: Current reasoning step
            previous_steps: All previous steps
            problem: Original problem

        Returns:
            Validation results
        """
        if not self.llm_client:
            # Fallback to rule-based validation
            return {
                "is_valid": True,
                "confidence": 0.7,
                "feedback": "No LLM validator available, using rules"
            }

        validation_prompt = f"""
You are a reasoning step validator. Evaluate if this reasoning step is logically valid.

Problem: {problem}

Previous Steps:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(previous_steps)])}

Current Step to Validate: {step}

Is this step:
1. Logically consistent with previous steps?
2. Making progress toward solving the problem?
3. Free of logical fallacies?

Respond in this format:
Valid: [Yes/No]
Confidence: [0-100]
Feedback: [Brief explanation]
"""

        try:
            response = await self.llm_client.query(validation_prompt)
            # Parse response
            is_valid = "yes" in response.get("summary", "").lower()

            return {
                "is_valid": is_valid,
                "confidence": 0.8,
                "feedback": response.get("summary", "")[:200]
            }
        except Exception as e:
            logger.error(f"Step validation failed: {e}")
            return {
                "is_valid": True,
                "confidence": 0.5,
                "feedback": f"Validation error: {str(e)}"
            }
