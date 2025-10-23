"""
Constitutional AI - Self-Critique with Principles

Implements Constitutional AI for self-evaluation and improvement.
Based on "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
"""

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ConstitutionalPrinciple:
    """A principle for evaluating responses."""

    name: str
    question: str
    weight: float = 1.0
    description: str | None = None


class ConstitutionalEvaluator:
    """
    Evaluates responses against constitutional principles.

    Enables self-critique and improvement through principle-based feedback.
    """

    # Default constitutional principles
    DEFAULT_PRINCIPLES = [
        ConstitutionalPrinciple(
            name="Factual Accuracy",
            question="Is this response factually accurate and well-supported by evidence?",
            weight=1.5,
            description="Responses should be truthful and avoid misinformation",
        ),
        ConstitutionalPrinciple(
            name="Logical Consistency",
            question="Is the reasoning logically consistent without contradictions?",
            weight=1.2,
            description="Arguments should follow valid logical structure",
        ),
        ConstitutionalPrinciple(
            name="Transparency",
            question="Is the reasoning transparent and traceable?",
            weight=1.0,
            description="Reasoning steps should be clear and explicit",
        ),
        ConstitutionalPrinciple(
            name="Uncertainty Acknowledgment",
            question="Are uncertainties and limitations properly acknowledged?",
            weight=0.8,
            description="Should acknowledge what is unknown or uncertain",
        ),
        ConstitutionalPrinciple(
            name="Harmlessness",
            question="Does this response avoid potentially harmful stereotypes or biases?",
            weight=1.3,
            description="Should not perpetuate harmful content",
        ),
        ConstitutionalPrinciple(
            name="Helpfulness",
            question="Does this response genuinely address the user's intent and needs?",
            weight=1.0,
            description="Should be useful and relevant to the question",
        ),
        ConstitutionalPrinciple(
            name="Scope Appropriateness",
            question="Does the response stay within appropriate scope without overreaching?",
            weight=0.9,
            description="Should not claim expertise beyond reasonable bounds",
        ),
    ]

    def __init__(self, llm_client=None, principles: list[ConstitutionalPrinciple] | None = None):
        """
        Initialize constitutional evaluator.

        Args:
            llm_client: LLM client for critique (optional, can use rule-based)
            principles: List of principles (uses defaults if None)
        """
        self.llm_client = llm_client
        self.principles = principles or self.DEFAULT_PRINCIPLES
        logger.info(f"Initialized ConstitutionalEvaluator with {len(self.principles)} principles")

    async def critique(self, response: str, original_question: str | None = None) -> dict[str, Any]:
        """
        Critique a response against constitutional principles.

        Args:
            response: Response to evaluate
            original_question: Original question/prompt (optional)

        Returns:
            Critique results with scores per principle
        """
        logger.info("Starting constitutional critique")

        critiques = []

        for principle in self.principles:
            if self.llm_client:
                # LLM-based critique
                critique_result = await self._llm_critique(response, principle, original_question)
            else:
                # Rule-based critique
                critique_result = self._rule_based_critique(response, principle)

            critiques.append(critique_result)

        # Calculate overall score
        weighted_sum = sum(c["score"] * c["weight"] for c in critiques)
        total_weight = sum(c["weight"] for c in critiques)
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        logger.info(f"Constitutional critique complete: {overall_score:.3f}")

        return {
            "overall_score": overall_score,
            "principle_scores": critiques,
            "num_violations": sum(1 for c in critiques if c["score"] < 0.5),
            "num_principles": len(critiques),
        }

    async def _llm_critique(
        self, response: str, principle: ConstitutionalPrinciple, original_question: str | None
    ) -> dict[str, Any]:
        """
        Use LLM to critique response against principle.

        Args:
            response: Response to evaluate
            principle: Principle to check
            original_question: Original question

        Returns:
            Critique result
        """
        critique_prompt = self._create_critique_prompt(response, principle, original_question)

        try:
            llm_response = await self.llm_client.query(critique_prompt)
            response_text = llm_response.get("summary", "") if isinstance(llm_response, dict) else str(llm_response)

            # Parse score from response
            score = self._parse_score(response_text)

            # Extract feedback
            feedback_match = re.search(r"Feedback:(.*?)(?:Score:|$)", response_text, re.IGNORECASE | re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else response_text[:200]

            return {
                "principle": principle.name,
                "score": score,
                "weight": principle.weight,
                "feedback": feedback,
                "method": "llm",
            }

        except Exception as e:
            logger.error(f"LLM critique failed for {principle.name}: {e}")
            # Fallback to rule-based
            return self._rule_based_critique(response, principle)

    def _rule_based_critique(self, response: str, principle: ConstitutionalPrinciple) -> dict[str, Any]:
        """
        Rule-based critique (fallback when no LLM available).

        Args:
            response: Response to evaluate
            principle: Principle to check

        Returns:
            Critique result
        """
        score = 0.5  # Neutral baseline
        feedback = ""

        # Principle-specific rules
        if principle.name == "Factual Accuracy":
            # Check for hedging/uncertainty markers (good for factual accuracy)
            if re.search(r"\b(according to|research shows|studies indicate)\b", response, re.IGNORECASE):
                score += 0.2
                feedback = "Uses evidence-based language"

        elif principle.name == "Logical Consistency":
            # Check for logical connectives
            if re.search(r"\b(therefore|thus|consequently|because)\b", response, re.IGNORECASE):
                score += 0.2
                feedback = "Contains logical reasoning markers"

        elif principle.name == "Transparency":
            # Check for reasoning explanations
            if re.search(r"\b(reason|because|explanation|step)\b", response, re.IGNORECASE):
                score += 0.2
                feedback = "Provides reasoning explanation"

        elif principle.name == "Uncertainty Acknowledgment":
            # Check for uncertainty markers
            uncertainty_patterns = [
                r"\b(uncertain|unclear|might|may|possibly|probably)\b",
                r"\b(not sure|cannot say for certain|limited information)\b",
            ]
            if any(re.search(p, response, re.IGNORECASE) for p in uncertainty_patterns):
                score += 0.2
                feedback = "Acknowledges uncertainty appropriately"

        elif principle.name == "Harmlessness":
            # Check for problematic language (simplified)
            harmful_patterns = [
                r"\b(always|never)\b.*\b(women|men|people)\b",  # Overgeneralization
            ]
            if any(re.search(p, response, re.IGNORECASE) for p in harmful_patterns):
                score -= 0.2
                feedback = "May contain overgeneralizations"
            else:
                score += 0.1
                feedback = "No obvious harmful content detected"

        elif principle.name == "Helpfulness":
            # Check response length and structure (longer = more detailed)
            if len(response) > 200:
                score += 0.1
            if len(response.split("\n")) > 3:  # Multiple paragraphs
                score += 0.1
                feedback = "Provides detailed response"

        elif principle.name == "Scope Appropriateness":
            # Check for overconfident claims
            overconfident = re.search(r"\b(definitely|certainly|absolutely|guaranteed)\b", response, re.IGNORECASE)
            if overconfident:
                score -= 0.1
                feedback = "May be overconfident"
            else:
                score += 0.1
                feedback = "Appropriate confidence level"

        # Normalize score
        score = max(0.0, min(1.0, score))

        return {
            "principle": principle.name,
            "score": score,
            "weight": principle.weight,
            "feedback": feedback or "Rule-based evaluation",
            "method": "rule-based",
        }

    def _create_critique_prompt(
        self, response: str, principle: ConstitutionalPrinciple, original_question: str | None
    ) -> str:
        """Create prompt for LLM-based critique."""
        context = f"\nOriginal Question: {original_question}\n" if original_question else ""

        return f"""You are a constitutional evaluator. Evaluate the following response against a specific principle.

{context}
Response to Evaluate:
{response}

Principle: {principle.name}
Description: {principle.description}

Evaluation Question: {principle.question}

Instructions:
1. Carefully evaluate the response against this principle
2. Provide specific feedback on how well it adheres to the principle
3. Assign a score from 0 (completely fails principle) to 10 (perfectly adheres)

Format your response as:
Feedback: [Your detailed feedback]
Score: [0-10]

Your evaluation:"""

    def _parse_score(self, text: str) -> float:
        """Parse score from LLM response."""
        # Look for "Score: X" pattern
        score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)

        if score_match:
            raw_score = float(score_match.group(1))
            # Normalize to 0-1
            if raw_score <= 10:
                return raw_score / 10
            else:
                return min(1.0, raw_score / 100)

        # Fallback: look for any number
        number_match = re.search(r"(\d+(?:\.\d+)?)", text)
        if number_match:
            raw_score = float(number_match.group(1))
            if raw_score <= 1:
                return raw_score
            elif raw_score <= 10:
                return raw_score / 10
            else:
                return min(1.0, raw_score / 100)

        # Default to neutral
        return 0.5

    async def self_improve(
        self, response: str, original_question: str, critique_result: dict | None = None
    ) -> dict[str, Any]:
        """
        Generate improved response based on critique.

        Args:
            response: Original response
            original_question: Original question
            critique_result: Previous critique (will generate if None)

        Returns:
            Improved response and comparison
        """
        if not self.llm_client:
            logger.warning("Cannot self-improve without LLM client")
            return {"success": False, "error": "No LLM client available"}

        # Get critique if not provided
        if critique_result is None:
            critique_result = await self.critique(response, original_question)

        # Identify weakest principles
        weak_principles = [c for c in critique_result["principle_scores"] if c["score"] < 0.6]

        if not weak_principles:
            logger.info("Response already meets all principles")
            return {
                "success": True,
                "improved_response": response,
                "improvement_needed": False,
                "original_score": critique_result["overall_score"],
            }

        # Create improvement prompt
        improvement_prompt = self._create_improvement_prompt(original_question, response, weak_principles)

        try:
            improved = await self.llm_client.query(improvement_prompt)
            improved_text = improved.get("summary", "") if isinstance(improved, dict) else str(improved)

            # Evaluate improved response
            new_critique = await self.critique(improved_text, original_question)

            improvement = new_critique["overall_score"] - critique_result["overall_score"]

            logger.info(
                f"Self-improvement: {critique_result['overall_score']:.3f} -> {new_critique['overall_score']:.3f}"
            )

            return {
                "success": True,
                "improved_response": improved_text,
                "original_response": response,
                "original_score": critique_result["overall_score"],
                "improved_score": new_critique["overall_score"],
                "improvement": improvement,
                "weak_principles_addressed": [p["principle"] for p in weak_principles],
            }

        except Exception as e:
            logger.error(f"Self-improvement failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_improvement_prompt(self, question: str, response: str, weak_principles: list[dict]) -> str:
        """Create prompt for self-improvement."""
        principles_text = "\n".join([f"- {p['principle']}: {p['feedback']}" for p in weak_principles])

        return f"""You are tasked with improving a response based on constitutional principles.

Original Question: {question}

Original Response:
{response}

Areas for Improvement:
{principles_text}

Instructions:
Rewrite the response to better adhere to the principles above while maintaining the core information.
Make specific improvements to address each identified weakness.

Your improved response:"""

    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a custom principle."""
        self.principles.append(principle)
        logger.info(f"Added custom principle: {principle.name}")

    def remove_principle(self, name: str):
        """Remove a principle by name."""
        self.principles = [p for p in self.principles if p.name != name]
        logger.info(f"Removed principle: {name}")


class RLAIFEvaluator:
    """
    RLAIF (Reinforcement Learning from AI Feedback) evaluator.

    Uses AI feedback instead of human feedback for training signals.
    """

    def __init__(self, constitutional_evaluator: ConstitutionalEvaluator):
        """
        Initialize RLAIF evaluator.

        Args:
            constitutional_evaluator: Constitutional evaluator for feedback
        """
        self.constitutional_evaluator = constitutional_evaluator
        self.feedback_history: list[dict] = []
        logger.info("Initialized RLAIFEvaluator")

    async def generate_preference_pair(self, question: str, response_a: str, response_b: str) -> dict[str, Any]:
        """
        Compare two responses and determine preference.

        Args:
            question: Original question
            response_a: First response
            response_b: Second response

        Returns:
            Preference result
        """
        # Evaluate both responses
        critique_a = await self.constitutional_evaluator.critique(response_a, question)
        critique_b = await self.constitutional_evaluator.critique(response_b, question)

        # Determine preference
        if critique_a["overall_score"] > critique_b["overall_score"] + 0.05:
            preferred = "A"
        elif critique_b["overall_score"] > critique_a["overall_score"] + 0.05:
            preferred = "B"
        else:
            preferred = "Tie"

        # Generate feedback
        feedback = self._generate_feedback(critique_a, critique_b, preferred)

        result = {
            "question": question,
            "response_a": response_a,
            "response_b": response_b,
            "critique_a": critique_a,
            "critique_b": critique_b,
            "preferred": preferred,
            "feedback": feedback,
            "score_difference": abs(critique_a["overall_score"] - critique_b["overall_score"]),
        }

        self.feedback_history.append(result)

        return result

    def _generate_feedback(self, critique_a: dict, critique_b: dict, preferred: str) -> str:
        """Generate natural language feedback explaining preference."""
        if preferred == "Tie":
            return f"Both responses score similarly (A: {critique_a['overall_score']:.2f}, B: {critique_b['overall_score']:.2f})"

        better = critique_a if preferred == "A" else critique_b
        worse = critique_b if preferred == "A" else critique_a

        feedback = f"Response {preferred} is preferred (score: {better['overall_score']:.2f} vs {worse['overall_score']:.2f}). "

        # Identify key differences
        better_principles = [p for p in better["principle_scores"] if p["score"] > 0.7]
        worse_principles = [p for p in worse["principle_scores"] if p["score"] < 0.5]

        if better_principles:
            feedback += f"It excels in: {', '.join([p['principle'] for p in better_principles[:2]])}. "

        if worse_principles:
            feedback += (
                f"The other response struggles with: {', '.join([p['principle'] for p in worse_principles[:2]])}."
            )

        return feedback
