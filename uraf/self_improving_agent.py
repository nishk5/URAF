"""
Self-Improving Agents - STaR (Self-Taught Reasoner) Implementation

Based on "STaR: Bootstrapping Reasoning With Reasoning" (Zelikman et al., 2022)
and extended with Constitutional AI and Process Reward Models.

This module enables agents to recursively improve their reasoning through:
1. Generate reasoning chains for problems
2. Evaluate quality using PRM + Constitutional principles
3. Filter high-quality reasoning paths
4. Learn from successful patterns
5. Iteratively improve
"""

import asyncio
from typing import Any

from loguru import logger

from uraf.constitutional_ai import ConstitutionalEvaluator
from uraf.llm_client import LLMClient
from uraf.process_reward_model import ProcessRewardModel


class STaRAgent:
    """Self-Taught Reasoner - Agent that improves through self-generated reasoning."""

    def __init__(
        self,
        llm_client: LLMClient,
        prm: ProcessRewardModel | None = None,
        constitutional: ConstitutionalEvaluator | None = None,
        quality_threshold: float = 0.7,
        max_iterations: int = 5,
    ):
        """
        Initialize STaR agent.

        Args:
            llm_client: LLM client for generation
            prm: Process reward model for step-by-step evaluation
            constitutional: Constitutional evaluator for principle-based critique
            quality_threshold: Minimum quality score to accept reasoning
            max_iterations: Maximum self-improvement iterations
        """
        self.llm = llm_client
        self.prm = prm or ProcessRewardModel()
        self.constitutional = constitutional or ConstitutionalEvaluator(llm_client=llm_client)
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations

        # Learning memory - stores high-quality reasoning patterns
        self.reasoning_library: list[dict[str, Any]] = []

        logger.info("Initialized STaRAgent for self-improvement")

    async def solve_with_improvement(
        self, problem: str, ground_truth: str | None = None, verbose: bool = False
    ) -> dict[str, Any]:
        """
        Solve problem with iterative self-improvement.

        Args:
            problem: Problem to solve
            ground_truth: Optional ground truth answer for validation
            verbose: Print improvement progress

        Returns:
            Best solution after self-improvement
        """
        logger.info(f"STaR solving with self-improvement: {problem[:100]}...")

        best_solution = None
        best_quality = 0.0
        improvement_history = []

        for iteration in range(self.max_iterations):
            if verbose:
                logger.info(f"STaR Iteration {iteration + 1}/{self.max_iterations}")

            # Generate candidate solution
            solution = await self._generate_reasoning(problem, iteration)

            # Evaluate quality
            quality_metrics = await self._evaluate_quality(solution["reasoning"], problem)

            # Check if correct (if ground truth provided)
            is_correct = True
            if ground_truth:
                is_correct = self._check_correctness(solution["answer"], ground_truth)

            overall_quality = quality_metrics["overall_quality"]

            if verbose:
                logger.info(f"  Quality: {overall_quality:.3f}, Correct: {is_correct}")

            # Store if high quality and correct
            if is_correct and overall_quality >= self.quality_threshold:
                self._add_to_library(problem, solution["reasoning"], quality_metrics)

                if overall_quality > best_quality:
                    best_solution = solution
                    best_quality = overall_quality

            improvement_history.append(
                {
                    "iteration": iteration + 1,
                    "quality": overall_quality,
                    "correct": is_correct,
                    "prm_score": quality_metrics["prm_score"],
                    "constitutional_score": quality_metrics["constitutional_score"],
                }
            )

            # Early stopping if quality is excellent
            if overall_quality >= 0.9:
                logger.info(f"STaR reached excellent quality ({overall_quality:.3f}), stopping early")
                break

        result = {
            "best_solution": best_solution or solution,
            "best_quality": best_quality if best_solution else overall_quality,
            "improvement_history": improvement_history,
            "final_iteration": len(improvement_history),
            "library_size": len(self.reasoning_library),
        }

        return result

    async def _generate_reasoning(self, problem: str, iteration: int) -> dict[str, str]:
        """Generate reasoning chain for problem."""
        # Retrieve similar examples from library
        examples = self._retrieve_similar_examples(problem, k=3)

        # Create prompt with examples
        prompt = self._create_star_prompt(problem, examples, iteration)

        # Generate reasoning
        response = await self.llm.query(prompt)

        # Parse answer and reasoning
        answer, reasoning = self._parse_response(response)

        return {"answer": answer, "reasoning": reasoning, "full_response": response}

    def _create_star_prompt(self, problem: str, examples: list[dict], iteration: int) -> str:
        """Create prompt with few-shot examples from reasoning library."""
        prompt_parts = [
            "You are a reasoning agent that thinks step-by-step to solve problems.",
            "",
            "Follow this format:",
            "*Understanding:* Restate the problem clearly.",
            "*Reasoning Pathway:* Think through the solution step-by-step.",
            "*Final Answer:* Provide the final answer.",
            "",
        ]

        # Add examples from library
        if examples:
            prompt_parts.append("Here are examples of high-quality reasoning:\n")
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Problem: {ex['problem']}")
                prompt_parts.append(f"Reasoning: {ex['reasoning'][:300]}...")
                prompt_parts.append("")

        # Add the actual problem
        prompt_parts.extend(
            [
                f"Now solve this problem (Attempt {iteration + 1}):",
                f"Problem: {problem}",
                "",
                "Your solution:",
            ]
        )

        return "\n".join(prompt_parts)

    async def _evaluate_quality(self, reasoning: str, problem: str) -> dict[str, Any]:
        """Evaluate reasoning quality using PRM and Constitutional AI."""
        # PRM evaluation
        prm_result = self.prm.evaluate_reasoning_chain(reasoning, problem)
        prm_score = prm_result["final_prm_score"]

        # Constitutional evaluation
        constitutional_result = await self.constitutional.critique(reasoning, problem)
        constitutional_score = constitutional_result["overall_score"]

        # Combined quality score
        overall_quality = (prm_score * 0.6) + (constitutional_score * 0.4)

        return {
            "prm_score": prm_score,
            "constitutional_score": constitutional_score,
            "overall_quality": overall_quality,
            "prm_details": prm_result,
            "constitutional_details": constitutional_result,
        }

    def _check_correctness(self, predicted_answer: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        # Normalize answers for comparison
        pred_normalized = predicted_answer.strip().lower()
        truth_normalized = ground_truth.strip().lower()

        # Exact match
        if pred_normalized == truth_normalized:
            return True

        # Substring match (for numeric answers, etc.)
        return truth_normalized in pred_normalized or pred_normalized in truth_normalized

    def _add_to_library(self, problem: str, reasoning: str, quality_metrics: dict):
        """Add high-quality reasoning to library."""
        entry = {"problem": problem, "reasoning": reasoning, "quality": quality_metrics["overall_quality"]}

        self.reasoning_library.append(entry)

        # Keep library size manageable (top N examples)
        if len(self.reasoning_library) > 100:
            self.reasoning_library.sort(key=lambda x: x["quality"], reverse=True)
            self.reasoning_library = self.reasoning_library[:100]

        logger.debug(f"Added reasoning to library (size: {len(self.reasoning_library)})")

    def _retrieve_similar_examples(self, problem: str, k: int = 3) -> list[dict]:
        """Retrieve similar examples from reasoning library."""
        if not self.reasoning_library:
            return []

        # Simple keyword-based similarity for now
        # In production, use embeddings + vector search
        problem_words = set(problem.lower().split())

        scored_examples = []
        for example in self.reasoning_library:
            example_words = set(example["problem"].lower().split())
            overlap = len(problem_words & example_words)
            scored_examples.append((overlap, example))

        # Sort by similarity
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        # Return top k
        return [ex for _, ex in scored_examples[:k]]

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse answer and reasoning from response."""
        # Extract final answer
        if "*Final Answer:*" in response:
            parts = response.split("*Final Answer:*")
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        elif "Final Answer:" in response:
            parts = response.split("Final Answer:")
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            # No explicit final answer marker
            reasoning = response
            # Try to extract last sentence as answer
            sentences = response.strip().split(".")
            answer = sentences[-1].strip() if sentences else ""

        return answer, reasoning


class RecursiveSelfImprovement:
    """Recursive self-improvement through critique and refinement."""

    def __init__(
        self, llm_client: LLMClient, constitutional: ConstitutionalEvaluator | None = None, max_rounds: int = 3
    ):
        """
        Initialize recursive self-improvement.

        Args:
            llm_client: LLM client
            constitutional: Constitutional evaluator for critique
            max_rounds: Maximum improvement rounds
        """
        self.llm = llm_client
        self.constitutional = constitutional or ConstitutionalEvaluator(llm_client=llm_client)
        self.max_rounds = max_rounds
        logger.info("Initialized RecursiveSelfImprovement")

    async def improve_response(
        self, initial_response: str, question: str, improvement_threshold: float = 0.8
    ) -> dict[str, Any]:
        """
        Recursively improve response through self-critique.

        Args:
            initial_response: Initial response to improve
            question: Original question
            improvement_threshold: Stop when score exceeds this

        Returns:
            Improved response and improvement history
        """
        logger.info("Starting recursive self-improvement")

        current_response = initial_response
        improvement_history = []

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Improvement round {round_num}/{self.max_rounds}")

            # Critique current response
            critique = await self.constitutional.critique(current_response, question)

            score = critique["overall_score"]
            improvement_history.append(
                {"round": round_num, "score": score, "response": current_response, "critique": critique}
            )

            # Check if good enough
            if score >= improvement_threshold:
                logger.info(f"Reached improvement threshold ({score:.3f} >= {improvement_threshold})")
                break

            # Identify weak areas
            weak_principles = [p for p in critique["principle_scores"] if p["score"] < 0.6]

            if not weak_principles and round_num < self.max_rounds:
                # No clear weaknesses, but can still improve
                weak_principles = sorted(critique["principle_scores"], key=lambda x: x["score"])[:2]

            # Generate improved response
            improved_response = await self._generate_improvement(current_response, question, weak_principles)

            current_response = improved_response

        return {
            "final_response": current_response,
            "improvement_history": improvement_history,
            "final_score": improvement_history[-1]["score"],
            "num_rounds": len(improvement_history),
            "improved": improvement_history[-1]["score"] > improvement_history[0]["score"],
        }

    async def _generate_improvement(self, current_response: str, question: str, weak_principles: list[dict]) -> str:
        """Generate improved response addressing weak principles."""
        critique_text = "\n".join(
            [f"- {p['principle']}: {p['feedback']} (score: {p['score']:.2f})" for p in weak_principles]
        )

        improvement_prompt = f"""You previously answered this question:

Question: {question}

Your previous answer:
{current_response}

Areas to improve:
{critique_text}

Please provide an improved answer that addresses these critiques while maintaining your correct insights.

Improved answer:"""

        improved = await self.llm.query(improvement_prompt)

        return improved.strip()


class MultiAgentSTaR:
    """Multi-agent STaR where agents learn from each other's successful reasoning."""

    def __init__(self, num_agents: int = 3, llm_client: LLMClient | None = None, quality_threshold: float = 0.7):
        """
        Initialize multi-agent STaR.

        Args:
            num_agents: Number of parallel agents
            llm_client: LLM client
            quality_threshold: Quality threshold for learning
        """
        self.num_agents = num_agents
        self.llm = llm_client or LLMClient()
        self.quality_threshold = quality_threshold

        # Shared reasoning library
        self.shared_library: list[dict] = []

        # Create agents
        self.agents = [
            STaRAgent(
                llm_client=self.llm,
                quality_threshold=quality_threshold,
                max_iterations=3,  # Fewer iterations per agent
            )
            for _ in range(num_agents)
        ]

        logger.info(f"Initialized MultiAgentSTaR with {num_agents} agents")

    async def collaborative_solve(self, problem: str, ground_truth: str | None = None) -> dict[str, Any]:
        """
        Agents collaborate to solve problem, learning from each other.

        Args:
            problem: Problem to solve
            ground_truth: Optional ground truth

        Returns:
            Best solution across all agents
        """
        logger.info(f"Multi-agent STaR collaborative solving: {problem[:100]}...")

        # All agents solve in parallel
        agent_results = await asyncio.gather(
            *[agent.solve_with_improvement(problem, ground_truth, verbose=False) for agent in self.agents]
        )

        # Find best solution
        best_result = max(agent_results, key=lambda x: x["best_quality"])

        # Share high-quality reasoning across agents
        for i, result in enumerate(agent_results):
            if result["best_quality"] >= self.quality_threshold:
                # Add to shared library
                best_solution = result["best_solution"]
                self.shared_library.append(
                    {
                        "agent_id": i,
                        "problem": problem,
                        "reasoning": best_solution["reasoning"],
                        "quality": result["best_quality"],
                    }
                )

        # Update all agents with shared library
        for agent in self.agents:
            agent.reasoning_library.extend(self.shared_library)

        return {
            "best_solution": best_result["best_solution"],
            "best_quality": best_result["best_quality"],
            "agent_results": agent_results,
            "shared_library_size": len(self.shared_library),
        }


def calculate_improvement_rate(improvement_history: list[dict]) -> dict[str, float]:
    """
    Calculate improvement rate from history.

    Args:
        improvement_history: List of iteration results with quality scores

    Returns:
        Improvement metrics
    """
    if len(improvement_history) < 2:
        return {"improvement_rate": 0.0, "total_improvement": 0.0, "converged": False}

    scores = [h["quality"] for h in improvement_history]

    initial_score = scores[0]
    final_score = scores[-1]
    total_improvement = final_score - initial_score

    # Calculate average improvement per iteration
    improvements = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    avg_improvement_rate = sum(improvements) / len(improvements)

    # Check convergence (improvement < 0.01 in last 2 iterations)
    converged = all(abs(imp) < 0.01 for imp in improvements[-2:]) if len(improvements) >= 2 else False

    return {
        "improvement_rate": avg_improvement_rate,
        "total_improvement": total_improvement,
        "initial_score": initial_score,
        "final_score": final_score,
        "converged": converged,
    }
