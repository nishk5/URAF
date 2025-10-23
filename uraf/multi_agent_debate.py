"""
Multi-Agent Debate System

Implements multi-agent debate for improved reasoning through diverse perspectives.
Based on "Improving Factuality and Reasoning through Multiagent Debate" (Du et al., 2023)
"""

import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class AgentResponse:
    """Response from a single agent in debate."""
    agent_id: str
    round_number: int
    response: str
    confidence: float = 0.5
    reasoning: Optional[str] = None


class DebateAgent:
    """Individual agent participating in debate."""

    def __init__(
        self,
        agent_id: str,
        llm_client,
        perspective: Optional[str] = None
    ):
        """
        Initialize debate agent.

        Args:
            agent_id: Unique identifier
            llm_client: LLM client for this agent
            perspective: Agent's perspective/role (e.g., "optimistic", "critical")
        """
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.perspective = perspective or "neutral"
        self.responses: List[AgentResponse] = []
        logger.info(f"Initialized DebateAgent: {agent_id} ({self.perspective})")

    async def generate_initial_response(self, problem: str) -> AgentResponse:
        """
        Generate initial response to problem.

        Args:
            problem: Problem statement

        Returns:
            Agent's initial response
        """
        prompt = self._create_initial_prompt(problem)

        try:
            response = await self.llm_client.query(prompt)
            response_text = response.get("summary", "") if isinstance(response, dict) else str(response)

            agent_response = AgentResponse(
                agent_id=self.agent_id,
                round_number=1,
                response=response_text,
                confidence=0.6
            )

            self.responses.append(agent_response)
            logger.debug(f"Agent {self.agent_id} generated initial response")

            return agent_response

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate response: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                round_number=1,
                response=f"Error: {str(e)}",
                confidence=0.0
            )

    async def generate_critique(
        self,
        problem: str,
        other_responses: List[AgentResponse],
        round_number: int
    ) -> AgentResponse:
        """
        Generate critique of other agents' responses.

        Args:
            problem: Original problem
            other_responses: Responses from other agents
            round_number: Current debate round

        Returns:
            Agent's critique and revised response
        """
        prompt = self._create_critique_prompt(problem, other_responses, round_number)

        try:
            response = await self.llm_client.query(prompt)
            response_text = response.get("summary", "") if isinstance(response, dict) else str(response)

            agent_response = AgentResponse(
                agent_id=self.agent_id,
                round_number=round_number,
                response=response_text,
                confidence=0.7
            )

            self.responses.append(agent_response)
            logger.debug(f"Agent {self.agent_id} generated round {round_number} critique")

            return agent_response

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed critique: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                round_number=round_number,
                response=f"Error: {str(e)}",
                confidence=0.0
            )

    def _create_initial_prompt(self, problem: str) -> str:
        """Create prompt for initial response."""
        perspective_instruction = ""
        if self.perspective == "optimistic":
            perspective_instruction = "Approach this problem with an optimistic, solution-focused perspective."
        elif self.perspective == "critical":
            perspective_instruction = "Approach this problem with a critical, skeptical perspective. Identify potential issues."
        elif self.perspective == "creative":
            perspective_instruction = "Approach this problem with creative, unconventional thinking."

        return f"""You are participating in a multi-agent debate to solve a problem.

{perspective_instruction}

Problem: {problem}

Provide your solution with clear reasoning. Structure your response as:

*Analysis:* [Your analysis of the problem]
*Solution:* [Your proposed solution]
*Reasoning:* [Why your solution is correct]

Your response:"""

    def _create_critique_prompt(
        self,
        problem: str,
        other_responses: List[AgentResponse],
        round_number: int
    ) -> str:
        """Create prompt for critique round."""
        other_responses_text = "\n\n".join([
            f"Agent {r.agent_id}:\n{r.response}"
            for r in other_responses
            if r.agent_id != self.agent_id
        ])

        return f"""You are in round {round_number} of a multi-agent debate.

Problem: {problem}

Other agents' responses:
{other_responses_text}

Your previous response:
{self.responses[-1].response if self.responses else "None"}

Instructions:
1. Critique the other agents' responses - identify strengths and weaknesses
2. Refine your own solution based on the discussion
3. Provide your revised answer

Format your response as:
*Critique:* [Analysis of other responses]
*Revised Solution:* [Your improved solution]
*Reasoning:* [Why your solution is better]

Your response:"""


class MediatorAgent:
    """Mediator that synthesizes debate into final answer."""

    def __init__(self, llm_client):
        """
        Initialize mediator.

        Args:
            llm_client: LLM client for mediation
        """
        self.llm_client = llm_client
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Initialized MediatorAgent")

    async def synthesize(
        self,
        problem: str,
        all_responses: List[List[AgentResponse]]
    ) -> Dict[str, Any]:
        """
        Synthesize debate into final answer.

        Args:
            problem: Original problem
            all_responses: All rounds of responses from all agents

        Returns:
            Synthesized answer with reasoning
        """
        # Flatten all responses
        flat_responses = [r for round_responses in all_responses for r in round_responses]

        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(problem, flat_responses)

        try:
            response = await self.llm_client.query(prompt)
            response_text = response.get("summary", "") if isinstance(response, dict) else str(response)

            # Calculate consensus score
            consensus_score = self._calculate_consensus(flat_responses)

            logger.info(f"Mediator synthesized final answer (consensus: {consensus_score:.2f})")

            return {
                "final_answer": response_text,
                "consensus_score": consensus_score,
                "num_agents": len(set(r.agent_id for r in flat_responses)),
                "num_rounds": len(all_responses)
            }

        except Exception as e:
            logger.error(f"Mediation failed: {e}")
            return {
                "final_answer": f"Mediation error: {str(e)}",
                "consensus_score": 0.0,
                "error": str(e)
            }

    def _create_synthesis_prompt(
        self,
        problem: str,
        responses: List[AgentResponse]
    ) -> str:
        """Create prompt for synthesis."""
        # Group by agent
        by_agent = {}
        for r in responses:
            if r.agent_id not in by_agent:
                by_agent[r.agent_id] = []
            by_agent[r.agent_id].append(r)

        responses_text = ""
        for agent_id, agent_responses in by_agent.items():
            responses_text += f"\nAgent {agent_id}:\n"
            for r in agent_responses:
                responses_text += f"  Round {r.round_number}: {r.response[:200]}...\n"

        return f"""You are a mediator synthesizing a multi-agent debate into a final answer.

Problem: {problem}

Debate Transcript:
{responses_text}

Instructions:
1. Identify points of agreement among agents
2. Evaluate the strength of each argument
3. Synthesize the best elements into a comprehensive answer
4. Acknowledge any remaining uncertainties

Provide your synthesis in this format:
*Key Agreements:* [Points where agents converged]
*Best Arguments:* [Strongest reasoning identified]
*Final Answer:* [Your synthesized solution]
*Confidence:* [High/Medium/Low and why]

Your synthesis:"""

    def _calculate_consensus(self, responses: List[AgentResponse]) -> float:
        """
        Calculate consensus score among responses.

        Uses semantic similarity between final round responses.

        Args:
            responses: All agent responses

        Returns:
            Consensus score (0-1)
        """
        if not responses:
            return 0.0

        # Get final round responses
        max_round = max(r.round_number for r in responses)
        final_responses = [r for r in responses if r.round_number == max_round]

        if len(final_responses) < 2:
            return 1.0  # Trivial consensus with single agent

        # Compute embeddings
        texts = [r.response for r in final_responses]
        embeddings = self.embedding_model.encode(texts)

        # Pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0


class MultiAgentDebate:
    """
    Multi-agent debate system.

    Orchestrates debate between multiple agents to reach better solutions.
    """

    def __init__(
        self,
        llm_clients: List,
        num_rounds: int = 3,
        perspectives: Optional[List[str]] = None
    ):
        """
        Initialize debate system.

        Args:
            llm_clients: List of LLM clients for agents
            num_rounds: Number of debate rounds
            perspectives: List of perspectives for agents
        """
        self.num_rounds = num_rounds

        # Create agents
        if perspectives is None:
            perspectives = ["optimistic", "critical", "creative"] * (len(llm_clients) // 3 + 1)

        self.agents = [
            DebateAgent(
                agent_id=f"agent_{i}",
                llm_client=client,
                perspective=perspectives[i] if i < len(perspectives) else "neutral"
            )
            for i, client in enumerate(llm_clients)
        ]

        self.mediator = MediatorAgent(llm_clients[0])
        self.debate_history: List[List[AgentResponse]] = []

        logger.info(f"Initialized MultiAgentDebate with {len(self.agents)} agents, {num_rounds} rounds")

    async def debate(self, problem: str) -> Dict[str, Any]:
        """
        Run multi-agent debate.

        Args:
            problem: Problem to solve

        Returns:
            Debate results with final answer
        """
        logger.info(f"Starting debate on: {problem[:100]}...")

        self.debate_history = []

        # Round 1: Initial responses
        logger.info("Round 1: Initial responses")
        initial_responses = await asyncio.gather(*[
            agent.generate_initial_response(problem)
            for agent in self.agents
        ])
        self.debate_history.append(initial_responses)

        # Subsequent rounds: Critique and refine
        for round_num in range(2, self.num_rounds + 1):
            logger.info(f"Round {round_num}: Critique and refine")

            # Each agent critiques others and refines their answer
            round_responses = await asyncio.gather(*[
                agent.generate_critique(problem, self.debate_history[-1], round_num)
                for agent in self.agents
            ])
            self.debate_history.append(round_responses)

        # Mediation: Synthesize final answer
        logger.info("Mediation: Synthesizing final answer")
        synthesis = await self.mediator.synthesize(problem, self.debate_history)

        return {
            "problem": problem,
            "num_agents": len(self.agents),
            "num_rounds": self.num_rounds,
            "debate_history": self.debate_history,
            "synthesis": synthesis,
            "final_answer": synthesis.get("final_answer", ""),
            "consensus_score": synthesis.get("consensus_score", 0.0)
        }

    def get_debate_summary(self) -> str:
        """
        Get human-readable summary of debate.

        Returns:
            Summary text
        """
        if not self.debate_history:
            return "No debate history"

        summary = f"Multi-Agent Debate Summary\n"
        summary += f"Agents: {len(self.agents)}, Rounds: {len(self.debate_history)}\n\n"

        for round_num, round_responses in enumerate(self.debate_history, 1):
            summary += f"Round {round_num}:\n"
            for response in round_responses:
                summary += f"  {response.agent_id}: {response.response[:100]}...\n"
            summary += "\n"

        return summary


class DebateEvaluator:
    """Evaluates quality of multi-agent debate."""

    def __init__(self):
        """Initialize debate evaluator."""
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate_debate(self, debate_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate debate quality.

        Args:
            debate_result: Result from MultiAgentDebate.debate()

        Returns:
            Evaluation metrics
        """
        history = debate_result.get("debate_history", [])

        if not history:
            return {
                "consensus_convergence": 0.0,
                "argument_diversity": 0.0,
                "refinement_quality": 0.0,
                "overall_debate_score": 0.0
            }

        # Consensus convergence: did agents converge over rounds?
        convergence = self._measure_convergence(history)

        # Argument diversity: how diverse were initial perspectives?
        diversity = self._measure_diversity(history[0] if history else [])

        # Refinement quality: did responses improve?
        refinement = self._measure_refinement(history)

        overall_score = (
            convergence * 0.4 +
            diversity * 0.3 +
            refinement * 0.3
        )

        return {
            "consensus_convergence": float(convergence),
            "argument_diversity": float(diversity),
            "refinement_quality": float(refinement),
            "overall_debate_score": float(overall_score),
            "consensus_score": debate_result.get("consensus_score", 0.0)
        }

    def _measure_convergence(self, history: List[List[AgentResponse]]) -> float:
        """Measure if agents converged over rounds."""
        if len(history) < 2:
            return 0.5

        # Compare first and last round similarity
        first_round = history[0]
        last_round = history[-1]

        first_embeddings = self.embedding_model.encode([r.response for r in first_round])
        last_embeddings = self.embedding_model.encode([r.response for r in last_round])

        # Average pairwise similarity
        first_sim = self._avg_pairwise_similarity(first_embeddings)
        last_sim = self._avg_pairwise_similarity(last_embeddings)

        # Convergence = increase in similarity
        convergence = max(0, last_sim - first_sim) / (1 - first_sim + 0.01)
        return float(np.clip(convergence, 0.0, 1.0))

    def _measure_diversity(self, responses: List[AgentResponse]) -> float:
        """Measure diversity of perspectives."""
        if len(responses) < 2:
            return 0.5

        embeddings = self.embedding_model.encode([r.response for r in responses])
        avg_sim = self._avg_pairwise_similarity(embeddings)

        # Diversity is inverse of similarity
        diversity = 1 - avg_sim
        return float(np.clip(diversity, 0.0, 1.0))

    def _measure_refinement(self, history: List[List[AgentResponse]]) -> float:
        """Measure if responses got more sophisticated."""
        if len(history) < 2:
            return 0.5

        # Simple heuristic: longer responses = more detailed (with limit)
        first_avg_length = np.mean([len(r.response) for r in history[0]])
        last_avg_length = np.mean([len(r.response) for r in history[-1]])

        # Improvement if final responses are 10-50% longer
        length_ratio = last_avg_length / (first_avg_length + 1)
        refinement = np.clip((length_ratio - 1) / 0.5, 0, 1)

        return float(refinement)

    def _avg_pairwise_similarity(self, embeddings: np.ndarray) -> float:
        """Calculate average pairwise cosine similarity."""
        if len(embeddings) < 2:
            return 1.0

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)

        return float(np.mean(similarities))
