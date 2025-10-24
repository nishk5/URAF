"""
Tests for advanced AI features (2024-2025).

Tests for:
- Self-Improving Agents (STaR)
- Uncertainty Quantification
- Tree-of-Thoughts reasoning
- Advanced RAG 2.0
- Causal Reasoning
"""

import pytest

from uraf.causal_reasoning import CausalReasoner, CausalDiscovery, detect_causal_language
from uraf.llm_client import LLMClient
from uraf.rag_system import AdvancedRAG, RetrievalMode, QueryTransformStrategy, SelfRAG
from uraf.self_improving_agent import STaRAgent, RecursiveSelfImprovement, calculate_improvement_rate
from uraf.tree_of_thoughts import TreeOfThoughts, SearchStrategy, GraphOfThoughts
from uraf.uncertainty_quantification import (
    SemanticUncertainty,
    ConformalPrediction,
    CalibrationModule,
    UncertaintyEstimator,
    calculate_expected_calibration_error,
)


class TestSelfImprovingAgent:
    """Tests for STaR (Self-Taught Reasoner) agent."""

    def test_star_agent_initialization(self):
        """Test STaR agent initialization."""
        llm = LLMClient()
        agent = STaRAgent(llm_client=llm, quality_threshold=0.7, max_iterations=3)

        assert agent is not None
        assert agent.quality_threshold == 0.7
        assert agent.max_iterations == 3
        assert len(agent.reasoning_library) == 0

    @pytest.mark.asyncio
    async def test_star_agent_solve(self):
        """Test STaR agent solving a problem."""
        llm = LLMClient()
        agent = STaRAgent(llm_client=llm, max_iterations=2)

        problem = "What is 15 + 27?"
        result = await agent.solve_with_improvement(problem, ground_truth="42", verbose=False)

        assert "best_solution" in result
        assert "best_quality" in result
        assert "improvement_history" in result
        assert len(result["improvement_history"]) > 0

    def test_improvement_rate_calculation(self):
        """Test improvement rate calculation."""
        history = [
            {"iteration": 1, "quality": 0.5},
            {"iteration": 2, "quality": 0.65},
            {"iteration": 3, "quality": 0.75},
        ]

        metrics = calculate_improvement_rate(history)

        assert "improvement_rate" in metrics
        assert "total_improvement" in metrics
        assert metrics["total_improvement"] == pytest.approx(0.25, abs=0.01)
        assert metrics["improvement_rate"] > 0


class TestRecursiveSelfImprovement:
    """Tests for recursive self-improvement."""

    def test_initialization(self):
        """Test recursive self-improvement initialization."""
        llm = LLMClient()
        rsi = RecursiveSelfImprovement(llm_client=llm, max_rounds=3)

        assert rsi is not None
        assert rsi.max_rounds == 3

    @pytest.mark.asyncio
    async def test_improve_response(self):
        """Test response improvement."""
        llm = LLMClient()
        rsi = RecursiveSelfImprovement(llm_client=llm, max_rounds=2)

        initial_response = "The answer is probably 42."
        question = "What is the ultimate answer?"

        result = await rsi.improve_response(initial_response, question, improvement_threshold=0.9)

        assert "final_response" in result
        assert "improvement_history" in result
        assert "final_score" in result
        assert len(result["improvement_history"]) > 0


class TestUncertaintyQuantification:
    """Tests for uncertainty quantification."""

    def test_semantic_uncertainty_initialization(self):
        """Test semantic uncertainty initialization."""
        su = SemanticUncertainty(similarity_threshold=0.85)

        assert su is not None
        assert su.similarity_threshold == 0.85

    @pytest.mark.asyncio
    async def test_semantic_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        llm = LLMClient()
        su = SemanticUncertainty()

        prompt = "What is 2 + 2?"
        result = await su.estimate_uncertainty(llm, prompt, num_samples=3, temperature=0.1)

        assert "uncertainty" in result
        assert "confidence" in result
        assert "num_clusters" in result
        assert "samples" in result
        assert 0 <= result["uncertainty"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_conformal_prediction_initialization(self):
        """Test conformal prediction initialization."""
        cp = ConformalPrediction(alpha=0.1)

        assert cp is not None
        assert cp.alpha == 0.1

    def test_conformal_prediction_calibration(self):
        """Test conformal prediction calibration."""
        cp = ConformalPrediction(alpha=0.1)

        # Simulate validation scores
        validation_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        cp.calibrate(validation_scores)

        assert len(cp.calibration_scores) == 5

    def test_conformal_prediction_set(self):
        """Test prediction set generation."""
        cp = ConformalPrediction(alpha=0.1)
        cp.calibrate([0.5, 0.6, 0.7, 0.8, 0.9])

        candidates = ["Answer A", "Answer B", "Answer C"]
        scores = [0.85, 0.65, 0.45]

        result = cp.get_prediction_set(candidates, scores)

        assert "prediction_set" in result
        assert "set_size" in result
        assert "threshold" in result
        assert result["set_size"] <= len(candidates)

    def test_calibration_module(self):
        """Test calibration module."""
        import numpy as np

        cal = CalibrationModule()

        confidences = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 0, 1])

        temp = cal.temperature_scaling(confidences, labels)

        assert temp > 0
        assert cal.temperature == temp

    def test_expected_calibration_error(self):
        """Test ECE calculation."""
        import numpy as np

        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        accuracies = np.array([1, 1, 0, 0])

        ece = calculate_expected_calibration_error(confidences, accuracies, n_bins=5)

        assert ece >= 0
        assert ece <= 1


class TestTreeOfThoughts:
    """Tests for Tree-of-Thoughts reasoning."""

    def test_tot_initialization(self):
        """Test ToT initialization."""
        llm = LLMClient()
        tot = TreeOfThoughts(
            llm_client=llm, search_strategy=SearchStrategy.BEAM, max_depth=3, branching_factor=2, beam_width=2
        )

        assert tot is not None
        assert tot.search_strategy == SearchStrategy.BEAM
        assert tot.max_depth == 3
        assert tot.branching_factor == 2

    @pytest.mark.asyncio
    async def test_tot_solve_beam_search(self):
        """Test ToT solving with beam search."""
        llm = LLMClient()
        tot = TreeOfThoughts(llm_client=llm, search_strategy=SearchStrategy.BEAM, max_depth=2, beam_width=2)

        problem = "Find a creative solution to reduce plastic waste."
        result = await tot.solve(problem, verbose=False)

        assert "solution" in result
        assert "statistics" in result
        assert result["statistics"]["total_nodes_explored"] > 0

    @pytest.mark.asyncio
    async def test_tot_solve_bfs(self):
        """Test ToT solving with BFS."""
        llm = LLMClient()
        tot = TreeOfThoughts(llm_client=llm, search_strategy=SearchStrategy.BFS, max_depth=2)

        problem = "What is 10 + 20?"
        result = await tot.solve(problem, verbose=False)

        assert "solution" in result
        assert "statistics" in result


class TestGraphOfThoughts:
    """Tests for Graph-of-Thoughts."""

    def test_got_initialization(self):
        """Test GoT initialization."""
        llm = LLMClient()
        got = GraphOfThoughts(llm_client=llm)

        assert got is not None
        assert len(got.nodes) == 0

    @pytest.mark.asyncio
    async def test_got_solve_with_merge(self):
        """Test GoT solving with path merging."""
        llm = LLMClient()
        got = GraphOfThoughts(llm_client=llm)

        problem = "What are the benefits of renewable energy?"
        result = await got.solve_with_merge(problem, num_initial_paths=2)

        assert "solution" in result
        assert "answer" in result
        assert "merge_points" in result


class TestAdvancedRAG:
    """Tests for Advanced RAG 2.0."""

    def test_rag_initialization(self):
        """Test RAG initialization."""
        llm = LLMClient()
        rag = AdvancedRAG(
            llm_client=llm,
            collection_name="test_rag",
            persist_directory="data/test_rag_db",
            retrieval_mode=RetrievalMode.DENSE,
        )

        assert rag is not None
        assert rag.retrieval_mode == RetrievalMode.DENSE

    def test_rag_add_documents(self):
        """Test adding documents to RAG."""
        llm = LLMClient()
        rag = AdvancedRAG(llm_client=llm, collection_name="test_docs")

        documents = [
            "The sky is blue during the day.",
            "Water freezes at 0 degrees Celsius.",
            "Python is a programming language.",
        ]

        rag.add_documents(documents)

        assert len(rag.sparse_docs) == 3

    @pytest.mark.asyncio
    async def test_rag_query(self):
        """Test RAG query."""
        llm = LLMClient()
        rag = AdvancedRAG(llm_client=llm, collection_name="test_query")

        # Add some documents
        documents = ["Artificial intelligence is transforming technology.", "Machine learning uses data to learn patterns."]

        rag.add_documents(documents)

        # Query
        result = await rag.query("What is AI?", top_k=2, rerank=False)

        assert "answer" in result
        assert "sources" in result
        assert "num_sources_used" in result

    @pytest.mark.asyncio
    async def test_rag_query_with_transform(self):
        """Test RAG query with transformation."""
        llm = LLMClient()
        rag = AdvancedRAG(llm_client=llm, collection_name="test_transform")

        documents = ["Climate change affects weather patterns.", "Renewable energy reduces carbon emissions."]

        rag.add_documents(documents)

        result = await rag.query("Tell me about climate", top_k=2, query_transform=QueryTransformStrategy.EXPANSION)

        assert "answer" in result
        assert result["query_transformed"] is True


class TestSelfRAG:
    """Tests for Self-RAG."""

    def test_self_rag_initialization(self):
        """Test Self-RAG initialization."""
        llm = LLMClient()
        rag = AdvancedRAG(llm_client=llm, collection_name="test_self_rag")
        self_rag = SelfRAG(rag_system=rag, llm_client=llm)

        assert self_rag is not None

    @pytest.mark.asyncio
    async def test_self_rag_query(self):
        """Test Self-RAG query with reflection."""
        llm = LLMClient()
        rag = AdvancedRAG(llm_client=llm, collection_name="test_self_rag_query")
        rag.add_documents(["The Earth orbits the Sun.", "Gravity is a fundamental force."])

        self_rag = SelfRAG(rag_system=rag, llm_client=llm)

        result = await self_rag.query_with_reflection("What is gravity?")

        assert "answer" in result
        assert "retrieval_used" in result
        assert "reflection" in result
        assert "confidence" in result["reflection"]


class TestCausalReasoning:
    """Tests for causal reasoning."""

    def test_causal_reasoner_initialization(self):
        """Test causal reasoner initialization."""
        llm = LLMClient()
        reasoner = CausalReasoner(llm_client=llm)

        assert reasoner is not None

    @pytest.mark.asyncio
    async def test_analyze_causality(self):
        """Test causal analysis."""
        llm = LLMClient()
        reasoner = CausalReasoner(llm_client=llm)

        text = "Smoking causes lung cancer. Exercise improves heart health."
        result = await reasoner.analyze_causality(text)

        assert "causal_relationships" in result
        assert "causal_graph" in result
        assert "num_relationships" in result

    @pytest.mark.asyncio
    async def test_intervention_reasoning(self):
        """Test intervention reasoning."""
        llm = LLMClient()
        reasoner = CausalReasoner(llm_client=llm)

        scenario = "Higher education leads to better jobs, which lead to higher income."
        intervention = "Provide free university education to everyone."

        result = await reasoner.intervention_reasoning(scenario, intervention)

        assert "intervention" in result
        assert "affected_variables" in result
        assert "predicted_effects" in result

    @pytest.mark.asyncio
    async def test_counterfactual_reasoning(self):
        """Test counterfactual reasoning."""
        llm = LLMClient()
        reasoner = CausalReasoner(llm_client=llm)

        scenario = "I studied hard and passed the exam."
        actual = "Passed the exam"
        counterfactual = "What if I had not studied?"

        result = await reasoner.counterfactual_reasoning(scenario, actual, counterfactual)

        assert "counterfactual_analysis" in result
        assert "confidence" in result

    def test_detect_causal_language(self):
        """Test causal language detection."""
        text = "Smoking causes cancer. Exercise affects health. What if we intervened?"

        result = detect_causal_language(text)

        assert "indicators_by_category" in result
        assert "total_indicators" in result
        assert "has_causal_language" in result
        assert result["has_causal_language"] is True

    def test_causal_discovery_initialization(self):
        """Test causal discovery initialization."""
        llm = LLMClient()
        discovery = CausalDiscovery(llm_client=llm)

        assert discovery is not None

    @pytest.mark.asyncio
    async def test_causal_discovery(self):
        """Test causal structure discovery."""
        llm = LLMClient()
        discovery = CausalDiscovery(llm_client=llm)

        observations = ["When it rains, the ground gets wet.", "Wet ground makes plants grow.", "Growing plants produce food."]

        result = await discovery.discover_causal_structure(observations)

        assert "causal_structure" in result
        assert "num_observations" in result
        assert result["num_observations"] == 3
