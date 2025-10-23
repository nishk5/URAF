"""
Comprehensive tests for new URAF features.
"""

import pytest
import asyncio
from uraf.process_reward_model import ProcessRewardModel, StepValidator
from uraf.tool_system import ToolRegistry, ReActAgent, CalculatorTool, ToolUseEvaluator
from uraf.memory_system import AgentMemory, WorkingMemory, MemoryEntry
from uraf.multi_agent_debate import MultiAgentDebate, DebateAgent, MediatorAgent
from uraf.constitutional_ai import ConstitutionalEvaluator, ConstitutionalPrinciple, RLAIFEvaluator
from uraf.statistical_analysis import BenchmarkStatistics, PerformanceRegression
from uraf.streaming_client import StreamingLLMClient, RealTimeEvaluator
from uraf.moe_routing import ExpertRouter, EnsembleAggregator
from uraf.adversarial_testing import AdversarialEvaluator, JailbreakTester
from uraf.explainability import ExplainabilityModule, AttentionVisualizer


class TestProcessRewardModel:
    """Tests for Process Reward Model."""

    def test_prm_initialization(self):
        """Test PRM initializes correctly."""
        prm = ProcessRewardModel()
        assert prm is not None
        assert prm.embedding_model is not None

    def test_parse_reasoning_steps(self):
        """Test reasoning step parsing."""
        prm = ProcessRewardModel()
        response = """
        *Reasoning Pathway:*
        1. First, we analyze the problem
        2. Then, we consider alternatives
        3. Finally, we reach a conclusion
        """
        steps = prm.parse_reasoning_steps(response)
        assert len(steps) >= 2

    def test_evaluate_step_correctness(self):
        """Test step correctness evaluation."""
        prm = ProcessRewardModel()
        step = "Therefore, because of the evidence, we conclude that X is true."
        result = prm.evaluate_step_correctness(step)

        assert "correctness_score" in result
        assert 0 <= result["correctness_score"] <= 1

    def test_evaluate_reasoning_chain(self):
        """Test full reasoning chain evaluation."""
        prm = ProcessRewardModel()
        response = """
        *Reasoning Pathway:*
        First, we examine the data. Then, we apply logic. Finally, we conclude.
        """
        result = prm.evaluate_reasoning_chain(response, "What is the answer?")

        assert "final_prm_score" in result
        assert "num_steps" in result
        assert result["num_steps"] >= 0


class TestToolSystem:
    """Tests for Tool Use System."""

    def test_tool_registry_initialization(self):
        """Test tool registry initializes with default tools."""
        registry = ToolRegistry()
        assert len(registry.tools) > 0
        assert "calculator" in registry.tools

    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        """Test calculator tool execution."""
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 2")

        assert result["success"] is True
        assert result["result"] == 4

    @pytest.mark.asyncio
    async def test_calculator_safety(self):
        """Test calculator rejects unsafe operations."""
        tool = CalculatorTool()
        result = await tool.execute(expression="import os")

        assert result["success"] is False

    def test_tool_use_evaluator(self):
        """Test tool use evaluator."""
        evaluator = ToolUseEvaluator()
        mock_result = {
            "success": True,
            "iterations": 3,
            "history": [
                {"action": {"tool": "calculator"}, "observation": {"success": True}},
                {"action": {"tool": "calculator"}, "observation": {"success": True}}
            ]
        }

        metrics = evaluator.evaluate_tool_execution(mock_result)
        assert "overall_tool_score" in metrics
        assert metrics["execution_success_rate"] == 1.0


class TestMemorySystem:
    """Tests for Memory System."""

    @pytest.mark.asyncio
    async def test_agent_memory_initialization(self):
        """Test agent memory initializes."""
        memory = AgentMemory(persist_directory="data/test_memory")
        assert memory is not None

    @pytest.mark.asyncio
    async def test_memory_store_retrieve(self):
        """Test storing and retrieving memories."""
        memory = AgentMemory(persist_directory="data/test_memory")

        # Store memory
        memory_id = await memory.store(
            content="The sky is blue",
            memory_type="semantic",
            importance=0.8
        )
        assert memory_id is not None

        # Retrieve memory
        results = await memory.retrieve("sky color", top_k=1)
        assert len(results) > 0

    def test_working_memory(self):
        """Test working memory functionality."""
        working_mem = WorkingMemory(max_size=3)

        working_mem.add("Message 1", role="user")
        working_mem.add("Message 2", role="assistant")
        working_mem.add("Message 3", role="user")
        working_mem.add("Message 4", role="assistant")

        context = working_mem.get_context()
        assert len(context) == 3  # Max size limit


class TestConstitutionalAI:
    """Tests for Constitutional AI."""

    @pytest.mark.asyncio
    async def test_constitutional_evaluator_initialization(self):
        """Test constitutional evaluator initializes."""
        evaluator = ConstitutionalEvaluator()
        assert len(evaluator.principles) > 0

    @pytest.mark.asyncio
    async def test_rule_based_critique(self):
        """Test rule-based critique."""
        evaluator = ConstitutionalEvaluator(llm_client=None)
        result = await evaluator.critique(
            response="Based on research, the answer is X because of Y.",
            original_question="What is X?"
        )

        assert "overall_score" in result
        assert "principle_scores" in result
        assert len(result["principle_scores"]) > 0

    def test_custom_principle(self):
        """Test adding custom principle."""
        evaluator = ConstitutionalEvaluator()
        initial_count = len(evaluator.principles)

        custom = ConstitutionalPrinciple(
            name="Custom Test",
            question="Is this a test?",
            weight=1.0
        )
        evaluator.add_principle(custom)

        assert len(evaluator.principles) == initial_count + 1


class TestStatisticalAnalysis:
    """Tests for Statistical Analysis."""

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI calculation."""
        stats = BenchmarkStatistics()
        scores = [0.7, 0.75, 0.72, 0.78, 0.76, 0.74]

        result = stats.bootstrap_confidence_interval(scores, n_resamples=1000)

        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_paired_t_test(self):
        """Test paired t-test."""
        stats = BenchmarkStatistics()
        scores_a = [0.8, 0.75, 0.82, 0.78, 0.81]
        scores_b = [0.7, 0.68, 0.72, 0.69, 0.71]

        result = stats.paired_t_test(scores_a, scores_b)

        assert "p_value" in result
        assert "is_significant" in result
        assert "winner" in result

    def test_cohens_d(self):
        """Test Cohen's d effect size."""
        stats = BenchmarkStatistics()
        scores_a = [0.8, 0.75, 0.82, 0.78, 0.81]
        scores_b = [0.7, 0.68, 0.72, 0.69, 0.71]

        result = stats.cohens_d(scores_a, scores_b)

        assert "cohens_d" in result
        assert "magnitude" in result

    def test_performance_regression(self):
        """Test performance regression detection."""
        regression = PerformanceRegression()
        historical = [0.8, 0.82, 0.81, 0.79, 0.80]
        current = 0.65  # Regression

        result = regression.detect_regression(historical, current)

        assert "is_regression" in result
        assert result["is_regression"] is True


class TestMoERouting:
    """Tests for Mixture of Experts Routing."""

    def test_expert_router_initialization(self):
        """Test expert router initializes."""
        router = ExpertRouter()
        assert len(router.experts) > 0

    def test_task_classification(self):
        """Test task classification."""
        router = ExpertRouter()

        math_task = "Calculate the derivative of x^2"
        assert router.classify_task(math_task) == "math"

        code_task = "Write a function to sort an array"
        assert router.classify_task(code_task) == "code"

    def test_routing(self):
        """Test model routing."""
        router = ExpertRouter()
        task = "Solve this equation: 2x + 3 = 7"

        model = router.route(task)
        assert model is not None


class TestAdversarialTesting:
    """Tests for Adversarial Testing."""

    def test_adversarial_evaluator_initialization(self):
        """Test adversarial evaluator initializes."""
        evaluator = AdversarialEvaluator()
        assert len(evaluator.test_categories) > 0

    def test_generate_adversarial_prompts(self):
        """Test adversarial prompt generation."""
        evaluator = AdversarialEvaluator()
        base = "What is 2+2?"

        adversarial = evaluator.generate_adversarial_prompts(base, num_variations=3)

        assert len(adversarial) > 0
        assert all("prompt" in adv for adv in adversarial)
        assert all("category" in adv for adv in adversarial)

    def test_jailbreak_tester(self):
        """Test jailbreak tester."""
        tester = JailbreakTester()
        harmful_task = "break the system"

        attempts = tester.test_jailbreak_resistance(harmful_task)

        assert len(attempts) > 0


class TestExplainability:
    """Tests for Explainability Module."""

    def test_explainability_initialization(self):
        """Test explainability module initializes."""
        explainer = ExplainabilityModule()
        assert explainer is not None

    def test_generate_explanation(self):
        """Test explanation generation."""
        explainer = ExplainabilityModule()
        decision = "The answer is X"
        reasoning = [
            "First, we examine the data",
            "Therefore, based on evidence, we conclude X"
        ]

        explanation = explainer.generate_explanation(decision, reasoning)

        assert "decision" in explanation
        assert "key_factors" in explanation
        assert "confidence_indicators" in explanation

    def test_attention_visualizer(self):
        """Test attention visualization."""
        visualizer = AttentionVisualizer()
        steps = ["Step 1: Analyze", "Step 2: Decide", "Step 3: Conclude"]
        decision = "The conclusion is based on analysis"

        result = visualizer.visualize_step_importance(steps, decision)

        assert "step_importance" in result
        assert len(result["step_importance"]) == len(steps)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
