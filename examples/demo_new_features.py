"""
Demo Script: Showcasing URAF's Cutting-Edge Features

Demonstrates all new features added to URAF framework.
"""

import asyncio
import sys
sys.path.insert(0, '/home/user/URAF')

from uraf.llm_client import LLMClient
from uraf.process_reward_model import ProcessRewardModel
from uraf.tool_system import ToolRegistry, ReActAgent
from uraf.memory_system import AgentMemory, WorkingMemory
from uraf.multi_agent_debate import MultiAgentDebate
from uraf.constitutional_ai import ConstitutionalEvaluator
from uraf.statistical_analysis import BenchmarkStatistics
from uraf.moe_routing import ExpertRouter
from uraf.adversarial_testing import AdversarialEvaluator
from uraf.explainability import ExplainabilityModule
from loguru import logger


async def demo_process_reward_model():
    """Demonstrate Process Reward Model (PRM)."""
    print("\n" + "="*60)
    print("DEMO 1: Process Reward Model (PRM)")
    print("="*60)

    prm = ProcessRewardModel()

    sample_response = """
    *Understanding:* The problem asks us to find the solution.
    *Reasoning Pathway:*
    First, we analyze the given data carefully.
    Then, we apply logical deduction to narrow down possibilities.
    Therefore, based on evidence, we can conclude the answer.
    Finally, we verify our conclusion is consistent.
    *Final Synthesis:* The answer is well-supported.
    """

    result = prm.evaluate_reasoning_chain(sample_response, "What is the answer?")

    print(f"\n📊 PRM Evaluation Results:")
    print(f"   Number of steps: {result['num_steps']}")
    print(f"   Average step correctness: {result['avg_step_correctness']:.3f}")
    print(f"   Consistency score: {result['consistency_metrics']['consistency_score']:.3f}")
    print(f"   Self-correction bonus: {result['self_correction']['correction_bonus']:.3f}")
    print(f"   Final PRM score: {result['final_prm_score']:.3f}")


async def demo_tool_use_system():
    """Demonstrate Tool Use with ReAct Agent."""
    print("\n" + "="*60)
    print("DEMO 2: Tool Use System (ReAct Agent)")
    print("="*60)

    registry = ToolRegistry()

    print(f"\n🔧 Available Tools:")
    for tool_info in registry.list_tools():
        print(f"   - {tool_info['name']}: {tool_info['description']}")

    # Test calculator
    print(f"\n🧮 Testing Calculator Tool:")
    result = await registry.execute_tool("calculator", expression="sqrt(16) + 2**3")
    print(f"   Result: {result}")


async def demo_memory_system():
    """Demonstrate Vector Memory System."""
    print("\n" + "="*60)
    print("DEMO 3: Vector Memory System")
    print("="*60)

    memory = AgentMemory(persist_directory="data/demo_memory")

    # Store some memories
    print(f"\n💾 Storing memories...")
    await memory.store("Python is a programming language", memory_type="semantic", importance=0.9)
    await memory.store("I learned about functions today", memory_type="episodic", importance=0.7)
    await memory.store("Always test code before deployment", memory_type="procedural", importance=0.8)

    # Retrieve relevant memories
    print(f"\n🔍 Retrieving memories about 'programming'...")
    results = await memory.retrieve("programming", top_k=2)

    for i, mem in enumerate(results, 1):
        print(f"   {i}. [{mem['metadata']['memory_type']}] {mem['content'][:60]}...")
        print(f"      Similarity: {mem['similarity']:.3f}, Importance: {mem['metadata']['importance']}")

    # Memory statistics
    stats = await memory.get_memory_statistics()
    print(f"\n📈 Memory Statistics:")
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   By type: {stats['memory_types']}")


async def demo_constitutional_ai():
    """Demonstrate Constitutional AI Self-Critique."""
    print("\n" + "="*60)
    print("DEMO 4: Constitutional AI Self-Critique")
    print("="*60)

    evaluator = ConstitutionalEvaluator()

    sample_response = "Based on research, the answer is likely X because multiple studies have shown this pattern, though there is some uncertainty in edge cases."

    critique = await evaluator.critique(sample_response, "What is X?")

    print(f"\n⚖️ Constitutional Critique Results:")
    print(f"   Overall score: {critique['overall_score']:.3f}")
    print(f"   Violations: {critique['num_violations']}")
    print(f"\n   Principle Scores:")

    for principle in critique['principle_scores'][:5]:  # Top 5
        print(f"   - {principle['principle']}: {principle['score']:.2f}")
        print(f"     Feedback: {principle['feedback'][:80]}...")


def demo_statistical_analysis():
    """Demonstrate Statistical Analysis."""
    print("\n" + "="*60)
    print("DEMO 5: Statistical Analysis")
    print("="*60)

    stats = BenchmarkStatistics()

    # Mock benchmark scores
    model_a_scores = [0.82, 0.85, 0.81, 0.84, 0.83, 0.86, 0.82, 0.84]
    model_b_scores = [0.75, 0.77, 0.74, 0.76, 0.75, 0.78, 0.76, 0.75]

    # Confidence intervals
    print(f"\n📊 Confidence Intervals:")
    ci_a = stats.bootstrap_confidence_interval(model_a_scores)
    ci_b = stats.bootstrap_confidence_interval(model_b_scores)

    print(f"   Model A: {ci_a['mean']:.3f} [{ci_a['ci_lower']:.3f}, {ci_a['ci_upper']:.3f}]")
    print(f"   Model B: {ci_b['mean']:.3f} [{ci_b['ci_lower']:.3f}, {ci_b['ci_upper']:.3f}]")

    # Statistical test
    print(f"\n🧪 Paired T-Test:")
    t_test = stats.paired_t_test(model_a_scores, model_b_scores)
    print(f"   Winner: {t_test['winner']}")
    print(f"   p-value: {t_test['p_value']:.4f}")
    print(f"   Significant: {t_test['is_significant']}")

    # Effect size
    print(f"\n📏 Effect Size (Cohen's d):")
    effect = stats.cohens_d(model_a_scores, model_b_scores)
    print(f"   Cohen's d: {effect['cohens_d']:.3f}")
    print(f"   Magnitude: {effect['magnitude']}")


def demo_moe_routing():
    """Demonstrate Mixture of Experts Routing."""
    print("\n" + "="*60)
    print("DEMO 6: Mixture of Experts (MoE) Routing")
    print("="*60)

    router = ExpertRouter()

    tasks = [
        "Calculate the derivative of f(x) = x^3 + 2x",
        "Write a Python function to merge two sorted lists",
        "Analyze the philosophical implications of free will",
        "Write a creative story about a time traveler"
    ]

    print(f"\n🎯 Task Routing:")
    for task in tasks:
        expert_type = router.classify_task(task)
        recommended_model = router.route(task)
        print(f"\n   Task: {task[:50]}...")
        print(f"   Expert Type: {expert_type}")
        print(f"   Recommended Model: {recommended_model}")


def demo_adversarial_testing():
    """Demonstrate Adversarial Testing."""
    print("\n" + "="*60)
    print("DEMO 7: Adversarial Testing")
    print("="*60)

    evaluator = AdversarialEvaluator()

    base_prompt = "What is the capital of France?"

    adversarial_prompts = evaluator.generate_adversarial_prompts(base_prompt, num_variations=3)

    print(f"\n🎭 Adversarial Test Cases:")
    for i, adv in enumerate(adversarial_prompts, 1):
        print(f"\n   {i}. Category: {adv['category']}")
        print(f"      Prompt: {adv['prompt'][:100]}...")


def demo_explainability():
    """Demonstrate Explainability Module."""
    print("\n" + "="*60)
    print("DEMO 8: Explainability & Interpretability")
    print("="*60)

    explainer = ExplainabilityModule()

    decision = "The best solution is to use approach A because it's more efficient."
    reasoning_chain = [
        "First, we evaluated all available options.",
        "We found that approach A uses less memory.",
        "Therefore, approach A is preferable for this use case."
    ]

    explanation = explainer.generate_explanation(decision, reasoning_chain)

    print(f"\n🔍 Explanation Generated:")
    print(f"   Decision: {explanation['decision']}")
    print(f"   Key Factors: {explanation['key_factors']}")
    print(f"   Confidence Level: {explanation['confidence_indicators']['level']}")
    print(f"   Important Terms: {explanation['important_terms'][:5]}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🚀 URAF CUTTING-EDGE FEATURES DEMO")
    print("="*60)
    print("\nDemonstrating all new features added to URAF framework...")

    try:
        await demo_process_reward_model()
        await demo_tool_use_system()
        await demo_memory_system()
        await demo_constitutional_ai()
        demo_statistical_analysis()
        demo_moe_routing()
        demo_adversarial_testing()
        demo_explainability()

        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
