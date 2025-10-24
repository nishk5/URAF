"""
Demo script for blog post: Advanced AI Features in URAF

This script demonstrates the new advanced features with practical examples.
Run with: uv run python examples/demo_advanced_features_blog.py
"""

import asyncio
from typing import Any

# Mock LLM for demonstration
class MockLLMClient:
    """Mock LLM client for demonstration purposes."""

    def __init__(self):
        self.params = {"temperature": 0.7}

    async def query(self, prompt: str) -> str:
        """Simulate LLM response."""
        # Simulate different responses based on prompt content
        prompt_lower = prompt.lower()

        if "2 + 2" in prompt_lower or "15 + 27" in prompt_lower:
            return "Let me solve this step by step.\n\n*Reasoning:* 15 + 27 = 42\n\n*Final Answer:* 42"

        elif "capital of france" in prompt_lower:
            return "The capital of France is Paris."

        elif "photosynthesis" in prompt_lower:
            return """*Understanding:* Photosynthesis is the process by which plants convert sunlight into chemical energy.

*Reasoning Pathway:*
1. Plants absorb sunlight through chlorophyll
2. They take in CO2 from the atmosphere
3. They use water from the soil
4. Through chemical reactions, they produce glucose and oxygen
5. The oxygen is released into the atmosphere

*Final Answer:* Photosynthesis converts sunlight, CO2, and water into glucose and oxygen."""

        elif "climate change" in prompt_lower or "sustainable" in prompt_lower:
            return """*Understanding:* We need to reduce carbon emissions and create sustainable systems.

*Reasoning Pathway:*
1. Transition to renewable energy (solar, wind)
2. Improve public transportation infrastructure
3. Promote electric vehicles
4. Create green spaces and urban forests
5. Implement carbon pricing

*Final Answer:* A multi-faceted approach combining renewable energy, better transit, and policy changes."""

        elif "exercise" in prompt_lower or "health" in prompt_lower:
            return "Regular exercise improves cardiovascular health, strengthens muscles, boosts mental well-being, and helps maintain a healthy weight."

        elif "smoking" in prompt_lower:
            return "Smoking causes lung cancer through the carcinogenic compounds in tobacco smoke that damage lung tissue DNA."

        elif "retrieve" in prompt_lower or "no_retrieve" in prompt_lower:
            return "RETRIEVE" if "require" in prompt_lower or "fact" in prompt_lower else "NO_RETRIEVE"

        elif "yes or no" in prompt_lower:
            return "YES" if "relevant" in prompt_lower or "supported" in prompt_lower else "NO"

        else:
            return f"This is a thoughtful response addressing: {prompt[:100]}..."


async def demo_self_improving_agent():
    """Demo 1: Self-Improving Agent (STaR)."""
    print("=" * 80)
    print("DEMO 1: Self-Improving Agent (STaR)")
    print("=" * 80)
    print()

    from uraf.self_improving_agent import STaRAgent, calculate_improvement_rate

    llm = MockLLMClient()
    agent = STaRAgent(
        llm_client=llm,
        quality_threshold=0.6,
        max_iterations=3
    )

    print("🎯 Problem: What is 15 + 27?")
    print()

    result = await agent.solve_with_improvement(
        problem="What is 15 + 27?",
        ground_truth="42",
        verbose=True
    )

    print()
    print("📊 Results:")
    print(f"  Best Quality: {result['best_quality']:.3f}")
    print(f"  Iterations: {result['final_iteration']}")
    print(f"  Reasoning Library Size: {result['library_size']}")

    # Calculate improvement metrics
    if len(result['improvement_history']) > 1:
        metrics = calculate_improvement_rate(result['improvement_history'])
        print(f"  Improvement Rate: {metrics['improvement_rate']:.3f}")
        print(f"  Total Improvement: {metrics['total_improvement']:.3f}")
        print(f"  Converged: {metrics['converged']}")

    print()
    print("✨ Key Insight: Agent improved its reasoning through self-critique!")
    print()


async def demo_uncertainty_quantification():
    """Demo 2: Uncertainty Quantification."""
    print("=" * 80)
    print("DEMO 2: Uncertainty Quantification")
    print("=" * 80)
    print()

    from uraf.uncertainty_quantification import SemanticUncertainty

    llm = MockLLMClient()
    su = SemanticUncertainty(similarity_threshold=0.85)

    print("🎯 Question: What are the benefits of exercise?")
    print()

    result = await su.estimate_uncertainty(
        llm_client=llm,
        prompt="What are the benefits of exercise?",
        num_samples=5,
        temperature=0.8
    )

    print("📊 Results:")
    print(f"  Uncertainty Score: {result['uncertainty']:.3f}")
    print(f"  Confidence Score: {result['confidence']:.3f}")
    print(f"  Number of Clusters: {result['num_clusters']}")
    print(f"  Agreement Rate: {result['agreement_rate']:.2%}")
    print()
    print(f"  Most Common Answer: {result['most_common_answer'][:100]}...")
    print()

    # Interpretation
    if result['uncertainty'] < 0.3:
        print("✨ Interpretation: Low uncertainty - high confidence in the answer")
    elif result['uncertainty'] < 0.6:
        print("✨ Interpretation: Moderate uncertainty - some variation in responses")
    else:
        print("✨ Interpretation: High uncertainty - diverse responses, seek more info")
    print()


async def demo_tree_of_thoughts():
    """Demo 3: Tree-of-Thoughts Reasoning."""
    print("=" * 80)
    print("DEMO 3: Tree-of-Thoughts Reasoning")
    print("=" * 80)
    print()

    from uraf.tree_of_thoughts import TreeOfThoughts, SearchStrategy

    llm = MockLLMClient()
    tot = TreeOfThoughts(
        llm_client=llm,
        search_strategy=SearchStrategy.BEAM,
        max_depth=3,
        branching_factor=2,
        beam_width=2
    )

    print("🎯 Problem: Design a sustainable transportation system for a city")
    print()
    print("🔍 Search Strategy: Beam Search (explores top-2 paths at each level)")
    print()

    result = await tot.solve(
        problem="Design a sustainable transportation system for a city",
        verbose=True
    )

    print()
    print("📊 Results:")
    print(f"  Solution Value: {result['value']:.3f}")
    print(f"  Nodes Explored: {result['statistics']['total_nodes_explored']}")
    print(f"  Max Depth Reached: {result['statistics']['max_depth_reached']}")
    print(f"  Search Strategy: {result['statistics']['search_strategy']}")
    print()
    print("✨ Key Insight: Explored multiple reasoning paths simultaneously!")
    print()


async def demo_advanced_rag():
    """Demo 4: Advanced RAG 2.0."""
    print("=" * 80)
    print("DEMO 4: Advanced RAG 2.0 with Self-Reflection")
    print("=" * 80)
    print()

    from uraf.rag_system import AdvancedRAG, RetrievalMode, QueryTransformStrategy

    llm = MockLLMClient()
    rag = AdvancedRAG(
        llm_client=llm,
        collection_name="demo_knowledge",
        persist_directory="data/demo_rag",
        retrieval_mode=RetrievalMode.HYBRID
    )

    # Add documents
    print("📚 Adding knowledge base documents...")
    documents = [
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "Plants absorb CO2 from the atmosphere during photosynthesis.",
        "Photosynthesis produces oxygen as a byproduct, which is essential for life.",
        "Climate change is affecting global weather patterns and ecosystems.",
        "Reducing carbon emissions is crucial for mitigating climate change.",
    ]

    rag.add_documents(documents)
    print(f"  Added {len(documents)} documents to knowledge base")
    print()

    # Query with hybrid retrieval
    print("🎯 Query: How does photosynthesis relate to climate change?")
    print("🔍 Using: Hybrid Retrieval (Dense + Sparse) with Reranking")
    print()

    result = await rag.query(
        query="How does photosynthesis relate to climate change?",
        top_k=3,
        rerank=True,
        query_transform=QueryTransformStrategy.NONE
    )

    print("📊 Results:")
    print(f"  Sources Used: {result['num_sources_used']}")
    print(f"  Retrieval Method: {result['retrieval_method']}")
    print()
    print(f"  Answer: {result['answer'][:200]}...")
    print()
    print("  Top Sources:")
    for i, source in enumerate(result['sources'][:3], 1):
        print(f"    {i}. Score: {source['score']:.3f} - {source['content'][:80]}...")
    print()
    print("✨ Key Insight: Hybrid search combines semantic and keyword matching!")
    print()


async def demo_causal_reasoning():
    """Demo 5: Causal Reasoning."""
    print("=" * 80)
    print("DEMO 5: Causal Reasoning (Pearl's Hierarchy)")
    print("=" * 80)
    print()

    from uraf.causal_reasoning import CausalReasoner, detect_causal_language

    llm = MockLLMClient()
    reasoner = CausalReasoner(llm_client=llm)

    # Detect causal language
    text = """
    Smoking causes lung cancer.
    Regular exercise improves cardiovascular health.
    Poor diet leads to obesity, which increases diabetes risk.
    """

    print("📝 Analyzing text for causal relationships...")
    print()

    causal_detection = detect_causal_language(text)
    print("📊 Causal Language Detection:")
    print(f"  Has Causal Language: {causal_detection['has_causal_language']}")
    print(f"  Total Indicators: {causal_detection['total_indicators']}")
    print()

    for category, data in causal_detection['indicators_by_category'].items():
        if data['count'] > 0:
            print(f"  {category}: {data['count']} occurrences")
            print(f"    Examples: {data['examples']}")

    print()

    # Analyze causality
    print("🔍 Extracting causal relationships...")
    result = await reasoner.analyze_causality(text)

    print()
    print("📊 Causal Analysis:")
    print(f"  Found {result['num_relationships']} causal relationships")
    print()

    # Intervention reasoning
    print("🎯 Intervention Reasoning Example:")
    print("  Scenario: Education → Jobs → Income → Healthcare")
    print("  Intervention: Provide free university education")
    print()

    intervention_result = await reasoner.intervention_reasoning(
        scenario="Education improves job prospects. Good jobs provide higher income. Higher income enables better healthcare.",
        intervention="Provide free university education to all citizens"
    )

    print("  Predicted Effects:")
    print(f"    {intervention_result['predicted_effects'][:300]}...")
    print()
    print("✨ Key Insight: Reasoning about cause and effect, not just correlation!")
    print()


async def demo_integrated_workflow():
    """Demo 6: Integrated Workflow - Combining Multiple Features."""
    print("=" * 80)
    print("DEMO 6: Integrated Workflow - All Features Together")
    print("=" * 80)
    print()

    from uraf.tree_of_thoughts import TreeOfThoughts, SearchStrategy
    from uraf.uncertainty_quantification import UncertaintyEstimator
    from uraf.self_improving_agent import STaRAgent

    llm = MockLLMClient()

    print("🎯 Complex Problem: Design a carbon-neutral city by 2040")
    print()

    # Step 1: Use ToT to explore solutions
    print("Step 1: Tree-of-Thoughts Exploration")
    print("  Exploring multiple solution paths...")
    tot = TreeOfThoughts(
        llm_client=llm,
        search_strategy=SearchStrategy.BEAM,
        max_depth=2,
        beam_width=2
    )

    tot_result = await tot.solve("Design a carbon-neutral city by 2040")
    print(f"  ✓ Explored {tot_result['statistics']['total_nodes_explored']} reasoning paths")
    print()

    # Step 2: Refine with self-improvement
    print("Step 2: Self-Improving Agent Refinement")
    print("  Iteratively improving the solution...")
    star = STaRAgent(llm_client=llm, max_iterations=2)

    improved = await star.solve_with_improvement(
        problem="Design a carbon-neutral city by 2040",
        ground_truth=None
    )
    print(f"  ✓ Improved quality from initial to {improved['best_quality']:.3f}")
    print()

    # Step 3: Assess uncertainty
    print("Step 3: Uncertainty Quantification")
    print("  Estimating confidence in the solution...")
    estimator = UncertaintyEstimator(llm_client=llm)

    uncertainty = await estimator.estimate_comprehensive_uncertainty(
        prompt="Design a carbon-neutral city by 2040",
        num_samples=3
    )
    print(f"  ✓ Confidence: {uncertainty['confidence_score']:.2%}")
    print(f"  ✓ Recommendation: {uncertainty['recommended_action']}")
    print()

    print("=" * 80)
    print("🎉 FINAL INTEGRATED RESULT")
    print("=" * 80)
    print()
    print("📊 Summary:")
    print(f"  • Reasoning Paths Explored: {tot_result['statistics']['total_nodes_explored']}")
    print(f"  • Solution Quality: {improved['best_quality']:.3f}")
    print(f"  • Confidence: {uncertainty['confidence_score']:.2%}")
    print(f"  • Recommendation: {uncertainty['recommended_action']}")
    print()
    print("✨ This demonstrates the power of combining multiple advanced AI techniques!")
    print()


async def main():
    """Run all demos."""
    print()
    print("🚀 URAF Advanced Features Demo")
    print("   Based on 2024-2025 Research")
    print()
    print("This demo showcases 5 cutting-edge AI modules:")
    print("  1. Self-Improving Agents (STaR)")
    print("  2. Uncertainty Quantification")
    print("  3. Tree-of-Thoughts Reasoning")
    print("  4. Advanced RAG 2.0")
    print("  5. Causal Reasoning")
    print()
    input("Press Enter to start...")
    print()

    try:
        await demo_self_improving_agent()
        input("Press Enter for next demo...")
        print()

        await demo_uncertainty_quantification()
        input("Press Enter for next demo...")
        print()

        await demo_tree_of_thoughts()
        input("Press Enter for next demo...")
        print()

        await demo_advanced_rag()
        input("Press Enter for next demo...")
        print()

        await demo_causal_reasoning()
        input("Press Enter for next demo...")
        print()

        await demo_integrated_workflow()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return

    print()
    print("=" * 80)
    print("🎉 Demo Complete!")
    print("=" * 80)
    print()
    print("📚 Learn more:")
    print("  • Documentation: docs/ADVANCED_FEATURES_GUIDE.md")
    print("  • Research: docs/ADVANCED_RESEARCH_2024_2025.md")
    print("  • Tests: tests/test_advanced_features.py")
    print()
    print("🚀 Start building with URAF's advanced features today!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
