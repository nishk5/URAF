# URAF Advanced Features Guide (2024-2025)

**Version 0.4.0** - Next-Generation AI Agency Framework

This guide covers the advanced AI features implemented in URAF based on the latest research from 2024-2025.

---

## Table of Contents

1. [Self-Improving Agents (STaR)](#self-improving-agents-star)
2. [Uncertainty Quantification](#uncertainty-quantification)
3. [Tree-of-Thoughts Reasoning](#tree-of-thoughts-reasoning)
4. [Advanced RAG 2.0](#advanced-rag-20)
5. [Causal Reasoning](#causal-reasoning)
6. [Quick Start Examples](#quick-start-examples)

---

## Self-Improving Agents (STaR)

**Based on**: "STaR: Bootstrapping Reasoning With Reasoning" (Zelikman et al., 2022, extended 2024)

### Overview

Self-Taught Reasoner (STaR) enables agents to recursively improve their reasoning by:
1. Generating multiple reasoning chains
2. Evaluating quality using Process Reward Models + Constitutional AI
3. Learning from high-quality reasoning patterns
4. Iteratively improving performance

### Features

- **STaRAgent**: Single agent that improves through iterations
- **RecursiveSelfImprovement**: Self-critique and refinement loops
- **MultiAgentSTaR**: Multiple agents learning from each other

### Usage Example

```python
from uraf.self_improving_agent import STaRAgent
from uraf.llm_client import LLMClient

# Initialize
llm = LLMClient()
agent = STaRAgent(
    llm_client=llm,
    quality_threshold=0.7,  # Minimum quality to learn from
    max_iterations=5         # Max improvement iterations
)

# Solve with self-improvement
result = await agent.solve_with_improvement(
    problem="What is the solution to 3x + 5 = 20?",
    ground_truth="x = 5",
    verbose=True
)

print(f"Best Quality: {result['best_quality']:.3f}")
print(f"Iterations: {result['final_iteration']}")
print(f"Solution: {result['best_solution']['reasoning']}")
```

### Recursive Self-Improvement

```python
from uraf.self_improving_agent import RecursiveSelfImprovement

rsi = RecursiveSelfImprovement(llm_client=llm, max_rounds=3)

result = await rsi.improve_response(
    initial_response="The answer is probably 42.",
    question="What is the ultimate answer to life?",
    improvement_threshold=0.8
)

print(f"Improved: {result['improved']}")
print(f"Final Score: {result['final_score']:.3f}")
print(f"Final Response: {result['final_response']}")
```

### Multi-Agent Collaborative Learning

```python
from uraf.self_improving_agent import MultiAgentSTaR

multi_star = MultiAgentSTaR(num_agents=3, llm_client=llm)

result = await multi_star.collaborative_solve(
    problem="Design an algorithm to optimize traffic flow.",
    ground_truth=None
)

print(f"Best Quality: {result['best_quality']:.3f}")
print(f"Shared Library Size: {result['shared_library_size']}")
```

---

## Uncertainty Quantification

**Based on**:
- "Semantic Uncertainty" (Kuhn et al., 2024)
- "Conformal Prediction for Language Models" (Angelopoulos et al., 2024)

### Overview

Provides statistically rigorous uncertainty estimates for LLM outputs, enabling:
- Confidence scoring with calibration guarantees
- Semantic uncertainty via clustering
- Conformal prediction sets
- Expected Calibration Error (ECE) metrics

### Features

- **SemanticUncertainty**: Cluster multiple outputs to measure disagreement
- **ConformalPrediction**: Statistically valid prediction sets
- **CalibrationModule**: Temperature and Platt scaling
- **UncertaintyEstimator**: Comprehensive uncertainty estimation

### Usage Example

```python
from uraf.uncertainty_quantification import UncertaintyEstimator
from uraf.llm_client import LLMClient

llm = LLMClient()
estimator = UncertaintyEstimator(llm_client=llm)

# Estimate uncertainty
result = await estimator.estimate_comprehensive_uncertainty(
    prompt="What is the capital of France?",
    num_samples=5
)

print(f"Uncertainty: {result['uncertainty_score']:.3f}")
print(f"Confidence: {result['confidence_score']:.3f}")
print(f"Interpretation: {result['interpretation']}")
print(f"Recommendation: {result['recommended_action']}")
```

### Semantic Uncertainty

```python
from uraf.uncertainty_quantification import SemanticUncertainty

semantic_unc = SemanticUncertainty(similarity_threshold=0.85)

result = await semantic_unc.estimate_uncertainty(
    llm_client=llm,
    prompt="What are the benefits of exercise?",
    num_samples=5,
    temperature=0.8
)

print(f"Num Clusters: {result['num_clusters']}")
print(f"Agreement Rate: {result['agreement_rate']:.2%}")
print(f"Most Common: {result['most_common_answer']}")
```

### Conformal Prediction

```python
from uraf.uncertainty_quantification import ConformalPrediction

cp = ConformalPrediction(alpha=0.1)  # 90% coverage guarantee

# Calibrate on validation set
validation_scores = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
cp.calibrate(validation_scores)

# Get prediction set
candidates = ["Paris", "London", "Berlin", "Rome"]
scores = [0.95, 0.65, 0.45, 0.40]

pred_set = cp.get_prediction_set(candidates, scores)

print(f"Prediction Set: {[p['answer'] for p in pred_set['prediction_set']]}")
print(f"Coverage: {pred_set['guaranteed_coverage']}")
```

### Calibration

```python
from uraf.uncertainty_quantification import CalibrationModule, calculate_expected_calibration_error
import numpy as np

cal = CalibrationModule()

# Calibration data
confidences = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
labels = np.array([1, 1, 0, 1, 1])

# Find optimal temperature
optimal_temp = cal.temperature_scaling(confidences, labels)

# Apply calibration
calibrated_conf = cal.apply_calibration(0.85, method="temperature")
print(f"Calibrated Confidence: {calibrated_conf:.3f}")

# Calculate ECE
accuracies = np.array([1, 1, 0, 1, 1])
ece = calculate_expected_calibration_error(confidences, accuracies)
print(f"ECE: {ece:.3f}")
```

---

## Tree-of-Thoughts Reasoning

**Based on**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2024)

### Overview

Tree-of-Thoughts (ToT) explores multiple reasoning paths as a tree structure, enabling:
- Exploration of alternative approaches
- Backtracking from dead ends
- Strategic lookahead and planning
- Better performance on complex problems

### Search Strategies

1. **BFS (Breadth-First Search)**: Explore all nodes at each depth level
2. **DFS (Depth-First Search)**: Explore one path to the end
3. **Beam Search**: Keep top-k best paths at each level
4. **MCTS (Monte Carlo Tree Search)**: Exploration-exploitation trade-off

### Usage Example

```python
from uraf.tree_of_thoughts import TreeOfThoughts, SearchStrategy
from uraf.llm_client import LLMClient

llm = LLMClient()

# Initialize with beam search
tot = TreeOfThoughts(
    llm_client=llm,
    search_strategy=SearchStrategy.BEAM,
    max_depth=5,
    branching_factor=3,
    beam_width=3
)

# Solve problem
result = await tot.solve(
    problem="Find three creative ways to reduce plastic waste in cities.",
    verbose=True
)

print(f"Solution:\n{result['solution']}")
print(f"Final Answer: {result['final_answer']}")
print(f"Value: {result['value']:.3f}")
print(f"Nodes Explored: {result['statistics']['total_nodes_explored']}")
```

### Different Search Strategies

```python
# BFS - Explore all options at each level
tot_bfs = TreeOfThoughts(
    llm_client=llm,
    search_strategy=SearchStrategy.BFS,
    max_depth=4
)

# DFS - Go deep into one path
tot_dfs = TreeOfThoughts(
    llm_client=llm,
    search_strategy=SearchStrategy.DFS,
    max_depth=5
)

# MCTS - Balanced exploration/exploitation
tot_mcts = TreeOfThoughts(
    llm_client=llm,
    search_strategy=SearchStrategy.MCTS,
    max_depth=4
)
```

### Graph-of-Thoughts (GoT)

```python
from uraf.tree_of_thoughts import GraphOfThoughts

got = GraphOfThoughts(llm_client=llm)

# Solve with path merging
result = await got.solve_with_merge(
    problem="What are the economic impacts of climate change?",
    num_initial_paths=3
)

print(f"Merged Solution: {result['solution']}")
print(f"Merge Points: {result['merge_points']}")
```

---

## Advanced RAG 2.0

**Based on**:
- "Self-RAG" (Asai et al., 2024)
- "Corrective RAG (CRAG)" (Yan et al., 2024)

### Overview

State-of-the-art retrieval-augmented generation with:
- **Query transformation**: HyDE, decomposition, expansion
- **Hybrid search**: Dense (embeddings) + Sparse (TF-IDF)
- **Multi-hop retrieval**: Iterative retrieval for complex questions
- **Cross-encoder reranking**: Accurate relevance scoring
- **Self-reflection**: Decide when to retrieve, assess relevance

### Features

- **Dense retrieval**: Semantic similarity with embeddings
- **Sparse retrieval**: Keyword-based (TF-IDF/BM25)
- **Hybrid retrieval**: Reciprocal Rank Fusion
- **Reranking**: Cross-encoder models
- **Query transformation**: HyDE, decomposition, expansion

### Usage Example

```python
from uraf.rag_system import AdvancedRAG, RetrievalMode, QueryTransformStrategy
from uraf.llm_client import LLMClient

llm = LLMClient()

# Initialize RAG system
rag = AdvancedRAG(
    llm_client=llm,
    collection_name="knowledge_base",
    persist_directory="data/rag_db",
    retrieval_mode=RetrievalMode.HYBRID  # Dense + Sparse
)

# Add documents
documents = [
    "Artificial intelligence is transforming healthcare through diagnostic tools.",
    "Machine learning models can predict disease outcomes with high accuracy.",
    "Natural language processing enables automated medical record analysis.",
]

rag.add_documents(documents, metadatas=[
    {"source": "medical_ai.pdf", "page": 1},
    {"source": "ml_healthcare.pdf", "page": 3},
    {"source": "nlp_records.pdf", "page": 5},
])

# Query with all features
result = await rag.query(
    query="How is AI used in medicine?",
    top_k=5,
    rerank=True,
    query_transform=QueryTransformStrategy.HYDE,
    multi_hop=False
)

print(f"Answer: {result['answer']}")
print(f"Sources Used: {result['num_sources_used']}")
print(f"Retrieval Method: {result['retrieval_method']}")

for i, source in enumerate(result['sources'], 1):
    print(f"\nSource {i}:")
    print(f"  Content: {source['content']}")
    print(f"  Score: {source['score']:.3f}")
    print(f"  Metadata: {source['metadata']}")
```

### Query Transformation Strategies

```python
# HyDE: Generate hypothetical document
result_hyde = await rag.query(
    query="What is quantum computing?",
    query_transform=QueryTransformStrategy.HYDE
)

# Decomposition: Break into sub-queries
result_decomp = await rag.query(
    query="What are the causes and effects of climate change?",
    query_transform=QueryTransformStrategy.DECOMPOSITION
)

# Expansion: Add related terms
result_expand = await rag.query(
    query="machine learning",
    query_transform=QueryTransformStrategy.EXPANSION
)
```

### Multi-Hop Retrieval

```python
# For complex questions requiring multiple retrieval steps
result = await rag.query(
    query="How does photosynthesis relate to climate change mitigation?",
    top_k=5,
    multi_hop=True  # Enable multi-hop
)
```

### Self-RAG with Reflection

```python
from uraf.rag_system import SelfRAG

self_rag = SelfRAG(rag_system=rag, llm_client=llm)

result = await self_rag.query_with_reflection(
    query="What is the speed of light?"
)

print(f"Answer: {result['answer']}")
print(f"Retrieval Used: {result['retrieval_used']}")
print(f"Reflection: {result['reflection']}")
print(f"  Should Retrieve: {result['reflection']['should_retrieve']}")
print(f"  Is Relevant: {result['reflection']['is_relevant']}")
print(f"  Is Supported: {result['reflection']['is_supported']}")
print(f"  Confidence: {result['reflection']['confidence']}")
```

---

## Causal Reasoning

**Based on**: Pearl's Causal Hierarchy (applied to LLMs, 2024)

### Overview

Enables reasoning about causality using Pearl's three-level hierarchy:

1. **Association (Seeing)**: P(Y|X) - Observational patterns
2. **Intervention (Doing)**: P(Y|do(X)) - Effects of actions
3. **Counterfactuals (Imagining)**: P(Y_x|X',Y') - What if scenarios

### Features

- **Causal relationship extraction**: Identify cause-effect pairs in text
- **Causal graph construction**: Build DAG of causal relationships
- **Intervention reasoning**: Predict effects of actions
- **Counterfactual reasoning**: Reason about alternative histories
- **Causal discovery**: Infer causal structure from observations

### Usage Example

```python
from uraf.causal_reasoning import CausalReasoner
from uraf.llm_client import LLMClient

llm = LLMClient()
reasoner = CausalReasoner(llm_client=llm)

# Analyze causality in text
text = """
Smoking causes lung cancer.
Regular exercise improves cardiovascular health.
Poor diet leads to obesity, which increases diabetes risk.
"""

result = await reasoner.analyze_causality(
    text=text,
    question="What causes diabetes?"
)

print(f"Found {result['num_relationships']} causal relationships")
for rel in result['causal_relationships']:
    print(f"  {rel['cause']} → {rel['effect']} (confidence: {rel['confidence']:.2f})")
    print(f"    Mechanism: {rel['mechanism']}")
    print(f"    Confounders: {rel['confounders']}")
```

### Intervention Reasoning

```python
# Reason about interventions (Doing)
scenario = """
Education improves job prospects.
Good jobs provide higher income.
Higher income enables better healthcare.
"""

intervention = "Provide free university education to all citizens."

result = await reasoner.intervention_reasoning(scenario, intervention)

print(f"Intervention: {result['intervention']}")
print(f"Affected Variables: {result['affected_variables']}")
print(f"\nPredicted Effects:\n{result['predicted_effects']}")
```

### Counterfactual Reasoning

```python
# Reason about counterfactuals (Imagining)
scenario = "I studied for 10 hours and scored 95% on the exam."
actual_outcome = "Scored 95%"
counterfactual = "What if I had only studied for 2 hours?"

result = await reasoner.counterfactual_reasoning(
    scenario=scenario,
    actual_outcome=actual_outcome,
    counterfactual_condition=counterfactual
)

print(f"Counterfactual Analysis:\n{result['counterfactual_analysis']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Causal Discovery

```python
from uraf.causal_reasoning import CausalDiscovery

discovery = CausalDiscovery(llm_client=llm)

observations = [
    "When temperature increases, ice cream sales increase.",
    "When temperature increases, drowning incidents increase.",
    "Ice cream sales and drowning incidents are correlated.",
]

result = await discovery.discover_causal_structure(observations)

print(f"Causal Structure:\n{result['causal_structure']}")
```

### Detect Causal Language

```python
from uraf.causal_reasoning import detect_causal_language

text = "Exercise causes improved health. Diet affects energy levels. What if we intervened?"

result = detect_causal_language(text)

print(f"Has Causal Language: {result['has_causal_language']}")
print(f"Total Indicators: {result['total_indicators']}")

for category, data in result['indicators_by_category'].items():
    print(f"\n{category}: {data['count']} occurrences")
    print(f"  Examples: {data['examples']}")
```

---

## Quick Start Examples

### Complete Agent Workflow

```python
from uraf.llm_client import LLMClient
from uraf.self_improving_agent import STaRAgent
from uraf.uncertainty_quantification import UncertaintyEstimator
from uraf.tree_of_thoughts import TreeOfThoughts, SearchStrategy

async def solve_with_confidence():
    llm = LLMClient()

    # 1. Use Tree-of-Thoughts to explore solutions
    tot = TreeOfThoughts(
        llm_client=llm,
        search_strategy=SearchStrategy.BEAM,
        max_depth=3
    )

    problem = "Design a sustainable transportation system for a city."
    tot_result = await tot.solve(problem)

    # 2. Improve the solution with STaR
    star = STaRAgent(llm_client=llm, max_iterations=3)
    improved = await star.solve_with_improvement(
        problem=problem,
        ground_truth=None
    )

    # 3. Estimate uncertainty
    estimator = UncertaintyEstimator(llm_client=llm)
    uncertainty = await estimator.estimate_comprehensive_uncertainty(
        prompt=problem,
        num_samples=5
    )

    # 4. Present results
    print("=== ToT Exploration ===")
    print(f"Solution: {tot_result['solution']}")
    print(f"Nodes Explored: {tot_result['statistics']['total_nodes_explored']}")

    print("\n=== Self-Improvement ===")
    print(f"Best Quality: {improved['best_quality']:.3f}")
    print(f"Iterations: {improved['final_iteration']}")

    print("\n=== Uncertainty ===")
    print(f"Confidence: {uncertainty['confidence_score']:.2%}")
    print(f"Interpretation: {uncertainty['interpretation']}")
    print(f"Recommendation: {uncertainty['recommended_action']}")

# Run
import asyncio
asyncio.run(solve_with_confidence())
```

### RAG + Causal Reasoning

```python
from uraf.rag_system import AdvancedRAG, RetrievalMode
from uraf.causal_reasoning import CausalReasoner

async def analyze_with_rag():
    llm = LLMClient()

    # Setup RAG
    rag = AdvancedRAG(
        llm_client=llm,
        retrieval_mode=RetrievalMode.HYBRID
    )

    # Add causal documents
    documents = [
        "Climate change causes sea level rise.",
        "Deforestation leads to biodiversity loss.",
        "Renewable energy reduces carbon emissions.",
    ]
    rag.add_documents(documents)

    # Query
    query = "What are the effects of climate change?"
    rag_result = await rag.query(query, rerank=True)

    # Extract causal relationships from answer
    reasoner = CausalReasoner(llm_client=llm)
    causal_analysis = await reasoner.analyze_causality(
        text=rag_result['answer'],
        question=query
    )

    print(f"RAG Answer: {rag_result['answer']}")
    print(f"\nCausal Relationships Found: {causal_analysis['num_relationships']}")
    for rel in causal_analysis['causal_relationships']:
        print(f"  {rel['cause']} → {rel['effect']}")

asyncio.run(analyze_with_rag())
```

---

## Performance Considerations

### Memory Usage

- **SemanticUncertainty**: ~500MB for embedding model
- **AdvancedRAG**: ~1GB for embeddings + reranker
- **TreeOfThoughts**: Scales with `max_depth * branching_factor`

### Latency

- **STaR** (5 iterations): ~10-30 seconds
- **ToT** (depth 5, beam 3): ~20-60 seconds
- **RAG query** (hybrid + rerank): ~2-5 seconds
- **Uncertainty estimation** (5 samples): ~5-15 seconds

### Optimization Tips

1. **Use beam search** for ToT (faster than BFS/DFS)
2. **Cache embeddings** in RAG for repeated queries
3. **Reduce num_samples** in uncertainty estimation for speed
4. **Use hybrid retrieval** only when necessary (dense is faster)
5. **Limit max_iterations** in STaR for production use

---

## Configuration

All features support configuration via `pyproject.toml` or environment variables:

```toml
[tool.uraf.advanced]
# STaR settings
star_quality_threshold = 0.7
star_max_iterations = 5

# Uncertainty settings
uncertainty_num_samples = 5
uncertainty_similarity_threshold = 0.85

# ToT settings
tot_max_depth = 5
tot_branching_factor = 3
tot_beam_width = 3

# RAG settings
rag_retrieval_mode = "hybrid"
rag_top_k = 5
rag_rerank = true
```

---

## Research Papers

Full references available in `docs/ADVANCED_RESEARCH_2024_2025.md`:

1. **STaR**: Zelikman et al. (2022, extended 2024)
2. **Tree-of-Thoughts**: Yao et al. (2024)
3. **Semantic Uncertainty**: Kuhn et al. (2024)
4. **Conformal Prediction**: Angelopoulos et al. (2024)
5. **Self-RAG**: Asai et al. (2024)
6. **CRAG**: Yan et al. (2024)
7. **Pearl's Causality**: Applied to LLMs (2024)

---

## Next Steps

1. Read the [Advanced Research Papers](ADVANCED_RESEARCH_2024_2025.md) for deeper understanding
2. Run the [demo script](../examples/demo_advanced_features.py) to see features in action
3. Check [test examples](../tests/test_advanced_features.py) for more usage patterns
4. Explore integration with existing URAF features (PRM, Constitutional AI, etc.)

---

**Version**: 0.4.0
**Last Updated**: October 2025
**Maintainers**: URAF Team
