"""
Causal Reasoning Module

Based on Pearl's Causal Hierarchy and recent applications to LLMs (2024).

Implements three levels of causal reasoning:
1. Association (Seeing): P(Y|X) - Observational patterns
2. Intervention (Doing): P(Y|do(X)) - Effects of actions
3. Counterfactuals (Imagining): P(Y_x|X',Y') - What if scenarios

Enables LLMs to reason about causality, not just correlation.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from uraf.llm_client import LLMClient


class CausalLevel(Enum):
    """Pearl's three levels of causation."""

    ASSOCIATION = 1  # Seeing: P(Y|X)
    INTERVENTION = 2  # Doing: P(Y|do(X))
    COUNTERFACTUAL = 3  # Imagining: P(Y_x|X',Y')


@dataclass
class CausalRelationship:
    """A causal relationship between variables."""

    cause: str
    effect: str
    confidence: float  # 0-1
    mechanism: str  # Explanation of how cause leads to effect
    confounders: list[str]  # Potential confounding variables
    evidence: str  # Supporting evidence


@dataclass
class CausalGraph:
    """A directed acyclic graph (DAG) of causal relationships."""

    nodes: list[str]  # Variables
    edges: list[tuple[str, str]]  # (cause, effect) pairs
    relationships: dict[tuple[str, str], CausalRelationship]


class CausalReasoner:
    """Causal reasoning system for LLMs."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize causal reasoner.

        Args:
            llm_client: LLM client for reasoning
        """
        self.llm = llm_client
        logger.info("Initialized CausalReasoner")

    async def analyze_causality(self, text: str, question: str | None = None) -> dict[str, Any]:
        """
        Analyze causal relationships in text.

        Args:
            text: Text to analyze
            question: Optional specific causal question

        Returns:
            Causal analysis with relationships and graph
        """
        logger.info("Analyzing causal relationships")

        # Extract causal relationships
        relationships = await self._extract_causal_relationships(text)

        # Build causal graph
        causal_graph = self._build_causal_graph(relationships)

        # Analyze question if provided
        question_analysis = None
        if question:
            question_analysis = await self._analyze_causal_question(question, causal_graph, text)

        return {
            "causal_relationships": [
                {
                    "cause": r.cause,
                    "effect": r.effect,
                    "confidence": r.confidence,
                    "mechanism": r.mechanism,
                    "confounders": r.confounders,
                }
                for r in relationships
            ],
            "causal_graph": {"nodes": causal_graph.nodes, "edges": causal_graph.edges},
            "num_relationships": len(relationships),
            "question_analysis": question_analysis,
        }

    async def _extract_causal_relationships(self, text: str) -> list[CausalRelationship]:
        """Extract causal relationships from text."""
        extraction_prompt = f"""Analyze the following text and identify all causal relationships.

Text: {text}

For each causal relationship, specify:
1. The cause
2. The effect
3. The causal mechanism (how the cause leads to the effect)
4. Potential confounding variables
5. Confidence (0-1)

Format each relationship as:
CAUSE: [cause]
EFFECT: [effect]
MECHANISM: [mechanism]
CONFOUNDERS: [confounders]
CONFIDENCE: [0-1]
---

Causal relationships:"""

        response = await self.llm.query(extraction_prompt)

        # Parse relationships
        relationships = self._parse_relationships(response)

        logger.debug(f"Extracted {len(relationships)} causal relationships")

        return relationships

    def _parse_relationships(self, response: str) -> list[CausalRelationship]:
        """Parse causal relationships from LLM response."""
        relationships = []

        # Split by separator
        blocks = response.split("---")

        for block in blocks:
            if not block.strip():
                continue

            # Extract fields using regex
            cause_match = re.search(r"CAUSE:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            effect_match = re.search(r"EFFECT:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            mechanism_match = re.search(r"MECHANISM:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            confounders_match = re.search(r"CONFOUNDERS:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", block, re.IGNORECASE)

            if cause_match and effect_match:
                cause = cause_match.group(1).strip()
                effect = effect_match.group(1).strip()
                mechanism = mechanism_match.group(1).strip() if mechanism_match else ""
                confounders_str = confounders_match.group(1).strip() if confounders_match else ""
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5

                # Parse confounders (comma-separated)
                confounders = [c.strip() for c in confounders_str.split(",") if c.strip()]

                relationships.append(
                    CausalRelationship(
                        cause=cause,
                        effect=effect,
                        confidence=confidence,
                        mechanism=mechanism,
                        confounders=confounders,
                        evidence="",
                    )
                )

        return relationships

    def _build_causal_graph(self, relationships: list[CausalRelationship]) -> CausalGraph:
        """Build causal graph from relationships."""
        nodes = set()
        edges = []
        relationship_map = {}

        for rel in relationships:
            nodes.add(rel.cause)
            nodes.add(rel.effect)
            edge = (rel.cause, rel.effect)
            edges.append(edge)
            relationship_map[edge] = rel

        return CausalGraph(nodes=list(nodes), edges=edges, relationships=relationship_map)

    async def _analyze_causal_question(self, question: str, graph: CausalGraph, context: str) -> dict[str, Any]:
        """Analyze a causal question."""
        # Determine causal level
        level = self._determine_causal_level(question)

        # Answer based on level
        if level == CausalLevel.ASSOCIATION:
            answer = await self._answer_association(question, graph, context)
        elif level == CausalLevel.INTERVENTION:
            answer = await self._answer_intervention(question, graph, context)
        elif level == CausalLevel.COUNTERFACTUAL:
            answer = await self._answer_counterfactual(question, graph, context)
        else:
            answer = "Unable to determine causal level"

        return {"question": question, "causal_level": level.name, "answer": answer}

    def _determine_causal_level(self, question: str) -> CausalLevel:
        """Determine which level of Pearl's hierarchy the question belongs to."""
        q_lower = question.lower()

        # Counterfactual indicators
        if any(indicator in q_lower for indicator in ["what if", "would have", "had", "instead", "counterfactual"]):
            return CausalLevel.COUNTERFACTUAL

        # Intervention indicators
        if any(
            indicator in q_lower
            for indicator in ["if we", "what would happen if", "effect of", "cause", "make", "do", "intervention"]
        ):
            return CausalLevel.INTERVENTION

        # Association (default)
        return CausalLevel.ASSOCIATION

    async def _answer_association(self, question: str, graph: CausalGraph, context: str) -> str:
        """Answer association-level question (observation)."""
        prompt = f"""Context: {context}

Causal relationships identified:
{self._format_graph_for_prompt(graph)}

Question (Association level - about observed patterns): {question}

Answer based on the observed patterns and correlations:"""

        return await self.llm.query(prompt)

    async def _answer_intervention(self, question: str, graph: CausalGraph, context: str) -> str:
        """Answer intervention-level question (doing)."""
        prompt = f"""Context: {context}

Causal graph:
{self._format_graph_for_prompt(graph)}

Question (Intervention level - about effects of actions): {question}

Use the causal graph to reason about what would happen if we perform the intervention. Consider:
1. Direct effects of the intervention
2. Indirect effects through causal chains
3. What remains unchanged

Answer:"""

        return await self.llm.query(prompt)

    async def _answer_counterfactual(self, question: str, graph: CausalGraph, context: str) -> str:
        """Answer counterfactual question (imagining)."""
        prompt = f"""Context: {context}

Causal graph:
{self._format_graph_for_prompt(graph)}

Question (Counterfactual - about alternative histories): {question}

Reason about the counterfactual scenario:
1. Identify what would have been different
2. Trace through the causal graph to find all affected variables
3. Consider what would remain the same (unaffected by the change)

Answer:"""

        return await self.llm.query(prompt)

    def _format_graph_for_prompt(self, graph: CausalGraph) -> str:
        """Format causal graph for prompt."""
        lines = []
        for cause, effect in graph.edges:
            rel = graph.relationships.get((cause, effect))
            if rel:
                lines.append(f"- {cause} → {effect} (confidence: {rel.confidence:.2f})")
                if rel.mechanism:
                    lines.append(f"  Mechanism: {rel.mechanism}")

        return "\n".join(lines)

    async def intervention_reasoning(self, scenario: str, intervention: str) -> dict[str, Any]:
        """
        Reason about the effects of an intervention.

        Args:
            scenario: Description of the scenario
            intervention: Proposed intervention

        Returns:
            Analysis of intervention effects
        """
        logger.info(f"Analyzing intervention: {intervention}")

        # Extract causal structure
        relationships = await self._extract_causal_relationships(scenario)
        graph = self._build_causal_graph(relationships)

        # Identify affected variables
        affected = self._identify_affected_variables(intervention, graph)

        # Predict effects
        effects_prompt = f"""Scenario: {scenario}

Causal structure:
{self._format_graph_for_prompt(graph)}

Intervention: {intervention}

Variables potentially affected: {", ".join(affected)}

Predict the effects of this intervention:
1. Immediate direct effects
2. Downstream causal effects
3. Unintended consequences
4. Variables that remain unchanged

Analysis:"""

        analysis = await self.llm.query(effects_prompt)

        return {
            "intervention": intervention,
            "affected_variables": affected,
            "causal_graph": {"nodes": graph.nodes, "edges": graph.edges},
            "predicted_effects": analysis,
            "num_affected_variables": len(affected),
        }

    def _identify_affected_variables(self, intervention: str, graph: CausalGraph) -> list[str]:
        """Identify variables affected by intervention."""
        affected = set()

        # Simple keyword matching to identify target variable
        intervention_lower = intervention.lower()

        for node in graph.nodes:
            if node.lower() in intervention_lower:
                # Found target - add all descendants
                affected.add(node)
                affected.update(self._get_descendants(node, graph))

        return list(affected)

    def _get_descendants(self, node: str, graph: CausalGraph) -> set[str]:
        """Get all descendants of a node in the causal graph."""
        descendants = set()
        to_visit = [node]

        while to_visit:
            current = to_visit.pop()

            # Find children
            for cause, effect in graph.edges:
                if cause == current and effect not in descendants:
                    descendants.add(effect)
                    to_visit.append(effect)

        return descendants

    async def counterfactual_reasoning(
        self, scenario: str, actual_outcome: str, counterfactual_condition: str
    ) -> dict[str, Any]:
        """
        Reason about counterfactual scenarios.

        Args:
            scenario: Original scenario
            actual_outcome: What actually happened
            counterfactual_condition: What if this had been different

        Returns:
            Counterfactual analysis
        """
        logger.info(f"Counterfactual reasoning: {counterfactual_condition}")

        counterfactual_prompt = f"""Scenario: {scenario}

What actually happened: {actual_outcome}

Counterfactual: {counterfactual_condition}

Reason through what would have happened differently:

1. **Identify the change**: What is different in the counterfactual world?

2. **Trace causal effects**: How would this change propagate through causal relationships?

3. **Determine counterfactual outcome**: What would have happened instead?

4. **Confidence**: How confident are you in this counterfactual reasoning? (0-1)

Analysis:"""

        analysis = await self.llm.query(counterfactual_prompt)

        # Extract confidence if provided
        confidence_match = re.search(r"confidence[:\s]*([\d.]+)", analysis, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        return {
            "scenario": scenario,
            "actual_outcome": actual_outcome,
            "counterfactual_condition": counterfactual_condition,
            "counterfactual_analysis": analysis,
            "confidence": confidence,
        }


class CausalDiscovery:
    """
    Causal discovery: Infer causal structure from data/text.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize causal discovery.

        Args:
            llm_client: LLM client
        """
        self.llm = llm_client
        logger.info("Initialized CausalDiscovery")

    async def discover_causal_structure(self, observations: list[str]) -> dict[str, Any]:
        """
        Discover causal structure from observations.

        Args:
            observations: List of observations

        Returns:
            Inferred causal structure
        """
        logger.info(f"Discovering causal structure from {len(observations)} observations")

        observations_text = "\n".join([f"{i + 1}. {obs}" for i, obs in enumerate(observations)])

        discovery_prompt = f"""Given these observations, infer the underlying causal structure:

Observations:
{observations_text}

Identify:
1. What are the key variables?
2. Which variables likely cause which others?
3. What is the causal graph (DAG) structure?
4. Are there potential confounders?
5. What experiments/interventions would help confirm the causal structure?

Analysis:"""

        analysis = await self.llm.query(discovery_prompt)

        return {"observations": observations, "num_observations": len(observations), "causal_structure": analysis}

    async def suggest_interventions(self, hypothesis: str, current_knowledge: str) -> dict[str, Any]:
        """
        Suggest interventions to test causal hypotheses.

        Args:
            hypothesis: Causal hypothesis to test
            current_knowledge: Current understanding

        Returns:
            Suggested interventions
        """
        logger.info(f"Suggesting interventions for hypothesis: {hypothesis}")

        intervention_prompt = f"""Causal hypothesis: {hypothesis}

Current knowledge: {current_knowledge}

Design interventions to test this causal hypothesis:

1. **Randomized intervention**: What should we randomly assign/manipulate?

2. **Measurements**: What outcomes should we measure?

3. **Control variables**: What should we hold constant?

4. **Expected results**: What would confirm vs. refute the hypothesis?

5. **Alternative explanations**: What else could explain the observations?

Intervention design:"""

        design = await self.llm.query(intervention_prompt)

        return {"hypothesis": hypothesis, "intervention_design": design}


def detect_causal_language(text: str) -> dict[str, Any]:
    """
    Detect causal language indicators in text.

    Args:
        text: Text to analyze

    Returns:
        Detected causal indicators
    """
    causal_indicators = {
        "strong_causal": [
            r"\bcause[sd]?\b",
            r"\bleads? to\b",
            r"\bresults? in\b",
            r"\bproduces?\b",
            r"\btriggers?\b",
            r"\bgenerates?\b",
        ],
        "weak_causal": [
            r"\baffects?\b",
            r"\binfluences?\b",
            r"\bimpacts?\b",
            r"\bcontributes? to\b",
            r"\bassociated with\b",
        ],
        "counterfactual": [r"\bwhat if\b", r"\bwould have\b", r"\bhad .+ been\b", r"\bcounterfactual\b"],
        "intervention": [r"\bif we\b", r"\bintervene\b", r"\bmanipulate\b", r"\bdo\(.+\)", r"\btreat(ment)?\b"],
    }

    detected = {}
    for category, patterns in causal_indicators.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)

        detected[category] = {"count": len(matches), "examples": matches[:5]}

    total_indicators = sum(cat["count"] for cat in detected.values())

    return {
        "indicators_by_category": detected,
        "total_indicators": total_indicators,
        "has_causal_language": total_indicators > 0,
    }
