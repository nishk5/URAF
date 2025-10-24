"""
Tree-of-Thoughts (ToT) Reasoning

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2024)

ToT explores multiple reasoning paths as a tree, enabling:
- Exploration of alternative approaches
- Backtracking from dead ends
- Lookahead and planning
- Better performance on complex reasoning tasks
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from uraf.llm_client import LLMClient
from uraf.process_reward_model import ProcessRewardModel


class SearchStrategy(Enum):
    """Search strategies for ToT."""

    BFS = "breadth_first"  # Explore all nodes at current depth
    DFS = "depth_first"  # Explore one path to end before backtracking
    BEAM = "beam_search"  # Keep top-k best paths at each level
    MCTS = "monte_carlo"  # Monte Carlo Tree Search with exploration-exploitation


@dataclass
class ThoughtNode:
    """A node in the tree of thoughts."""

    thought: str  # The thought/reasoning step text
    depth: int  # Depth in tree (0 = root)
    parent: "ThoughtNode | None"  # Parent node
    children: list["ThoughtNode"]  # Child nodes
    value: float  # Estimated value/quality of this thought
    visits: int  # Number of times visited (for MCTS)
    is_solution: bool  # Whether this is a complete solution
    metadata: dict[str, Any]  # Additional metadata

    def __post_init__(self):
        """Initialize empty children list if not provided."""
        if not hasattr(self, "children") or self.children is None:
            self.children = []

    def get_path_from_root(self) -> list["ThoughtNode"]:
        """Get full path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_reasoning_chain(self) -> str:
        """Get reasoning chain as text."""
        path = self.get_path_from_root()
        return "\n".join([f"Step {i + 1}: {node.thought}" for i, node in enumerate(path) if node.thought])


class TreeOfThoughts:
    """Tree-of-Thoughts reasoning system."""

    def __init__(
        self,
        llm_client: LLMClient,
        prm: ProcessRewardModel | None = None,
        search_strategy: SearchStrategy = SearchStrategy.BEAM,
        max_depth: int = 5,
        branching_factor: int = 3,
        beam_width: int = 3,
    ):
        """
        Initialize Tree-of-Thoughts.

        Args:
            llm_client: LLM client for generating thoughts
            prm: Process reward model for evaluating thoughts
            search_strategy: Search strategy to use
            max_depth: Maximum tree depth
            branching_factor: Number of child thoughts to generate per node
            beam_width: Beam width for beam search
        """
        self.llm = llm_client
        self.prm = prm or ProcessRewardModel()
        self.search_strategy = search_strategy
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width

        logger.info(f"Initialized ToT with {search_strategy.value} search")

    async def solve(self, problem: str, verbose: bool = False) -> dict[str, Any]:
        """
        Solve problem using Tree-of-Thoughts.

        Args:
            problem: Problem to solve
            verbose: Print search progress

        Returns:
            Best solution and search statistics
        """
        logger.info(f"ToT solving: {problem[:100]}...")

        # Create root node
        root = ThoughtNode(
            thought="",  # Root has empty thought
            depth=0,
            parent=None,
            children=[],
            value=0.0,
            visits=0,
            is_solution=False,
            metadata={"problem": problem},
        )

        # Perform tree search based on strategy
        if self.search_strategy == SearchStrategy.BFS:
            best_solution = await self._breadth_first_search(root, problem, verbose)
        elif self.search_strategy == SearchStrategy.DFS:
            best_solution = await self._depth_first_search(root, problem, verbose)
        elif self.search_strategy == SearchStrategy.BEAM:
            best_solution = await self._beam_search(root, problem, verbose)
        elif self.search_strategy == SearchStrategy.MCTS:
            best_solution = await self._monte_carlo_tree_search(root, problem, verbose)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")

        # Gather statistics
        stats = self._gather_statistics(root, best_solution)

        return {
            "solution": best_solution.get_reasoning_chain() if best_solution else None,
            "final_answer": best_solution.thought if best_solution and best_solution.is_solution else None,
            "value": best_solution.value if best_solution else 0.0,
            "statistics": stats,
        }

    async def _breadth_first_search(self, root: ThoughtNode, problem: str, verbose: bool) -> ThoughtNode | None:
        """BFS: Explore all nodes at each depth level."""
        from collections import deque

        queue = deque([root])
        best_solution = None
        best_value = -float("inf")
        nodes_explored = 0

        while queue:
            node = queue.popleft()
            nodes_explored += 1

            if verbose:
                logger.info(f"BFS exploring depth {node.depth}, node {nodes_explored}")

            # Check if solution
            if node.is_solution:
                if node.value > best_value:
                    best_solution = node
                    best_value = node.value
                continue

            # Don't expand beyond max depth
            if node.depth >= self.max_depth:
                continue

            # Generate and evaluate child thoughts
            children = await self._generate_children(node, problem)

            # Add to queue
            queue.extend(children)

        return best_solution

    async def _depth_first_search(self, root: ThoughtNode, problem: str, verbose: bool) -> ThoughtNode | None:
        """DFS: Explore one path to the end before backtracking."""
        best_solution = None
        best_value = -float("inf")
        nodes_explored = [0]  # Use list to allow modification in nested function

        async def dfs_recursive(node: ThoughtNode) -> None:
            nonlocal best_solution, best_value

            nodes_explored[0] += 1

            if verbose and nodes_explored[0] % 10 == 0:
                logger.info(f"DFS explored {nodes_explored[0]} nodes, depth {node.depth}")

            # Check if solution
            if node.is_solution:
                if node.value > best_value:
                    best_solution = node
                    best_value = node.value
                return

            # Don't expand beyond max depth
            if node.depth >= self.max_depth:
                return

            # Generate children
            children = await self._generate_children(node, problem)

            # Recursively explore children
            for child in children:
                await dfs_recursive(child)

        await dfs_recursive(root)

        return best_solution

    async def _beam_search(self, root: ThoughtNode, problem: str, verbose: bool) -> ThoughtNode | None:
        """Beam search: Keep top-k best paths at each level."""
        current_beam = [root]
        best_solution = None
        best_value = -float("inf")
        nodes_explored = 0

        for depth in range(self.max_depth):
            if verbose:
                logger.info(f"Beam search depth {depth}, beam size {len(current_beam)}")

            # Generate children for all nodes in beam
            all_children = []
            for node in current_beam:
                children = await self._generate_children(node, problem)
                all_children.extend(children)
                nodes_explored += len(children)

            if not all_children:
                break

            # Check for solutions
            solutions = [c for c in all_children if c.is_solution]
            for sol in solutions:
                if sol.value > best_value:
                    best_solution = sol
                    best_value = sol.value

            # Keep top beam_width non-solution nodes for next level
            non_solutions = [c for c in all_children if not c.is_solution]
            non_solutions.sort(key=lambda x: x.value, reverse=True)
            current_beam = non_solutions[: self.beam_width]

            if not current_beam:
                break

        if verbose:
            logger.info(f"Beam search completed, explored {nodes_explored} nodes")

        return best_solution

    async def _monte_carlo_tree_search(self, root: ThoughtNode, problem: str, verbose: bool) -> ThoughtNode | None:
        """MCTS with UCB1 for exploration-exploitation."""
        num_simulations = 50
        exploration_constant = 1.414  # sqrt(2)

        best_solution = None
        best_value = -float("inf")

        for sim in range(num_simulations):
            if verbose and sim % 10 == 0:
                logger.info(f"MCTS simulation {sim}/{num_simulations}")

            # Selection: Select most promising node using UCB1
            node = self._select_node_ucb1(root, exploration_constant)

            # Expansion: Generate children if not terminal
            if not node.is_solution and node.depth < self.max_depth:
                children = await self._generate_children(node, problem)
                if children:
                    # Select random child for rollout
                    node = children[0]

            # Simulation: Evaluate node
            value = node.value

            # Backpropagation: Update values and visits
            self._backpropagate(node, value)

            # Track best solution
            if node.is_solution and value > best_value:
                best_solution = node
                best_value = value

        return best_solution

    def _select_node_ucb1(self, root: ThoughtNode, c: float) -> ThoughtNode:
        """Select node using UCB1 (Upper Confidence Bound)."""
        node = root

        while node.children and not node.is_solution and node.depth < self.max_depth:
            # UCB1 formula: value + c * sqrt(log(parent_visits) / child_visits)
            ucb_scores = []
            for child in node.children:
                if child.visits == 0:
                    # Prioritize unvisited nodes
                    ucb_scores.append(float("inf"))
                else:
                    exploitation = child.value
                    exploration = c * math.sqrt(math.log(node.visits + 1) / child.visits)
                    ucb_scores.append(exploitation + exploration)

            # Select child with highest UCB score
            best_idx = ucb_scores.index(max(ucb_scores))
            node = node.children[best_idx]

        return node

    def _backpropagate(self, node: ThoughtNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            # Update value (moving average)
            current.value = (current.value * (current.visits - 1) + value) / current.visits
            current = current.parent

    async def _generate_children(self, parent: ThoughtNode, problem: str) -> list[ThoughtNode]:
        """Generate child thoughts for a node."""
        # Get current reasoning chain
        current_chain = parent.get_reasoning_chain()

        # Generate multiple alternative next thoughts
        thoughts = await self._generate_candidate_thoughts(problem, current_chain, self.branching_factor)

        # Evaluate each thought
        children = []
        for thought_text in thoughts:
            # Evaluate thought quality
            value = await self._evaluate_thought(thought_text, current_chain, problem)

            # Check if this is a solution
            is_solution = self._is_complete_solution(thought_text, parent.depth + 1)

            # Create child node
            child = ThoughtNode(
                thought=thought_text,
                depth=parent.depth + 1,
                parent=parent,
                children=[],
                value=value,
                visits=0,
                is_solution=is_solution,
                metadata={},
            )

            children.append(child)

        # Add to parent
        parent.children.extend(children)

        return children

    async def _generate_candidate_thoughts(self, problem: str, current_chain: str, num_candidates: int) -> list[str]:
        """Generate candidate next thoughts."""
        prompt = f"""Problem: {problem}

Current reasoning:
{current_chain}

Generate {num_candidates} alternative next reasoning steps. Each should be a distinct approach to making progress.

Format:
Thought 1: [reasoning step]
Thought 2: [reasoning step]
Thought 3: [reasoning step]
"""

        response = await self.llm.query(prompt)

        # Parse thoughts
        thoughts = []
        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("Thought"):
                # Extract thought text after colon
                parts = line.split(":", 1)
                if len(parts) == 2:
                    thoughts.append(parts[1].strip())

        # Fallback if parsing fails
        if not thoughts:
            thoughts = [response.strip()]

        return thoughts[:num_candidates]

    async def _evaluate_thought(self, thought: str, current_chain: str, problem: str) -> float:
        """Evaluate quality of a thought."""
        # Use PRM to evaluate
        full_reasoning = current_chain + "\n" + thought if current_chain else thought

        prm_result = self.prm.evaluate_reasoning_chain(full_reasoning, problem)

        return prm_result["final_prm_score"]

    def _is_complete_solution(self, thought: str, depth: int) -> bool:
        """Check if thought represents a complete solution."""
        # Heuristic: Contains "final answer" or "conclusion" or reached max depth
        thought_lower = thought.lower()
        has_conclusion = any(
            marker in thought_lower for marker in ["final answer", "conclusion", "therefore the answer", "solution is"]
        )

        return has_conclusion or depth >= self.max_depth

    def _gather_statistics(self, root: ThoughtNode, best_solution: ThoughtNode | None) -> dict[str, Any]:
        """Gather statistics about the search."""
        total_nodes = 0
        total_values = []
        max_depth_reached = 0

        def traverse(node: ThoughtNode):
            nonlocal total_nodes, max_depth_reached

            total_nodes += 1
            total_values.append(node.value)
            max_depth_reached = max(max_depth_reached, node.depth)

            for child in node.children:
                traverse(child)

        traverse(root)

        return {
            "total_nodes_explored": total_nodes,
            "max_depth_reached": max_depth_reached,
            "avg_node_value": sum(total_values) / len(total_values) if total_values else 0.0,
            "best_solution_depth": best_solution.depth if best_solution else None,
            "search_strategy": self.search_strategy.value,
        }


class GraphOfThoughts:
    """
    Graph-of-Thoughts (GoT): Extension of ToT where thoughts can merge and split.

    Represents reasoning as a DAG instead of a tree.
    """

    def __init__(self, llm_client: LLMClient, prm: ProcessRewardModel | None = None):
        """
        Initialize Graph-of-Thoughts.

        Args:
            llm_client: LLM client
            prm: Process reward model
        """
        self.llm = llm_client
        self.prm = prm or ProcessRewardModel()
        self.nodes: list[ThoughtNode] = []
        self.edges: dict[int, list[int]] = defaultdict(list)  # node_id -> [child_ids]

        logger.info("Initialized GraphOfThoughts (GoT)")

    async def solve_with_merge(self, problem: str, num_initial_paths: int = 3) -> dict[str, Any]:
        """
        Solve using GoT with path merging.

        Args:
            problem: Problem to solve
            num_initial_paths: Number of initial reasoning paths

        Returns:
            Solution with merged reasoning
        """
        logger.info(f"GoT solving with {num_initial_paths} initial paths")

        # Generate multiple initial reasoning paths
        initial_nodes = await self._generate_initial_paths(problem, num_initial_paths)

        # Identify opportunities to merge paths
        merged_node = await self._merge_paths(initial_nodes, problem)

        # Continue reasoning from merged node
        final_solution = await self._continue_from_merge(merged_node, problem)

        return {
            "solution": final_solution["reasoning"],
            "answer": final_solution["answer"],
            "merge_points": 1,  # Simplified
            "initial_paths": len(initial_nodes),
        }

    async def _generate_initial_paths(self, problem: str, num_paths: int) -> list[ThoughtNode]:
        """Generate diverse initial reasoning paths."""
        nodes = []

        for i in range(num_paths):
            prompt = f"""Problem: {problem}

Provide an initial reasoning step to approach this problem. Take approach {i + 1} (be creative and different from common approaches).

Reasoning step:"""

            response = await self.llm.query(prompt)

            # Evaluate
            value = self.prm.evaluate_step_correctness(response.strip())["correctness_score"]

            node = ThoughtNode(
                thought=response.strip(),
                depth=1,
                parent=None,
                children=[],
                value=value,
                visits=1,
                is_solution=False,
                metadata={},
            )

            nodes.append(node)

        return nodes

    async def _merge_paths(self, nodes: list[ThoughtNode], problem: str) -> ThoughtNode:
        """Merge multiple reasoning paths into one."""
        thoughts = [node.thought for node in nodes]
        thoughts_text = "\n\n".join([f"Path {i + 1}: {t}" for i, t in enumerate(thoughts)])

        merge_prompt = f"""Problem: {problem}

Multiple reasoning approaches have been explored:

{thoughts_text}

Synthesize these approaches into a single, coherent reasoning step that combines the best insights from each path.

Synthesized reasoning:"""

        merged = await self.llm.query(merge_prompt)

        # Evaluate merged thought
        value = self.prm.evaluate_step_correctness(merged.strip())["correctness_score"]

        merged_node = ThoughtNode(
            thought=merged.strip(),
            depth=2,
            parent=None,
            children=[],
            value=value,
            visits=1,
            is_solution=False,
            metadata={"merged_from": len(nodes)},
        )

        return merged_node

    async def _continue_from_merge(self, node: ThoughtNode, problem: str) -> dict[str, str]:
        """Continue reasoning from merged node to solution."""
        prompt = f"""Problem: {problem}

Current reasoning:
{node.thought}

Complete the reasoning to reach a final answer.

Final answer:"""

        response = await self.llm.query(prompt)

        return {"reasoning": node.thought + "\n\n" + response, "answer": response.strip()}
