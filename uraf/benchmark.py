import random
from loguru import logger

class Benchmark:
    """
    Generates benchmark questions for evaluating different types of agents in URAF.
    """

    # Mapping agent types to reasoning techniques
    AGENT_TECHNIQUE_MAP = {
        "Multi-Step Critical Thinking Agent": "tree-of-thoughts",
        "Backtracking & Self-Correcting Agent": "self-critique",
        "Multi-Perspective Analysis Agent": "multi-perspective",
        "Decision-Making Agent": "self-consistency",
        "Autonomous Planning Agent": "default"
    }

    # Mapping agent types to relevant benchmarks (Including ARC & MMLU-Advanced)
    AGENT_BENCHMARK_MAP = {
        "Multi-Step Critical Thinking Agent": ["BIG-Bench Hard (BBH)", "ARC (AI2 Reasoning Challenge)"],
        "Backtracking & Self-Correcting Agent": ["MATH (Math 500)", "PhysicsQA"],
        "Multi-Perspective Analysis Agent": ["TruthfulQA", "LawBench", "MMLU-Advanced"],
        "Decision-Making Agent": ["BBH (BigBench Hard Subset)", "MMLU (Advanced Topics)"],
        "Autonomous Planning Agent": ["HumanEval", "MBPP"]
    }

    # Expanded Benchmark Question Bank (Including ARC & MMLU-Advanced)
    BENCHMARK_QUESTIONS = {
        "MMLU (Advanced Topics)": [
            "Explain the key principles of Bayesian inference and their application in real-world decision-making.",
            "Describe the role of entropy in information theory and how it impacts data compression."
        ],
        "BIG-Bench Hard (BBH)": [
            "Given a logical rule set, determine the most probable conclusion.",
            "Solve this multi-step reasoning puzzle using deductive logic."
        ],
        "MATH (Math 500)": [
            "Solve for x: If 2x + 3y = 7 and x - y = 2, find the values of x and y.",
            "Compute the definite integral of (3x^2 + 2x - 5) dx."
        ],
        "PhysicsQA": [
            "Explain the relationship between energy and momentum in classical mechanics.",
            "Given a projectile motion equation, determine the optimal launch angle for maximum range."
        ],
        "BBH (BigBench Hard Subset)": [
            "How should a startup balance growth and profitability when seeking funding?",
            "Analyze the strategic trade-offs between vertical and horizontal scaling in cloud infrastructure."
        ],
        "LawBench": [
            "Explain how case law precedents influence judicial decisions.",
            "How does international law address conflicts between sovereignty and human rights?"
        ],
        "HumanEval": [
            "Write a Python function that returns the nth Fibonacci number.",
            "Implement an efficient sorting algorithm that operates in O(n log n) complexity."
        ],
        "TruthfulQA": [
            "What are the ethical implications of AI making autonomous medical diagnoses?",
            "How can policymakers balance free speech and misinformation in digital platforms?"
        ],
        "ARC (AI2 Reasoning Challenge)": [
            "What is the next number in the pattern: 2, 6, 12, 20, ...?",
            "Given a sequence of geometric transformations, determine the final shape and position."
        ],
        "MMLU-Advanced": [
            "Discuss the philosophical implications of the Turing Test in the age of advanced AI.",
            "Analyze the economic impact of machine learning-driven automation in developing nations."
        ]
    }

    @classmethod
    def get_agent_types(cls):
        """Returns the list of agent types URAF evaluates."""
        return list(cls.AGENT_TECHNIQUE_MAP.keys())

    @classmethod
    def get_technique_for_agent(cls, agent_type):
        """Returns the reasoning technique mapped to an agent type."""
        technique = cls.AGENT_TECHNIQUE_MAP.get(agent_type, "default")
        logger.info(f"Selected technique '{technique}' for agent type: {agent_type}")
        return technique

    @classmethod
    def get_benchmarks_for_agent(cls, agent_type):
        """Returns the relevant benchmarks mapped to an agent type."""
        return cls.AGENT_BENCHMARK_MAP.get(agent_type, [])

    @classmethod
    def generate(cls, agent_type):
        """
        Generates a benchmark question for a given agent type.
        If no valid agent type is provided, it returns a default message.
        """
        benchmarks = cls.get_benchmarks_for_agent(agent_type)
        if not benchmarks:
            logger.warning(f"No benchmarks found for agent type '{agent_type}'. Returning default question.")
            return "No valid benchmark question available for this agent type."

        selected_benchmark = random.choice(benchmarks)
        question_pool = cls.BENCHMARK_QUESTIONS.get(selected_benchmark, [])

        if not question_pool:
            logger.warning(f"No questions found for benchmark '{selected_benchmark}'. Returning fallback question.")
            return "No valid question available for this benchmark."

        question = random.choice(question_pool)
        logger.info(f"Generated question from '{selected_benchmark}' for '{agent_type}': {question}")
        return question
