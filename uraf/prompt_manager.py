from loguru import logger

class PromptManager:
    """
    Handles advanced prompting techniques for structured reasoning and decision-making.
    - Integrates multiple reasoning strategies into URAF.
    - Standardizes prompts to enforce structured responses.
    - Dynamically selects the best reasoning pathways.
    """

    SYSTEM_PROMPT = """You are an advanced reasoning system employing multiple cognitive frameworks:

    🧠 Cognitive Architecture:
    1. Meta-Learning Framework
       - Dynamically adapt reasoning strategies
       - Learn from intermediate deductions
       - Refine approach based on solution quality

    2. Multi-Agent Simulation
       - Simulate diverse expert perspectives
       - Debate and critique solutions
       - Resolve conflicts through synthesis

    3. Structured Reasoning Pipeline
       - Decompose → Analyze → Synthesize → Validate
       - Generate multiple solution paths
       - Cross-validate through different frameworks

    4. Knowledge Integration
       - Connect with domain expertise
       - Apply relevant frameworks
       - Validate against known patterns

    5. Solution Optimization
       - Evaluate trade-offs quantitatively
       - Consider edge cases and limitations
       - Optimize for robustness and generality

    ### **📌 Response Format (Aligned with URAF & Alpaca)**
    *Understanding:* [Context and problem breakdown through chain of thought]  
    *Reasoning Pathway:* [Logical breakdown of the approach]  
    *Comparative Insights:* [Optional – Pros & Cons, trade-offs]  
    *Illustrative Example:* [Optional – A real-world analogy or code/math example]  
    *Final Synthesis:* [Validated output, optimized response]"""

    MULTI_PERSPECTIVE_CATEGORIES = [
        "Deductive Reasoning", "Inductive Reasoning", "Probabilistic Thinking", "Counterfactual Reasoning",
        "Scientific & Technical", "Mathematical & Statistical", "Legal & Policy", "Business & Strategy",
        "Economic & Financial", "Psychological & Behavioral", "Sociological & Cultural", "Ethical & Moral",
        "User Experience & Human-Centric", "First-Principles Thinking", "Systems & Complexity",
        "Optimization & Trade-offs"
    ]

    @staticmethod
    def determine_relevant_perspectives(prompt):
        """
        Asks the LLM to dynamically determine the most relevant perspectives for the given prompt.
        """
        perspective_prompt = (
            f"Analyze the following problem and determine the most relevant perspectives for reasoning:\n\n{prompt}\n\n"
            f"Choose from the following categories:\n- " + "\n- ".join(PromptManager.MULTI_PERSPECTIVE_CATEGORIES) +
            "\n\nReturn the most relevant perspectives in order of importance."
        )
        return perspective_prompt

    @classmethod
    def get_prompt_with_technique(cls, prompt, technique):
        """Applies advanced cognitive frameworks while maintaining token efficiency."""
        
        if technique == "tree-of-thoughts":
            reasoning_guide = """
            Apply Tree-of-Thoughts framework:
            1. Meta-cognitive decomposition
            2. Branch exploration:
               └─ A: First principles analysis
               └─ B: Pattern-based reasoning
               └─ C: Constraint satisfaction
            3. Cross-branch evaluation
            4. Solution synthesis

            Your response MUST follow the exact format with *Header:*"""
            
        elif technique == "self-consistency":
            reasoning_guide = """
            Apply Self-Consistency framework:
            1. Generate diverse solutions
            2. Cross-framework validation
            3. Consistency analysis
            4. Robust synthesis

            Your response MUST follow the exact format with *Header:*"""
            
        else:
            reasoning_guide = """
            Apply Meta-Learning framework:
            1. Problem characterization
            2. Framework selection
            3. Solution optimization

            Your response MUST follow the exact format with *Header:*"""

        formatted_prompt = f"""
        {PromptManager.SYSTEM_PROMPT}

        Task: {prompt}

        {reasoning_guide}"""

        return formatted_prompt
