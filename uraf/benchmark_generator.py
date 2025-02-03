import asyncio
from loguru import logger
from uraf.llm_client import LLMClient

class BenchmarkGenerator:
    """
    Uses an LLM to generate **new** benchmark questions dynamically for specific agent types.
    """

    SYSTEM_PROMPT = """You are an advanced benchmark architect specializing in cognitive assessment design.

    Core Assessment Frameworks:
    1. Multi-Step Critical Thinking
       - Complex logical deductions (BIG-Bench Hard)
       - Interconnected reasoning chains
       - Hidden edge cases and gotchas
       - Requires: Systematic problem decomposition

    2. Self-Correcting & Validation
       - Error detection scenarios (MATH-500)
       - Assumption validation challenges
       - Multi-path verification tasks
       - Requires: Metacognitive monitoring

    3. Multi-Perspective Analysis
       - Stakeholder conflict resolution
       - Ethical dilemma navigation
       - Cross-domain synthesis
       - Requires: Framework integration

    4. Strategic Decision-Making
       - Resource optimization under constraints
       - Risk-reward analysis
       - Long-term impact assessment
       - Requires: Trade-off quantification

    5. Autonomous Planning
       - Dynamic constraint satisfaction
       - Failure mode analysis
       - Recovery strategy design
       - Requires: Robust contingency planning

    Question Structure Requirements:
    1. Context: Rich, relevant background
    2. Challenge: Clear, testable objective
    3. Constraints: Explicit limitations
    4. Evaluation Criteria: Measurable outcomes
    5. Complexity Hooks: Strategic challenges

    Ensure Questions Are:
    - Novel: No standard patterns
    - Robust: Multiple valid approaches
    - Scalable: Extensible difficulty
    - Diagnostic: Reveals reasoning quality
    - Bounded: Clear success criteria"""

    def __init__(self, model="qwen2.5-7b-instruct-1m", api_url="http://localhost:1234/v1/completions"):
        self.llm = LLMClient(model=model, api_url=api_url)

    async def generate_new_question(self, agent_type):
        """Generates benchmark questions using cognitive frameworks."""
        
        agent_prompts = {
            "Multi-Step Critical Thinking Agent": """
                Design a question requiring:
                - Multiple logical deductions
                - Hidden dependencies
                - Error checking
                Focus: System analysis or algorithmic thinking
                
                Example: "Given a sequence of numbers, identify the underlying pattern and predict the next value."
                """,
                
            "Backtracking & Self-Correcting Agent": """
                Design a question requiring:
                - Multiple solution paths
                - Error detection
                - Recovery strategies
                Focus: Proofs or optimization
                
                Example: "Find all possible ways to arrange N queens on an NxN chessboard without any queen threatening another."
                """,
                
            "Multi-Perspective Analysis Agent": """
                Design a question requiring:
                - Multiple viewpoints
                - Trade-off analysis
                - Framework synthesis
                Focus: Policy or ethics
                
                Example: "Analyze a complex policy decision considering economic, social, and environmental impacts."
                """,
                
            "Decision-Making Agent": """
                Design a question requiring:
                - Trade-off analysis
                - Risk assessment
                - Resource allocation
                Focus: Strategy or design
                
                Example: "Optimize a portfolio allocation given risk constraints and return objectives."
                """,
                
            "Autonomous Planning Agent": """
                Design a question requiring:
                - Constraint handling
                - Contingency planning
                - Failure recovery
                Focus: Planning or architecture
                
                Example: "Design a robust system architecture that handles component failures gracefully."
                """
        }

        # Generate question based on agent type
        prompt = f"""You are an expert in cognitive assessment design.

Task: Generate a challenging benchmark question for the {agent_type}.

Requirements:
{agent_prompts.get(agent_type, "Design a challenging reasoning question.")}

Response Format:
*Understanding:* [Context and problem breakdown]
*Reasoning Pathway:* [Step-by-step solution approach]
*Final Synthesis:* [Clear, concise answer]

Your response must be:
1. Clear and unambiguous
2. Challenging but solvable
3. Focused on reasoning over recall
4. Structured with the exact headers shown above
"""

        logger.info(f"📝 Generating benchmark for {agent_type}...")
        
        response = await self.llm.query(prompt)

        if response and "summary" in response:
            logger.info(f"✅ Generated: {response['summary']}")
            return response['summary']
        
        logger.error("❌ Generation failed")
        return "Error: Failed to generate question."
