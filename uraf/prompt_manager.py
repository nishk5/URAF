import guidance
from loguru import logger

class PromptManager:
    """
    Enforces structured reasoning responses using guidance.
    """

    @staticmethod
    def base_template(task):
        return guidance('''
        {{#system}}
        You are an advanced reasoning system employing multiple cognitive frameworks.
        Ensure that responses follow strict reasoning formats.
        {{/system}}

        {{#user}} Task: {{task}} {{/user}}

        {{#assistant}}
        *Understanding:* {{gen 'understanding' max_tokens=200}}
        *Reasoning Pathway:* {{gen 'reasoning_pathway' max_tokens=300}}
        *Comparative Insights:* {{gen 'comparative_insights' max_tokens=200}}
        *Illustrative Example:* {{gen 'illustrative_example' max_tokens=200}}
        *Final Synthesis:* {{gen 'final_synthesis' max_tokens=250}}
        {{/assistant}}
        ''')(task=task)

    @staticmethod
    def tree_of_thoughts(task):
        return guidance('''
        {{#system}}
        Apply Tree of Thoughts reasoning:
        1. Decompose the problem into components
        2. Explore multiple reasoning pathways:
           - Logical Deduction
           - Pattern Recognition
           - Constraint Analysis
        3. Evaluate and select the strongest path
        4. Synthesize a structured response
        {{/system}}

        {{#user}} {{task}} {{/user}}

        {{#assistant}}
        *Understanding:* {{gen 'understanding' max_tokens=200}}

        *Reasoning Pathway:*
        1. Problem Decomposition:
        {{gen 'decomposition' max_tokens=150}}

        2. Path Exploration:
        a) Logical Deduction:
        {{gen 'logical_deduction' max_tokens=150}}
        
        b) Pattern Recognition:
        {{gen 'pattern_recognition' max_tokens=150}}
        
        c) Constraint Analysis:
        {{gen 'constraint_analysis' max_tokens=150}}

        3. Path Evaluation:
        {{gen 'path_evaluation' max_tokens=150}}

        *Final Synthesis:* {{gen 'final_synthesis' max_tokens=200}}
        {{/assistant}}
        ''')(task=task)

    @staticmethod
    def self_consistency(task):
        return guidance('''
        {{#system}}
        Apply Self-Consistency reasoning:
        1. Generate multiple independent solutions
        2. Compare responses for coherence
        3. Identify inconsistencies
        4. Select the most consistent answer
        {{/system}}

        {{#user}} {{task}} {{/user}}

        {{#assistant}}
        *Understanding:* {{gen 'understanding' max_tokens=200}}

        *Reasoning Pathway:*
        1. Solution Generation:
        {{#loop 3}}
        Solution {{@index}}:
        {{gen (concat 'solution_' @index) max_tokens=150}}
        {{/loop}}

        2. Consistency Analysis:
        {{gen 'consistency_analysis' max_tokens=200}}

        *Final Synthesis:* {{gen 'final_synthesis' max_tokens=200}}
        {{/assistant}}
        ''')(task=task)

    @staticmethod
    def self_critique(task):
        return guidance('''
        {{#system}}
        Apply Self-Critique reasoning:
        1. Generate initial response
        2. Identify flaws or gaps
        3. Revise and improve
        4. Compare and finalize
        {{/system}}

        {{#user}} {{task}} {{/user}}

        {{#assistant}}
        *Understanding:* {{gen 'understanding' max_tokens=200}}

        *Reasoning Pathway:*
        1. Initial Response:
        {{gen 'initial_response' max_tokens=200}}

        2. Critical Analysis:
        {{gen 'critical_analysis' max_tokens=150}}

        3. Improved Response:
        {{gen 'improved_response' max_tokens=200}}

        *Final Synthesis:* {{gen 'final_synthesis' max_tokens=200}}
        {{/assistant}}
        ''')(task=task)

    @staticmethod
    def get_structured_prompt(task, technique=None):
        """
        Generates a prompt with enforced response structure using guidance.
        """
        if technique == "tree-of-thoughts":
            return PromptManager.tree_of_thoughts(task)
        elif technique == "self-consistency":
            return PromptManager.self_consistency(task)
        elif technique == "self-critique":
            return PromptManager.self_critique(task)
        else:
            return PromptManager.base_template(task)

    @staticmethod
    def validate_structure(response):
        """
        Validates that the response follows the required structure.
        """
        required_sections = [
            "*Understanding:*",
            "*Reasoning Pathway:*",
            "*Final Synthesis:*"
        ]
        return all(section in response for section in required_sections)
