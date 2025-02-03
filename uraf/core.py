from uraf.llm_client import LLMClient
from uraf.response_processor import ResponseProcessor
from uraf.evaluator import Evaluator

class URAF:
    """
    Unified Reasoning and Aggregation Framework (URAF)
    - High-level abstraction for executing LLM evaluation pipelines.
    """

    def __init__(self, model="openai/gpt-4", api_url="http://localhost:1234/v1/completions"):
        self.llm = LLMClient(model=model, api_url=api_url)
        self.processor = ResponseProcessor()
        self.evaluator = Evaluator()

    def run(self, prompt):
        """
        Executes the URAF evaluation pipeline:
        1. Sends prompt to LLM
        2. Processes the LLM response
        3. Evaluates response using structured reasoning
        """
        raw_response = self.llm.query(prompt)
        processed_response = self.processor.process(raw_response)
        evaluation_result = self.evaluator.evaluate(processed_response)

        return {
            "prompt": prompt,
            "raw_response": raw_response,
            "processed_response": processed_response,
            "evaluation_result": evaluation_result
        }
