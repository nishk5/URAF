import json
import os
from datetime import datetime

class BenchmarkTracker:
    """
    Tracks LLM benchmark results for agent-based evaluations.
    Stores results and provides comparisons over time.
    """

    def __init__(self, save_path="data/benchmark_results.json"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def save_result(self, model_name, agent_type, evaluation):
        """
        Saves benchmark results for an LLM on a specific agent-type test.
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "agent_type": agent_type,
            "evaluation": evaluation
        }

        with open(self.save_path, "a") as f:
            json.dump(result, f)
            f.write("\n")

    def load_results(self):
        """
        Loads all stored benchmark results.
        """
        if not os.path.exists(self.save_path):
            return []

        with open(self.save_path, "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def compare_models(self):
        """
        Generates a summary of how different models perform across agent types.
        """
        results = self.load_results()
        performance = {}

        for result in results:
            model = result["model"]
            agent_type = result["agent_type"]
            score = result["evaluation"]["URAF Score"]

            if model not in performance:
                performance[model] = {}

            if agent_type not in performance[model]:
                performance[model][agent_type] = []

            performance[model][agent_type].append(score)

        return performance
