class URAFScorer:
    """
    Computes structured reasoning scores for LLM evaluations.
    """

    DEFAULT_WEIGHTS = {
        "Understanding": 2,
        "Reasoning Pathway": 3,
        "Comparative Insights": 2,
        "Illustrative Example": 1,
        "Final Synthesis": 2
    }

    @staticmethod
    def evaluate(response, weights=None):
        weights = weights or URAFScorer.DEFAULT_WEIGHTS
        scores = {key: 0 for key in weights.keys()}
        for key in weights.keys():
            if key in response and response[key]:
                scores[key] = weights[key]
        scores["Total Score"] = sum(scores.values())
        return scores
