"""
Statistical Analysis Module

Provides rigorous statistical evaluation for benchmark comparisons.
Includes hypothesis testing, confidence intervals, and effect sizes.
"""

from typing import Any

import numpy as np
from loguru import logger
from scipy import stats
from scipy.stats import bootstrap


class BenchmarkStatistics:
    """Statistical analysis for benchmark results."""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize benchmark statistics.

        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        logger.info(f"Initialized BenchmarkStatistics (confidence={self.confidence_level:.0%})")

    def bootstrap_confidence_interval(
        self, scores: list[float], n_resamples: int = 10000, confidence: float | None = None
    ) -> dict[str, float]:
        """
        Calculate bootstrap confidence interval for scores.

        Args:
            scores: List of benchmark scores
            n_resamples: Number of bootstrap samples
            confidence: Confidence level (uses self.confidence_level if None)

        Returns:
            Statistics with confidence intervals
        """
        if not scores:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "error": "No scores provided"}

        scores_array = np.array(scores)
        confidence = confidence or self.confidence_level

        # Calculate mean
        mean_score = np.mean(scores_array)

        # Bootstrap confidence interval
        rng = np.random.default_rng()

        def statistic(x):
            return np.mean(x)

        try:
            # Perform bootstrap
            result = bootstrap(
                (scores_array,),
                statistic,
                n_resamples=n_resamples,
                confidence_level=confidence,
                random_state=rng,
                method="percentile",
            )

            ci_lower, ci_upper = result.confidence_interval

            logger.debug(f"Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

            return {
                "mean": float(mean_score),
                "std": float(np.std(scores_array)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "confidence_level": confidence,
                "n_samples": len(scores),
                "margin_of_error": float((ci_upper - ci_lower) / 2),
            }

        except Exception as e:
            logger.error(f"Bootstrap CI calculation failed: {e}")
            # Fallback to normal approximation
            sem = stats.sem(scores_array)
            ci = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)

            return {
                "mean": float(mean_score),
                "std": float(np.std(scores_array)),
                "ci_lower": float(mean_score - ci),
                "ci_upper": float(mean_score + ci),
                "confidence_level": confidence,
                "n_samples": len(scores),
                "margin_of_error": float(ci),
                "method": "t-distribution (fallback)",
            }

    def paired_t_test(self, scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
        """
        Perform paired t-test to compare two models.

        Tests if Model A is significantly different from Model B.

        Args:
            scores_a: Scores from Model A
            scores_b: Scores from Model B

        Returns:
            Test results with interpretation
        """
        if len(scores_a) != len(scores_b):
            logger.error("Score lists must have same length for paired t-test")
            return {"error": "Mismatched lengths", "valid": False}

        if not scores_a or not scores_b:
            return {"error": "Empty score lists", "valid": False}

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)

        # Determine significance
        is_significant = p_value < self.alpha

        # Determine winner
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        if is_significant:
            winner = "Model A" if mean_a > mean_b else "Model B"
            interpretation = f"{winner} is statistically significantly better (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        logger.info(f"Paired t-test: {interpretation}")

        return {
            "test": "paired_t_test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "is_significant": is_significant,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "mean_difference": float(mean_a - mean_b),
            "winner": winner if is_significant else "Tie",
            "interpretation": interpretation,
            "n_pairs": len(scores_a),
        }

    def independent_t_test(
        self, scores_a: list[float], scores_b: list[float], equal_var: bool = True
    ) -> dict[str, Any]:
        """
        Perform independent samples t-test.

        Use when comparing different test sets (not paired).

        Args:
            scores_a: Scores from Model A
            scores_b: Scores from Model B
            equal_var: Assume equal variance (True) or use Welch's t-test (False)

        Returns:
            Test results
        """
        if not scores_a or not scores_b:
            return {"error": "Empty score lists", "valid": False}

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        # Perform independent t-test
        statistic, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=equal_var)

        is_significant = p_value < self.alpha

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        if is_significant:
            winner = "Model A" if mean_a > mean_b else "Model B"
            interpretation = f"{winner} is significantly better (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"

        test_name = "Welch's t-test" if not equal_var else "Student's t-test"
        logger.info(f"{test_name}: {interpretation}")

        return {
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "is_significant": is_significant,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "mean_difference": float(mean_a - mean_b),
            "winner": winner if is_significant else "Tie",
            "interpretation": interpretation,
            "n_a": len(scores_a),
            "n_b": len(scores_b),
        }

    def cohens_d(self, scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
        """
        Calculate Cohen's d effect size.

        Measures practical significance (not just statistical significance).

        Effect size interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large

        Args:
            scores_a: Scores from Model A
            scores_b: Scores from Model B

        Returns:
            Effect size metrics
        """
        if not scores_a or not scores_b:
            return {"error": "Empty score lists", "valid": False}

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        # Calculate means and standard deviations
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)

        # Pooled standard deviation
        n_a = len(scores_a)
        n_b = len(scores_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        # Cohen's d
        d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Interpret effect size
        abs_d = abs(d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        logger.info(f"Cohen's d = {d:.3f} ({magnitude} effect)")

        return {
            "cohens_d": float(d),
            "magnitude": magnitude,
            "interpretation": f"{magnitude.capitalize()} effect size (d={d:.3f})",
            "mean_difference": float(mean_a - mean_b),
            "pooled_std": float(pooled_std),
        }

    def anova(self, model_scores: dict[str, list[float]]) -> dict[str, Any]:
        """
        Perform one-way ANOVA to compare multiple models.

        Tests if there are significant differences among 3+ models.

        Args:
            model_scores: Dict mapping model names to score lists

        Returns:
            ANOVA results
        """
        if len(model_scores) < 2:
            return {"error": "Need at least 2 models for ANOVA", "valid": False}

        # Extract scores
        score_lists = [np.array(scores) for scores in model_scores.values()]
        model_names = list(model_scores.keys())

        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*score_lists)

        is_significant = p_value < self.alpha

        # Calculate group means
        group_means = {name: float(np.mean(scores)) for name, scores in model_scores.items()}

        if is_significant:
            interpretation = f"Significant differences detected among models (p={p_value:.4f})"
        else:
            interpretation = f"No significant differences among models (p={p_value:.4f})"

        logger.info(f"ANOVA: {interpretation}")

        return {
            "test": "one_way_anova",
            "f_statistic": float(f_statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "is_significant": is_significant,
            "num_models": len(model_scores),
            "group_means": group_means,
            "interpretation": interpretation,
        }

    def multiple_comparisons(self, model_scores: dict[str, list[float]], method: str = "bonferroni") -> dict[str, Any]:
        """
        Perform pairwise comparisons with multiple comparison correction.

        Args:
            model_scores: Dict mapping model names to score lists
            method: Correction method ('bonferroni', 'holm', or 'none')

        Returns:
            Pairwise comparison results
        """
        model_names = list(model_scores.keys())
        n_comparisons = len(model_names) * (len(model_names) - 1) // 2

        if n_comparisons == 0:
            return {"error": "Need at least 2 models", "valid": False}

        # Adjust alpha for multiple comparisons
        if method == "bonferroni":
            adjusted_alpha = self.alpha / n_comparisons
        elif method == "holm":
            # Holm-Bonferroni will be applied after sorting p-values
            adjusted_alpha = self.alpha
        else:
            adjusted_alpha = self.alpha

        # Perform all pairwise comparisons
        comparisons = []
        p_values = []

        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1 :]:
                scores_a = model_scores[name_a]
                scores_b = model_scores[name_b]

                # Perform t-test
                result = self.independent_t_test(scores_a, scores_b)

                comparisons.append(
                    {
                        "model_a": name_a,
                        "model_b": name_b,
                        "p_value": result["p_value"],
                        "mean_difference": result["mean_difference"],
                        "winner": result["winner"],
                    }
                )
                p_values.append(result["p_value"])

        # Apply correction
        if method == "holm":
            # Sort by p-value
            sorted_indices = np.argsort(p_values)
            for rank, idx in enumerate(sorted_indices, 1):
                adjusted_alpha_holm = self.alpha / (n_comparisons - rank + 1)
                comparisons[idx]["adjusted_alpha"] = adjusted_alpha_holm
                comparisons[idx]["is_significant"] = comparisons[idx]["p_value"] < adjusted_alpha_holm
        else:
            for comp in comparisons:
                comp["adjusted_alpha"] = adjusted_alpha
                comp["is_significant"] = comp["p_value"] < adjusted_alpha

        logger.info(f"Multiple comparisons: {n_comparisons} pairs, method={method}")

        return {
            "method": method,
            "n_comparisons": n_comparisons,
            "adjusted_alpha": adjusted_alpha if method == "bonferroni" else "variable (Holm)",
            "comparisons": comparisons,
            "significant_pairs": [
                f"{c['model_a']} vs {c['model_b']}" for c in comparisons if c.get("is_significant", False)
            ],
        }

    def power_analysis(self, effect_size: float, n_samples: int, alpha: float | None = None) -> dict[str, Any]:
        """
        Calculate statistical power for a given effect size and sample size.

        Power = probability of detecting an effect if it exists.

        Args:
            effect_size: Cohen's d effect size
            n_samples: Number of samples per group
            alpha: Significance level (uses self.alpha if None)

        Returns:
            Power analysis results
        """
        from scipy.stats import t as t_dist

        alpha = alpha or self.alpha

        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_samples / 2)

        # Critical value for two-tailed test
        df = 2 * n_samples - 2
        critical_value = t_dist.ppf(1 - alpha / 2, df)

        # Power calculation
        power = 1 - t_dist.cdf(critical_value, df, ncp) + t_dist.cdf(-critical_value, df, ncp)

        interpretation = ""
        if power < 0.7:
            interpretation = "Low power: high risk of Type II error"
        elif power < 0.8:
            interpretation = "Moderate power: acceptable for exploratory studies"
        elif power < 0.9:
            interpretation = "Good power: suitable for most studies"
        else:
            interpretation = "Excellent power: very low risk of Type II error"

        logger.info(f"Power analysis: power={power:.3f} ({interpretation})")

        return {
            "power": float(power),
            "effect_size": effect_size,
            "n_samples": n_samples,
            "alpha": alpha,
            "interpretation": interpretation,
        }

    def sample_size_calculation(
        self, effect_size: float, desired_power: float = 0.8, alpha: float | None = None
    ) -> dict[str, Any]:
        """
        Calculate required sample size for desired power.

        Args:
            effect_size: Expected Cohen's d effect size
            desired_power: Desired statistical power (default 0.8)
            alpha: Significance level

        Returns:
            Required sample size
        """
        alpha = alpha or self.alpha

        # Simplified approximation
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(desired_power)

        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        # Round up
        n_per_group = int(np.ceil(n_per_group))

        logger.info(f"Required sample size: {n_per_group} per group")

        return {
            "required_n_per_group": n_per_group,
            "effect_size": effect_size,
            "desired_power": desired_power,
            "alpha": alpha,
            "interpretation": f"Need {n_per_group} samples per group to detect effect size {effect_size} with {desired_power:.0%} power",
        }


class PerformanceRegression:
    """Detect performance regressions over time."""

    def __init__(self):
        """Initialize performance regression detector."""
        logger.info("Initialized PerformanceRegression")

    def detect_regression(
        self, historical_scores: list[float], current_score: float, threshold: float = -0.05
    ) -> dict[str, Any]:
        """
        Detect if current performance is a regression.

        Args:
            historical_scores: Previous benchmark scores
            current_score: Current benchmark score
            threshold: Minimum acceptable change (default -5%)

        Returns:
            Regression detection result
        """
        if not historical_scores:
            return {"is_regression": False, "reason": "No historical data", "valid": False}

        historical_array = np.array(historical_scores)
        historical_mean = np.mean(historical_array)
        historical_std = np.std(historical_array)

        # Calculate z-score
        z_score = (current_score - historical_mean) / (historical_std + 1e-6)

        # Relative change
        relative_change = (current_score - historical_mean) / (historical_mean + 1e-6)

        # Detect regression
        is_regression = relative_change < threshold or z_score < -2.0

        if is_regression:
            severity = "severe" if z_score < -3.0 else "moderate" if z_score < -2.5 else "mild"
            interpretation = f"{severity.capitalize()} regression detected: {relative_change:.1%} change"
        else:
            interpretation = f"No regression: {relative_change:+.1%} change"

        logger.info(f"Regression check: {interpretation}")

        return {
            "is_regression": is_regression,
            "current_score": float(current_score),
            "historical_mean": float(historical_mean),
            "historical_std": float(historical_std),
            "z_score": float(z_score),
            "relative_change": float(relative_change),
            "interpretation": interpretation,
        }
