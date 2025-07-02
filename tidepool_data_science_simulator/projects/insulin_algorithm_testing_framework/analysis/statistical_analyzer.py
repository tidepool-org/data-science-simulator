"""
Statistical analysis for insulin algorithm testing.

This module provides comprehensive statistical analysis capabilities for comparing
insulin delivery algorithms, including hypothesis testing, non-inferiority analysis,
and mixed-effects modeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ttest_ind
import warnings

# Optional imports for advanced analysis
try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.formula.api import mixedlm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    # warnings.warn("statsmodels not available. Some advanced statistical features will be disabled.")

from ..config.experiment_config import ExperimentConfig, AnalysisConfig
from ..core.metrics_calculator import MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None


@dataclass
class NonInferiorityResult:
    """Container for non-inferiority test results."""
    
    metric_name: str
    margin: float
    difference: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_non_inferior: bool
    interpretation: str


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for insulin algorithm comparisons.
    
    Provides methods for:
    - Paired and unpaired hypothesis testing
    - Non-inferiority analysis
    - Effect size calculations
    - Multiple comparison corrections
    - Mixed-effects modeling (if statsmodels available)
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the statistical analyzer.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.analysis_config = config.get_analysis_config()
        
        logger.info(f"Initialized StatisticalAnalyzer with config: {config}")
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Mixed-effects models will be disabled.")
    
    def compare_paired_metrics(
        self,
        reference_metrics: List[MetricsResult],
        comparison_metrics: List[MetricsResult],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compare paired metrics using multiple statistical tests.
        
        Args:
            reference_metrics: List of reference algorithm metrics
            comparison_metrics: List of comparison algorithm metrics
            metric_names: List of metric names to analyze (None for all)
            
        Returns:
            Dictionary of metric_name -> StatisticalTestResult
        """
        if len(reference_metrics) != len(comparison_metrics):
            raise ValueError("Reference and comparison metrics must have same length")
        
        if len(reference_metrics) == 0:
            raise ValueError("No metrics provided for comparison")
        
        # Convert to DataFrames for easier handling
        ref_df = pd.DataFrame([m.to_dict() for m in reference_metrics])
        comp_df = pd.DataFrame([m.to_dict() for m in comparison_metrics])
        
        if metric_names is None:
            metric_names = list(ref_df.columns)
        
        results = {}
        
        for metric in metric_names:
            if metric not in ref_df.columns or metric not in comp_df.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue
            
            ref_values = ref_df[metric].values
            comp_values = comp_df[metric].values
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(ref_values) | np.isnan(comp_values))
            ref_values = ref_values[valid_mask]
            comp_values = comp_values[valid_mask]
            
            if len(ref_values) < 3:
                logger.warning(f"Insufficient data for metric {metric}")
                continue
            
            # Perform statistical tests
            test_result = self._perform_paired_tests(ref_values, comp_values, metric)
            results[metric] = test_result
        
        return results
    
    def _perform_paired_tests(
        self,
        reference: np.ndarray,
        comparison: np.ndarray,
        metric_name: str
    ) -> StatisticalTestResult:
        """Perform paired statistical tests."""
        
        # Calculate differences
        differences = comparison - reference
        
        # Paired t-test
        try:
            t_stat, t_p = ttest_rel(comparison, reference)
        except Exception as e:
            logger.warning(f"Paired t-test failed for {metric_name}: {e}")
            t_stat, t_p = np.nan, np.nan
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = wilcoxon(comparison, reference, alternative='two-sided')
        except Exception as e:
            logger.warning(f"Wilcoxon test failed for {metric_name}: {e}")
            w_stat, w_p = np.nan, np.nan
        
        # Effect size (Cohen's d for paired data)
        effect_size = self._calculate_cohens_d_paired(differences)
        
        # Confidence interval for mean difference
        ci = self._calculate_confidence_interval(differences)
        
        # Choose primary test result (prefer t-test if assumptions met)
        if self._check_normality(differences):
            primary_stat, primary_p = t_stat, t_p
            test_name = "Paired t-test"
        else:
            primary_stat, primary_p = w_stat, w_p
            test_name = "Wilcoxon signed-rank test"
        
        # Interpretation
        interpretation = self._interpret_paired_result(
            differences, primary_p, effect_size, metric_name
        )
        
        return asdict(StatisticalTestResult(
            test_name=test_name,
            statistic=primary_stat,
            p_value=primary_p,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        ))
    
    def compare_unpaired_metrics(
        self,
        group1_metrics: List[MetricsResult],
        group2_metrics: List[MetricsResult],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compare unpaired metrics using appropriate statistical tests.
        
        Args:
            group1_metrics: List of group 1 metrics
            group2_metrics: List of group 2 metrics
            metric_names: List of metric names to analyze (None for all)
            
        Returns:
            Dictionary of metric_name -> StatisticalTestResult
        """
        # Convert to DataFrames
        group1_df = pd.DataFrame([m.to_dict() for m in group1_metrics])
        group2_df = pd.DataFrame([m.to_dict() for m in group2_metrics])
        
        if metric_names is None:
            metric_names = list(group1_df.columns)
        
        results = {}
        
        for metric in metric_names:
            if metric not in group1_df.columns or metric not in group2_df.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue
            
            group1_values = group1_df[metric].dropna().values
            group2_values = group2_df[metric].dropna().values
            
            if len(group1_values) < 3 or len(group2_values) < 3:
                logger.warning(f"Insufficient data for metric {metric}")
                continue
            
            # Perform statistical tests
            test_result = self._perform_unpaired_tests(group1_values, group2_values, metric)
            results[metric] = test_result
        
        return results
    
    def _perform_unpaired_tests(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        metric_name: str
    ) -> StatisticalTestResult:
        """Perform unpaired statistical tests."""
        
        # Independent t-test
        try:
            t_stat, t_p = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
        except Exception as e:
            logger.warning(f"Independent t-test failed for {metric_name}: {e}")
            t_stat, t_p = np.nan, np.nan
        
        # Mann-Whitney U test
        try:
            u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed for {metric_name}: {e}")
            u_stat, u_p = np.nan, np.nan
        
        # Effect size (Cohen's d for independent groups)
        effect_size = self._calculate_cohens_d_independent(group1, group2)
        
        # Confidence interval for difference in means
        ci = self._calculate_confidence_interval_independent(group1, group2)
        
        # Choose primary test result
        if self._check_normality(group1) and self._check_normality(group2):
            primary_stat, primary_p = t_stat, t_p
            test_name = "Independent t-test (Welch)"
        else:
            primary_stat, primary_p = u_stat, u_p
            test_name = "Mann-Whitney U test"
        
        # Interpretation
        interpretation = self._interpret_unpaired_result(
            group1, group2, primary_p, effect_size, metric_name
        )
        
        return asdict(StatisticalTestResult(
            test_name=test_name,
            statistic=primary_stat,
            p_value=primary_p,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        ))
    
    def perform_non_inferiority_analysis(
        self,
        reference_metrics: List[MetricsResult],
        comparison_metrics: List[MetricsResult],
        safety_metrics: Optional[List[str]] = None
    ) -> Dict[str, NonInferiorityResult]:
        """
        Perform non-inferiority analysis for safety metrics.
        
        Args:
            reference_metrics: List of reference algorithm metrics
            comparison_metrics: List of comparison algorithm metrics
            safety_metrics: List of safety metric names (None for config default)
            
        Returns:
            Dictionary of metric_name -> NonInferiorityResult
        """
        if not self.analysis_config.non_inferiority_enabled:
            logger.info("Non-inferiority analysis disabled in configuration")
            return {}
        
        if safety_metrics is None:
            safety_metrics = self.analysis_config.safety_metrics
        
        margins = self.analysis_config.non_inferiority_margins
        
        results = {}
        
        # Convert to DataFrames
        ref_df = pd.DataFrame([m.to_dict() for m in reference_metrics])
        comp_df = pd.DataFrame([m.to_dict() for m in comparison_metrics])
        
        for metric in safety_metrics:
            if metric not in margins:
                logger.warning(f"No non-inferiority margin specified for {metric}")
                continue
            
            if metric not in ref_df.columns or metric not in comp_df.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue
            
            margin = margins[metric]
            
            ref_values = ref_df[metric].dropna().values
            comp_values = comp_df[metric].dropna().values
            
            if len(ref_values) != len(comp_values):
                logger.warning(f"Mismatched data lengths for {metric}")
                continue
            
            # Perform non-inferiority test
            ni_result = self._perform_non_inferiority_test(
                ref_values, comp_values, margin, metric
            )
            results[metric] = ni_result
        
        return results
    
    def _perform_non_inferiority_test(
        self,
        reference: np.ndarray,
        comparison: np.ndarray,
        margin: float,
        metric_name: str
    ) -> NonInferiorityResult:
        """Perform non-inferiority test for a single metric."""
        
        # Calculate difference (comparison - reference)
        differences = comparison - reference
        mean_diff = np.mean(differences)
        
        # Calculate confidence interval for the difference
        ci = self._calculate_confidence_interval(differences, confidence_level=0.95)
        
        # For safety metrics, we typically want to show that the comparison
        # is not worse than reference by more than the margin
        # Non-inferiority is established if the upper bound of CI < margin
        
        # One-sided test for non-inferiority
        # H0: difference >= margin (comparison is inferior)
        # H1: difference < margin (comparison is non-inferior)
        
        try:
            # One-sided t-test
            t_stat, p_value = ttest_rel(comparison, reference)
            # Convert to one-sided p-value
            if mean_diff < 0:
                p_value = p_value / 2  # One-sided test
            else:
                p_value = 1 - (p_value / 2)
        except Exception as e:
            logger.warning(f"Non-inferiority test failed for {metric_name}: {e}")
            p_value = np.nan
        
        # Non-inferiority criterion
        is_non_inferior = ci[1] < margin  # Upper bound of CI < margin
        
        # Interpretation
        if is_non_inferior:
            interpretation = f"Non-inferiority established: upper 95% CI ({ci[1]:.3f}) < margin ({margin})"
        else:
            interpretation = f"Non-inferiority not established: upper 95% CI ({ci[1]:.3f}) >= margin ({margin})"
        
        return asdict(NonInferiorityResult(
            metric_name=metric_name,
            margin=margin,
            difference=mean_diff,
            confidence_interval=ci,
            p_value=p_value,
            is_non_inferior=bool(is_non_inferior),
            interpretation=interpretation
        ))
    
    def correct_multiple_comparisons(
        self,
        test_results: Dict[str, StatisticalTestResult]
    ) -> Dict[str, StatisticalTestResult]:
        """
        Apply multiple comparison correction to test results.
        
        Args:
            test_results: Dictionary of test results
            
        Returns:
            Dictionary of corrected test results
        """
        if len(test_results) <= 1:
            return test_results
        
        method = self.analysis_config.multiple_comparisons_method
        alpha = self.analysis_config.alpha
        
        # Extract p-values
        p_values = [result["p_value"] for result in test_results.values()]
        metric_names = list(test_results.keys())
        
        # Remove any NaN p-values
        valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
        valid_p_values = [p_values[i] for i in valid_indices]
        valid_names = [metric_names[i] for i in valid_indices]
        
        if len(valid_p_values) == 0:
            return test_results
        
        try:
            # Apply correction
            rejected, corrected_p_values, _, _ = multipletests(
                valid_p_values, alpha=alpha, method=method
            )
            
            # Update results with corrected p-values
            corrected_results = {}
            
            for i, name in enumerate(metric_names):
                if i in valid_indices:
                    valid_idx = valid_indices.index(i)
                    original_result = test_results[name]
                    
                    # Create new result with corrected p-value
                    corrected_result = asdict(StatisticalTestResult(
                        test_name=f"{original_result['test_name']} ({method} corrected)",
                        statistic=original_result['statistic'],
                        p_value=corrected_p_values[valid_idx],
                        effect_size=original_result['effect_size'],
                        confidence_interval=original_result['confidence_interval'],
                        interpretation=f"{original_result['interpretation']} (corrected p={corrected_p_values[valid_idx]:.4f})"
                    ))
                    corrected_results[name] = corrected_result
                else:
                    # Keep original result if p-value was NaN
                    corrected_results[name] = test_results[name]
            
            logger.info(f"Applied {method} correction to {len(valid_p_values)} tests")
            return corrected_results
        
        except Exception as e:
            logger.error(f"Multiple comparison correction failed: {e}")
            return test_results
    
    def _calculate_cohens_d_paired(self, differences: np.ndarray) -> float:
        """Calculate Cohen's d for paired data."""
        if len(differences) == 0:
            return np.nan
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0:
            return 0.0
        
        return mean_diff / std_diff
    
    def _calculate_cohens_d_independent(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d for independent groups."""
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        if len(data) == 0:
            return (np.nan, np.nan)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        
        if np.isnan(sem) or sem == 0:
            return (mean, mean)
        
        alpha = 1 - confidence_level
        df = len(data) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * sem
        
        return (mean - margin_error, mean + margin_error)
    
    def _calculate_confidence_interval_independent(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        if len(group1) == 0 or len(group2) == 0:
            return (np.nan, np.nan)
        
        mean_diff = np.mean(group1) - np.mean(group2)
        
        # Standard error of difference
        se1 = np.std(group1, ddof=1) / np.sqrt(len(group1))
        se2 = np.std(group2, ddof=1) / np.sqrt(len(group2))
        se_diff = np.sqrt(se1**2 + se2**2)
        
        if np.isnan(se_diff) or se_diff == 0:
            return (mean_diff, mean_diff)
        
        # Degrees of freedom (Welch's formula)
        df = (se1**2 + se2**2)**2 / (se1**4/(len(group1)-1) + se2**4/(len(group2)-1))
        
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * se_diff
        
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _check_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Check if data is normally distributed using Shapiro-Wilk test."""
        if len(data) < 3:
            return False
        
        try:
            _, p_value = stats.shapiro(data)
            return p_value > alpha
        except Exception:
            return False
    
    def _interpret_paired_result(
        self,
        differences: np.ndarray,
        p_value: float,
        effect_size: float,
        metric_name: str
    ) -> str:
        """Generate interpretation for paired test result."""
        
        mean_diff = np.mean(differences)
        alpha = self.analysis_config.alpha
        
        significant = "significant" if p_value < alpha else "not significant"
        
        # Effect size interpretation
        if np.isnan(effect_size):
            effect_desc = "unknown"
        elif abs(effect_size) < 0.2:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.5:
            effect_desc = "small"
        elif abs(effect_size) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        direction = "higher" if mean_diff > 0 else "lower"
        
        return (f"Comparison algorithm shows {direction} {metric_name} "
                f"(mean difference: {mean_diff:.3f}, p={p_value:.4f}, "
                f"effect size: {effect_desc}). Result is {significant}.")
    
    def _interpret_unpaired_result(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        p_value: float,
        effect_size: float,
        metric_name: str
    ) -> str:
        """Generate interpretation for unpaired test result."""
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        alpha = self.analysis_config.alpha
        
        significant = "significant" if p_value < alpha else "not significant"
        
        # Effect size interpretation
        if np.isnan(effect_size):
            effect_desc = "unknown"
        elif abs(effect_size) < 0.2:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.5:
            effect_desc = "small"
        elif abs(effect_size) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        direction = "higher" if mean1 > mean2 else "lower"
        
        return (f"Group 1 shows {direction} {metric_name} than Group 2 "
                f"(means: {mean1:.3f} vs {mean2:.3f}, p={p_value:.4f}, "
                f"effect size: {effect_desc}). Result is {significant}.")
    
    def compare_algorithms(
        self,
        metrics_df: pd.DataFrame,
        reference_algorithm: str,
        comparison_algorithms: List[str],
        metrics_to_analyze: Optional[List[str]] = None,
        paired: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of insulin delivery algorithms.
        
        This is the main entry point for algorithm comparisons, providing:
        - Statistical testing (paired or unpaired)
        - Non-inferiority analysis for safety metrics
        - Effect size calculations
        - Multiple comparison corrections
        - Summary statistics
        
        Args:
            metrics_df: DataFrame with metrics for all algorithms
            reference_algorithm: Name of reference algorithm (e.g., 'tempbasal')
            comparison_algorithms: List of algorithms to compare against reference
            metrics_to_analyze: List of metrics to analyze (None for all available)
            paired: Whether to perform paired comparisons (requires matching scenarios)
            
        Returns:
            Dictionary containing comprehensive comparison results
        """
        logger.info(f"Starting algorithm comparison: {reference_algorithm} vs {comparison_algorithms}")
        
        # Validate inputs
        self._validate_comparison_inputs(metrics_df, reference_algorithm, comparison_algorithms)
        
        # Determine metrics to analyze
        if metrics_to_analyze is None:
            metrics_to_analyze = self._get_default_metrics_for_analysis(metrics_df)
        
        logger.info(f"Analyzing {len(metrics_to_analyze)} metrics: {metrics_to_analyze}")
        
        # Initialize results dictionary
        results = {
            'reference_algorithm': reference_algorithm,
            'comparison_algorithms': comparison_algorithms,
            'metrics_analyzed': metrics_to_analyze,
            'analysis_type': 'paired' if paired else 'unpaired',
            'summary_statistics': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'non_inferiority_results': {},
            'multiple_comparisons_corrected': False
        }
        
        # Calculate summary statistics for each algorithm
        results['summary_statistics'] = self._calculate_summary_statistics(
            metrics_df, [reference_algorithm] + comparison_algorithms, metrics_to_analyze
        )
        
        # Perform statistical comparisons for each comparison algorithm
        for comp_algorithm in comparison_algorithms:
            logger.info(f"Comparing {reference_algorithm} vs {comp_algorithm}")
            
            # Extract data for this comparison
            ref_data, comp_data = self._extract_algorithm_data(
                metrics_df, reference_algorithm, comp_algorithm, paired
            )
            
            if len(ref_data) == 0 or len(comp_data) == 0:
                logger.warning(f"No data available for comparison: {reference_algorithm} vs {comp_algorithm}")
                continue
            
            # Perform statistical tests
            if paired:
                test_results = self._perform_paired_algorithm_comparison(
                    ref_data, comp_data, metrics_to_analyze, comp_algorithm
                )
            else:
                test_results = self._perform_unpaired_algorithm_comparison(
                    ref_data, comp_data, metrics_to_analyze, comp_algorithm
                )
            
            # Store results
            for metric, test_result in test_results.items():
                if metric not in results['statistical_tests']:
                    results['statistical_tests'][metric] = {}
                results['statistical_tests'][metric][comp_algorithm] = test_result
            
            # Calculate effect sizes
            effect_sizes = self._calculate_algorithm_effect_sizes(
                ref_data, comp_data, metrics_to_analyze, paired
            )
            results['effect_sizes'][comp_algorithm] = effect_sizes
        
        # Apply multiple comparison corrections
        if len(comparison_algorithms) > 1 or len(metrics_to_analyze) > 1:
            results = self._apply_multiple_comparison_corrections(results)
        
        # Perform non-inferiority analysis
        if self.analysis_config.non_inferiority_enabled:
            results['non_inferiority_results'] = self._perform_algorithm_non_inferiority(
                metrics_df, reference_algorithm, comparison_algorithms, paired
            )
        
        # Generate interpretation summary
        results['interpretation'] = self._generate_comparison_interpretation(results)
        
        logger.info("Algorithm comparison completed successfully")
        return results
    
    def _validate_comparison_inputs(
        self,
        metrics_df: pd.DataFrame,
        reference_algorithm: str,
        comparison_algorithms: List[str]
    ) -> None:
        """Validate inputs for algorithm comparison."""
        
        if metrics_df.empty:
            raise ValueError("Metrics DataFrame is empty")
        
        if 'alg' not in metrics_df.columns:
            raise ValueError("Metrics DataFrame must contain 'alg' column")
        
        available_algorithms = set(metrics_df['alg'].unique())
        
        if reference_algorithm not in available_algorithms:
            raise ValueError(f"Reference algorithm '{reference_algorithm}' not found in data. "
                           f"Available: {available_algorithms}")
        
        missing_algorithms = set(comparison_algorithms) - available_algorithms
        if missing_algorithms:
            raise ValueError(f"Comparison algorithms not found in data: {missing_algorithms}. "
                           f"Available: {available_algorithms}")
    
    def _get_default_metrics_for_analysis(self, metrics_df: pd.DataFrame) -> List[str]:
        """Get default metrics for analysis based on available columns."""
        
        # Priority order for metrics
        priority_metrics = [
            'time_in_range_70_180',
            'time_below_70',
            'time_below_54',
            'time_above_180',
            'mean_glucose',
            'cv_glucose',
            'lbgi',
            'hbgi',
            'cumulative_insulin'
        ]
        
        available_metrics = []
        for metric in priority_metrics:
            if metric in metrics_df.columns:
                available_metrics.append(metric)
        
        # Add any other numeric columns not in priority list
        for col in metrics_df.columns:
            if (col not in available_metrics and 
                col not in ['simulation_id', 'alg', 'patient', 'ibg', 'meal', 'paf', 'isf', 'cir', 'basal'] and
                pd.api.types.is_numeric_dtype(metrics_df[col])):
                available_metrics.append(col)
        
        return available_metrics
    
    def _calculate_summary_statistics(
        self,
        metrics_df: pd.DataFrame,
        algorithms: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate summary statistics for each algorithm."""
        
        summary_stats = {}
        
        for algorithm in algorithms:
            alg_data = metrics_df[metrics_df['alg'] == algorithm]
            summary_stats[algorithm] = {}
            
            for metric in metrics:
                if metric in alg_data.columns:
                    values = alg_data[metric].dropna()
                    if len(values) > 0:
                        summary_stats[algorithm][metric] = {
                            'count': len(values),
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'median': float(values.median()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'q25': float(values.quantile(0.25)),
                            'q75': float(values.quantile(0.75))
                        }
                    else:
                        summary_stats[algorithm][metric] = {
                            'count': 0, 'mean': 0.0, 'std': 0.0, 'median': 0.0,
                            'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0
                        }
        
        return summary_stats
    
    def _is_significant(self, p_value: float, alpha: float = 0.05) -> bool:
        """
        Check if a p-value is significant against a given alpha.

        Args:
            p_value: The p-value from a statistical test.
            alpha: The significance threshold (default 0.05).

        Returns:
            True if p_value < alpha, False otherwise.
        """
        return p_value < alpha
    
    def _extract_algorithm_data(
        self,
        metrics_df: pd.DataFrame,
        reference_algorithm: str,
        comparison_algorithm: str,
        paired: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract data for algorithm comparison."""
        
        ref_data = metrics_df[metrics_df['alg'] == reference_algorithm].copy()
        comp_data = metrics_df[metrics_df['alg'] == comparison_algorithm].copy()
        
        if paired:
            # For paired analysis, match scenarios
            # Identify pairing columns (exclude algorithm-specific ones)
            pairing_cols = []
            for col in ['patient', 'ibg', 'meal', 'isf', 'cir', 'basal']:
                if col in metrics_df.columns:
                    pairing_cols.append(col)
            
            if pairing_cols:
                # Merge on pairing columns to ensure matched pairs
                merged = ref_data.merge(
                    comp_data, 
                    on=pairing_cols, 
                    suffixes=('_ref', '_comp')
                )
                
                # Extract matched data
                ref_cols = [col for col in ref_data.columns if col not in pairing_cols]
                comp_cols = [col for col in comp_data.columns if col not in pairing_cols]
                
                ref_matched = merged[[col + '_ref' for col in ref_cols if col + '_ref' in merged.columns]]
                comp_matched = merged[[col + '_comp' for col in comp_cols if col + '_comp' in merged.columns]]
                
                # Rename columns back
                ref_matched.columns = [col.replace('_ref', '') for col in ref_matched.columns]
                comp_matched.columns = [col.replace('_comp', '') for col in comp_matched.columns]
                
                return ref_matched, comp_matched
            else:
                logger.warning("No pairing columns found for paired analysis")
        
        return ref_data, comp_data
    
    def _perform_paired_algorithm_comparison(
        self,
        ref_data: pd.DataFrame,
        comp_data: pd.DataFrame,
        metrics: List[str],
        comparison_algorithm: str
    ) -> Dict[str, StatisticalTestResult]:
        """Perform paired statistical comparison between algorithms."""
        
        results = {}
        
        for metric in metrics:
            if metric not in ref_data.columns or metric not in comp_data.columns:
                continue
            
            ref_values = ref_data[metric].dropna().values
            comp_values = comp_data[metric].dropna().values
            
            # Ensure same length for paired test
            min_length = min(len(ref_values), len(comp_values))
            if min_length < 3:
                logger.warning(f"Insufficient paired data for {metric}: {min_length} pairs")
                continue
            
            ref_values = ref_values[:min_length]
            comp_values = comp_values[:min_length]
            
            # Perform paired test
            test_result = self._perform_paired_tests(ref_values, comp_values, metric)
            results[metric] = test_result
        
        return results
    
    def _perform_unpaired_algorithm_comparison(
        self,
        ref_data: pd.DataFrame,
        comp_data: pd.DataFrame,
        metrics: List[str],
        comparison_algorithm: str
    ) -> Dict[str, StatisticalTestResult]:
        """Perform unpaired statistical comparison between algorithms."""
        
        results = {}
        
        for metric in metrics:
            if metric not in ref_data.columns or metric not in comp_data.columns:
                continue
            
            ref_values = ref_data[metric].dropna().values
            comp_values = comp_data[metric].dropna().values
            
            if len(ref_values) < 3 or len(comp_values) < 3:
                logger.warning(f"Insufficient data for {metric}: ref={len(ref_values)}, comp={len(comp_values)}")
                continue
            
            # Perform unpaired test
            test_result = self._perform_unpaired_tests(ref_values, comp_values, metric)
            results[metric] = test_result
        
        return results
    
    def _calculate_algorithm_effect_sizes(
        self,
        ref_data: pd.DataFrame,
        comp_data: pd.DataFrame,
        metrics: List[str],
        paired: bool
    ) -> Dict[str, float]:
        """Calculate effect sizes for algorithm comparison."""
        
        effect_sizes = {}
        
        for metric in metrics:
            if metric not in ref_data.columns or metric not in comp_data.columns:
                continue
            
            ref_values = ref_data[metric].dropna().values
            comp_values = comp_data[metric].dropna().values
            
            if len(ref_values) == 0 or len(comp_values) == 0:
                continue
            
            if paired:
                # For paired data, calculate effect size on differences
                min_length = min(len(ref_values), len(comp_values))
                differences = comp_values[:min_length] - ref_values[:min_length]
                effect_size = self._calculate_cohens_d_paired(differences)
            else:
                # For unpaired data, calculate independent groups effect size
                effect_size = self._calculate_cohens_d_independent(comp_values, ref_values)
            
            effect_sizes[metric] = effect_size
        
        return effect_sizes
    
    def _apply_multiple_comparison_corrections(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparison corrections to statistical test results."""
        
        # Collect all p-values for correction
        all_test_results = {}
        
        for metric, algorithm_results in results['statistical_tests'].items():
            for algorithm, test_result in algorithm_results.items():
                key = f"{metric}_{algorithm}"
                all_test_results[key] = test_result
        
        if len(all_test_results) > 1:
            corrected_results = self.correct_multiple_comparisons(all_test_results)
            
            # Update results with corrected p-values
            for metric, algorithm_results in results['statistical_tests'].items():
                for algorithm in algorithm_results.keys():
                    key = f"{metric}_{algorithm}"
                    if key in corrected_results:
                        results['statistical_tests'][metric][algorithm] = corrected_results[key]
            
            results['multiple_comparisons_corrected'] = True
            logger.info(f"Applied multiple comparison correction to {len(all_test_results)} tests")
        
        return results
    
    def _perform_algorithm_non_inferiority(
        self,
        metrics_df: pd.DataFrame,
        reference_algorithm: str,
        comparison_algorithms: List[str],
        paired: bool
    ) -> Dict[str, Dict[str, NonInferiorityResult]]:
        """Perform non-inferiority analysis for algorithm comparison."""
        
        ni_results = {}
        
        for comp_algorithm in comparison_algorithms:
            # Extract data for this comparison
            ref_data, comp_data = self._extract_algorithm_data(
                metrics_df, reference_algorithm, comp_algorithm, paired
            )
            
            if len(ref_data) == 0 or len(comp_data) == 0:
                continue
            
            # Convert to MetricsResult format for non-inferiority analysis
            ref_metrics = [self._dataframe_row_to_metrics_result(row) for _, row in ref_data.iterrows()]
            comp_metrics = [self._dataframe_row_to_metrics_result(row) for _, row in comp_data.iterrows()]
            
            # Perform non-inferiority analysis
            ni_result = self.perform_non_inferiority_analysis(ref_metrics, comp_metrics)
            
            if ni_result:
                ni_results[comp_algorithm] = ni_result
        
        return ni_results
    
    def _dataframe_row_to_metrics_result(self, row: pd.Series) -> MetricsResult:
        """Convert DataFrame row to MetricsResult object."""
        
        # Import here to avoid circular imports
        from ..core.metrics_calculator import MetricsResult
        
        # Create MetricsResult with available data
        kwargs = {}
        for field in MetricsResult.__dataclass_fields__:
            if field in row.index and pd.notna(row[field]):
                kwargs[field] = row[field]
            else:
                kwargs[field] = 0.0  # Default value
        
        return MetricsResult(**kwargs)
    
    def _generate_comparison_interpretation(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable interpretation of comparison results."""
        
        interpretation = {}
        
        # Overall summary
        ref_alg = results['reference_algorithm']
        comp_algs = results['comparison_algorithms']
        
        interpretation['summary'] = (
            f"Comparison of {ref_alg} (reference) vs {', '.join(comp_algs)} "
            f"using {results['analysis_type']} analysis"
        )
        
        # Significant findings
        significant_findings = []
        
        for metric, algorithm_results in results['statistical_tests'].items():
            for algorithm, test_result in algorithm_results.items():
                if self._is_significant(test_result["p_value"], self.analysis_config.alpha):
                    effect_size = results['effect_sizes'].get(algorithm, {}).get(metric, 0)
                    
                    if abs(effect_size) >= 0.5:  # Medium or large effect
                        significant_findings.append(
                            f"{algorithm} shows significant difference in {metric} "
                            f"(p={test_result["p_value"]:.3f}, effect size={effect_size:.2f})"
                        )
        
        interpretation['significant_findings'] = significant_findings
        
        # Non-inferiority summary
        if results['non_inferiority_results']:
            ni_summary = []
            for algorithm, ni_results in results['non_inferiority_results'].items():
                for metric, ni_result in ni_results.items():
                    if ni_result["is_non_inferior"]:
                        ni_summary.append(f"{algorithm} is non-inferior to {ref_alg} for {metric}")
                    else:
                        ni_summary.append(f"{algorithm} failed non-inferiority test for {metric}")
            
            interpretation['non_inferiority'] = ni_summary
        
        return interpretation
