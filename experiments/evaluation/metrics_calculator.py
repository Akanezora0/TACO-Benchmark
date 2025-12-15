"""
Comprehensive Metrics Calculator

Calculates various evaluation metrics for Text-to-SQL experiments.
"""

from typing import Dict, List, Tuple, Optional
import statistics
from scipy import stats
import numpy as np


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize metrics calculator
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.calculate_ci = self.config.get('calculate_confidence_intervals', True)
        self.confidence_level = self.config.get('confidence_level', 0.95)
    
    def calculate_execution_accuracy(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Calculate execution accuracy and related metrics
        
        Args:
            results: List of result dictionaries with 'is_correct' field
            
        Returns:
            Dictionary with accuracy metrics
        """
        total = len(results)
        correct = sum(1 for r in results if r.get('is_correct', False))
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            'total_queries': total,
            'correct_executions': correct,
            'execution_accuracy': accuracy,
            'error_count': total - correct
        }
        
        # Calculate confidence interval
        if self.calculate_ci and total > 0:
            ci = self._calculate_confidence_interval(accuracy, total)
            metrics['confidence_interval'] = ci
            metrics['confidence_level'] = self.confidence_level
        
        return metrics
    
    def calculate_error_breakdown(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Calculate error type breakdown
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with error breakdown
        """
        error_types = {
            'syntax_errors': 0,
            'execution_errors': 0,
            'wrong_results': 0,
            'timeout_errors': 0,
            'other_errors': 0
        }
        
        for result in results:
            if result.get('is_correct', False):
                continue
            
            error_type = result.get('error_type', 'other_errors')
            if error_type in error_types:
                error_types[error_type] += 1
            else:
                error_types['other_errors'] += 1
        
        total_errors = sum(error_types.values())
        error_percentages = {
            k: (v / total_errors * 100) if total_errors > 0 else 0.0
            for k, v in error_types.items()
        }
        
        return {
            'error_counts': error_types,
            'error_percentages': error_percentages,
            'total_errors': total_errors
        }
    
    def calculate_performance_metrics(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Calculate performance metrics (latency, token usage, etc.)
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with performance metrics
        """
        latencies = [r.get('latency', 0) for r in results if 'latency' in r]
        prompt_tokens = [r.get('prompt_tokens', 0) for r in results if 'prompt_tokens' in r]
        completion_tokens = [r.get('completion_tokens', 0) for r in results if 'completion_tokens' in r]
        
        metrics = {}
        
        if latencies:
            metrics['latency'] = {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                'min': min(latencies),
                'max': max(latencies)
            }
        
        if prompt_tokens:
            metrics['prompt_tokens'] = {
                'mean': statistics.mean(prompt_tokens),
                'median': statistics.median(prompt_tokens),
                'total': sum(prompt_tokens)
            }
        
        if completion_tokens:
            metrics['completion_tokens'] = {
                'mean': statistics.mean(completion_tokens),
                'median': statistics.median(completion_tokens),
                'total': sum(completion_tokens)
            }
        
        return metrics
    
    def calculate_statistical_significance(
        self,
        results1: List[Dict],
        results2: List[Dict],
        metric: str = 'is_correct'
    ) -> Dict:
        """
        Calculate statistical significance between two result sets
        
        Args:
            results1: First result set
            results2: Second result set
            metric: Metric to compare ('is_correct', 'latency', etc.)
            
        Returns:
            Dictionary with statistical test results
        """
        if len(results1) != len(results2):
            return {'error': 'Result sets must have the same length'}
        
        values1 = [r.get(metric, 0) for r in results1]
        values2 = [r.get(metric, 0) for r in results2]
        
        # Paired t-test for continuous metrics
        if metric in ['latency', 'prompt_tokens', 'completion_tokens']:
            statistic, p_value = stats.ttest_rel(values1, values2)
            test_type = 'paired_t_test'
        # McNemar's test for binary metrics
        else:
            # Convert to binary
            binary1 = [1 if v else 0 for v in values1]
            binary2 = [1 if v else 0 for v in values2]
            
            # Create contingency table
            both_correct = sum(1 for a, b in zip(binary1, binary2) if a == 1 and b == 1)
            both_wrong = sum(1 for a, b in zip(binary1, binary2) if a == 0 and b == 0)
            only1 = sum(1 for a, b in zip(binary1, binary2) if a == 1 and b == 0)
            only2 = sum(1 for a, b in zip(binary1, binary2) if a == 0 and b == 1)
            
            # McNemar's test
            if only1 + only2 > 0:
                chi2 = ((abs(only1 - only2) - 1) ** 2) / (only1 + only2)
                from scipy.stats import chi2 as chi2_dist
                p_value = 1 - chi2_dist.cdf(chi2, df=1)
                statistic = chi2
                test_type = 'mcnemar_test'
            else:
                statistic = 0
                p_value = 1.0
                test_type = 'mcnemar_test'
        
        return {
            'test_type': test_type,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'metric': metric
        }
    
    def _calculate_confidence_interval(
        self,
        proportion: float,
        n: int
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for proportion
        
        Args:
            proportion: Sample proportion
            n: Sample size
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n == 0:
            return (0.0, 0.0)
        
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        se = np.sqrt(proportion * (1 - proportion) / n)
        margin = z * se
        
        lower = max(0.0, proportion - margin)
        upper = min(1.0, proportion + margin)
        
        return (float(lower), float(upper))
    
    def calculate_comprehensive_metrics(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Calculate all metrics comprehensively
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {
            'execution_accuracy': self.calculate_execution_accuracy(results),
            'error_breakdown': self.calculate_error_breakdown(results),
            'performance': self.calculate_performance_metrics(results)
        }
        
        return metrics


def calculate_metrics(results: List[Dict], config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to calculate metrics
    
    Args:
        results: List of result dictionaries
        config: Optional configuration
        
    Returns:
        Dictionary with metrics
    """
    calculator = MetricsCalculator(config)
    return calculator.calculate_comprehensive_metrics(results)

