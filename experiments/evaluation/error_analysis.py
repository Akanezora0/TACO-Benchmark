"""
Error Analysis Tools

Analyze and categorize errors in Text-to-SQL experiments.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import re


class ErrorAnalyzer:
    """Analyze errors in experiment results"""
    
    def __init__(self):
        """Initialize error analyzer"""
        self.error_patterns = {
            'syntax_error': [
                r'syntax error',
                r'SQL syntax',
                r'unexpected token',
                r'invalid syntax',
                r'parse error'
            ],
            'table_not_found': [
                r'table.*not found',
                r'no such table',
                r'unknown table'
            ],
            'column_not_found': [
                r'column.*not found',
                r'no such column',
                r'unknown column'
            ],
            'type_mismatch': [
                r'type mismatch',
                r'incompatible types',
                r'cannot convert'
            ],
            'aggregate_error': [
                r'aggregate.*group by',
                r'must appear in.*group by',
                r'not a group by expression'
            ],
            'join_error': [
                r'ambiguous column',
                r'join.*on',
                r'foreign key'
            ]
        }
    
    def categorize_error(
        self,
        error_message: str,
        sql: Optional[str] = None
    ) -> str:
        """
        Categorize error based on error message and SQL
        
        Args:
            error_message: Error message string
            sql: Optional SQL that caused the error
            
        Returns:
            Error category string
        """
        error_lower = error_message.lower()
        
        # Check against patterns
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return category
        
        # Check SQL for common issues
        if sql:
            sql_upper = sql.upper()
            
            # Missing quotes
            if re.search(r'[^"]\w+\.\w+[^"]', sql) and 'ambiguous' in error_lower:
                return 'join_error'
            
            # Aggregate without GROUP BY
            if re.search(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper) and 'group by' not in sql_lower:
                return 'aggregate_error'
        
        return 'other_error'
    
    def analyze_errors(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Analyze errors in results
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with error analysis
        """
        error_categories = defaultdict(list)
        error_details = defaultdict(int)
        
        for result in results:
            if result.get('is_correct', False):
                continue
            
            error_message = result.get('error', '')
            error_type = result.get('error_type', '')
            sql = result.get('generated_sql', '')
            
            # Categorize error
            category = self.categorize_error(error_message, sql)
            
            error_categories[category].append({
                'item_id': result.get('item_id', ''),
                'error': error_message,
                'sql': sql,
                'error_type': error_type
            })
            
            error_details[category] += 1
        
        # Calculate percentages
        total_errors = sum(error_details.values())
        error_percentages = {
            k: (v / total_errors * 100) if total_errors > 0 else 0.0
            for k, v in error_details.items()
        }
        
        return {
            'error_categories': dict(error_categories),
            'error_counts': dict(error_details),
            'error_percentages': error_percentages,
            'total_errors': total_errors,
            'total_queries': len(results)
        }
    
    def find_common_patterns(
        self,
        error_categories: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Find common error patterns
        
        Args:
            error_categories: Dictionary of error categories
            
        Returns:
            Dictionary with common patterns
        """
        patterns = {}
        
        for category, errors in error_categories.items():
            if not errors:
                continue
            
            # Extract common substrings from error messages
            error_messages = [e.get('error', '') for e in errors]
            
            # Find most common words
            words = defaultdict(int)
            for msg in error_messages:
                for word in msg.lower().split():
                    if len(word) > 3:  # Ignore short words
                        words[word] += 1
            
            common_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:10]
            
            patterns[category] = {
                'count': len(errors),
                'common_words': [w[0] for w in common_words],
                'sample_errors': error_messages[:5]  # First 5 as samples
            }
        
        return patterns
    
    def generate_error_report(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Generate comprehensive error report
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with error report
        """
        analysis = self.analyze_errors(results)
        patterns = self.find_common_patterns(analysis['error_categories'])
        
        return {
            'summary': {
                'total_queries': analysis['total_queries'],
                'total_errors': analysis['total_errors'],
                'error_rate': analysis['total_errors'] / analysis['total_queries'] if analysis['total_queries'] > 0 else 0.0
            },
            'error_breakdown': {
                'counts': analysis['error_counts'],
                'percentages': analysis['error_percentages']
            },
            'common_patterns': patterns,
            'detailed_errors': analysis['error_categories']
        }


def analyze_errors(results: List[Dict]) -> Dict:
    """
    Convenience function to analyze errors
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with error analysis
    """
    analyzer = ErrorAnalyzer()
    return analyzer.generate_error_report(results)

