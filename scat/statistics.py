"""
Statistical analysis module for SCAT.
Provides group comparisons, normality tests, and effect sizes.
Supports multi-group analysis with control/treatment designation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# scipy is imported lazily when needed

# Keywords for auto-detecting group types
CONTROL_KEYWORDS = ['control', 'ctrl', 'neg', 'negative', 'vehicle', 'wt', 'wild']
POSITIVE_CONTROL_KEYWORDS = ['positive', 'pos', 'pos_ctrl', 'positive_control']
TREATMENT_KEYWORDS = ['treatment', 'treat', 'treated', 'exp', 'experimental', 'test']


def detect_group_type(group_name) -> str:
    """
    Auto-detect group type from name.
    
    Returns:
        'negative_control', 'positive_control', 'treatment', or 'unknown'
    """
    # Convert to string in case of numeric group names
    name_lower = str(group_name).lower().replace(' ', '_').replace('-', '_')
    
    # Check positive control first (more specific)
    for kw in POSITIVE_CONTROL_KEYWORDS:
        if kw in name_lower:
            return 'positive_control'
    
    # Then negative control
    for kw in CONTROL_KEYWORDS:
        if kw in name_lower:
            return 'negative_control'
    
    # Then treatment
    for kw in TREATMENT_KEYWORDS:
        if kw in name_lower:
            return 'treatment'
    
    return 'unknown'


def categorize_groups(group_names: List) -> Dict[str, List[str]]:
    """
    Categorize groups into control/treatment categories.
    
    Returns:
        Dict with keys: 'negative_control', 'positive_control', 'treatment', 'unknown'
    """
    categories = {
        'negative_control': [],
        'positive_control': [],
        'treatment': [],
        'unknown': []
    }
    
    for name in group_names:
        group_type = detect_group_type(name)
        categories[group_type].append(str(name))  # Convert to string
    
    return categories


class StatisticalAnalyzer:
    """Statistical analysis for excreta data."""
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats  # Store for use in methods
        self.alpha = alpha
    
    def normality_test(self, data: np.ndarray, method: str = 'shapiro') -> Dict:
        """
        Test for normality.
        
        Args:
            data: Array of values
            method: 'shapiro' or 'jarque_bera'
            
        Returns:
            Dict with statistic, p-value, and is_normal
        """
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) < 3:
            return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False, 'method': method}
        
        if method == 'shapiro':
            # Shapiro-Wilk (better for small samples)
            stat, p = self.stats.shapiro(data[:5000])  # Limit for large samples
        else:
            # Jarque-Bera (better for large samples)
            stat, p = self.stats.jarque_bera(data)
        
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'is_normal': p > self.alpha,
            'method': method
        }
    
    def compare_two_groups(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        group1_name: str = 'Group1',
        group2_name: str = 'Group2',
        paired: bool = False
    ) -> Dict:
        """
        Compare two groups with appropriate test.
        
        Automatically selects t-test or Mann-Whitney U based on normality.
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) < 3 or len(group2) < 3:
            return {'error': 'Insufficient samples', 'n1': len(group1), 'n2': len(group2)}
        
        # Test normality
        norm1 = self.normality_test(group1)
        norm2 = self.normality_test(group2)
        both_normal = norm1['is_normal'] and norm2['is_normal']
        
        # Select and run test
        if both_normal:
            if paired and len(group1) == len(group2):
                stat, p = self.stats.ttest_rel(group1, group2)
                test_name = 'Paired t-test'
            else:
                stat, p = self.stats.ttest_ind(group1, group2)
                test_name = 'Independent t-test'
        else:
            if paired and len(group1) == len(group2):
                stat, p = self.stats.wilcoxon(group1, group2)
                test_name = 'Wilcoxon signed-rank'
            else:
                stat, p = self.stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = 'Mann-Whitney U'
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            effect_interpretation = 'negligible'
        elif d_abs < 0.5:
            effect_interpretation = 'small'
        elif d_abs < 0.8:
            effect_interpretation = 'medium'
        else:
            effect_interpretation = 'large'
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'n1': len(group1),
            'n2': len(group2),
            'mean1': float(np.mean(group1)),
            'mean2': float(np.mean(group2)),
            'std1': float(np.std(group1)),
            'std2': float(np.std(group2)),
            'test_name': test_name,
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha,
            'cohens_d': float(cohens_d),
            'effect_size': effect_interpretation,
            'normality_group1': norm1['is_normal'],
            'normality_group2': norm2['is_normal']
        }
    
    def compare_multiple_groups(
        self,
        groups: Dict[str, np.ndarray],
        correction: str = 'holm',
        control_groups: List[str] = None
    ) -> Dict:
        """
        Compare multiple groups with correction for multiple comparisons.
        
        Args:
            groups: Dict mapping group names to data arrays
            correction: 'holm', 'bonferroni', or 'none'
            control_groups: List of group names to use as controls (for Dunnett-style comparison)
        """
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]
        
        # Filter out empty groups and NaN values
        valid_groups = {}
        for name, data in zip(group_names, group_data):
            clean_data = data[~np.isnan(data)] if len(data) > 0 else np.array([])
            if len(clean_data) >= 2:  # Need at least 2 samples per group
                valid_groups[name] = clean_data
        
        if len(valid_groups) < 2:
            return {
                'error': 'Insufficient valid groups (need at least 2 groups with 2+ samples each)',
                'n_groups': len(valid_groups),
                'group_names': list(valid_groups.keys())
            }
        
        group_names = list(valid_groups.keys())
        group_data = list(valid_groups.values())
        
        # Overall test (ANOVA or Kruskal-Wallis)
        normality_results = [self.normality_test(g) for g in group_data]
        all_normal = all(r['is_normal'] for r in normality_results)
        
        try:
            if all_normal:
                stat, p = self.stats.f_oneway(*group_data)
                overall_test = 'One-way ANOVA'
            else:
                stat, p = self.stats.kruskal(*group_data)
                overall_test = 'Kruskal-Wallis H'
        except Exception as e:
            return {
                'error': f'Statistical test failed: {str(e)}',
                'n_groups': len(valid_groups),
                'group_names': group_names
            }
        
        # Pairwise comparisons
        pairwise = []
        
        if control_groups:
            # Dunnett-style: only compare treatments vs controls
            controls = [c for c in control_groups if c in valid_groups]
            treatments = [g for g in group_names if g not in controls]
            
            for control in controls:
                for treatment in treatments:
                    result = self.compare_two_groups(
                        valid_groups[control], valid_groups[treatment],
                        group1_name=control, group2_name=treatment
                    )
                    result['comparison_type'] = 'control_vs_treatment'
                    pairwise.append(result)
        else:
            # All pairwise comparisons
            pairs = list(combinations(group_names, 2))
            for name1, name2 in pairs:
                result = self.compare_two_groups(
                    valid_groups[name1], valid_groups[name2],
                    group1_name=name1, group2_name=name2
                )
                result['comparison_type'] = 'pairwise'
                pairwise.append(result)
        
        # Apply correction (only for valid results with p_value)
        if correction != 'none' and pairwise:
            valid_results = [r for r in pairwise if 'p_value' in r]
            
            if valid_results:
                p_values = [r['p_value'] for r in valid_results]
                corrected_p = self._correct_pvalues(p_values, correction)
                
                for result, p_corr in zip(valid_results, corrected_p):
                    result['p_value_corrected'] = p_corr
                    result['significant_corrected'] = p_corr < self.alpha
        
        # Calculate group statistics summary
        group_stats = {}
        for name in group_names:
            data = valid_groups[name]
            group_stats[name] = {
                'n': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'ci_95': self._confidence_interval(data)
            }
        
        return {
            'overall_test': overall_test,
            'overall_statistic': float(stat),
            'overall_p_value': float(p),
            'overall_significant': p < self.alpha,
            'n_groups': len(valid_groups),
            'group_names': group_names,
            'group_statistics': group_stats,
            'correction_method': correction,
            'pairwise_comparisons': pairwise,
            'control_groups': control_groups or []
        }
    
    def compare_control_vs_treatment(
        self,
        groups: Dict[str, np.ndarray],
        group_types: Dict[str, str] = None
    ) -> Dict:
        """
        Compare aggregated control groups vs treatment groups.
        
        Args:
            groups: Dict mapping group names to data arrays
            group_types: Optional dict mapping group names to types 
                        ('negative_control', 'positive_control', 'treatment')
                        If None, auto-detects from group names.
        
        Returns:
            Dict with control vs treatment comparison results
        """
        # Auto-detect group types if not provided
        if group_types is None:
            categories = categorize_groups(list(groups.keys()))
            group_types = {}
            for gtype, names in categories.items():
                for name in names:
                    group_types[name] = gtype
        
        # Separate controls and treatments
        control_data = []
        treatment_data = []
        control_names = []
        treatment_names = []
        
        for name, data in groups.items():
            gtype = group_types.get(name, 'unknown')
            clean_data = np.array(data)
            clean_data = clean_data[~np.isnan(clean_data)]
            
            if len(clean_data) == 0:
                continue
                
            if gtype in ['negative_control', 'positive_control']:
                control_data.extend(clean_data)
                control_names.append(name)
            elif gtype == 'treatment':
                treatment_data.extend(clean_data)
                treatment_names.append(name)
        
        control_data = np.array(control_data)
        treatment_data = np.array(treatment_data)
        
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                'error': 'Insufficient data for control vs treatment comparison',
                'n_control': len(control_data),
                'n_treatment': len(treatment_data),
                'control_groups': control_names,
                'treatment_groups': treatment_names
            }
        
        # Compare aggregated groups
        result = self.compare_two_groups(
            control_data, treatment_data,
            group1_name='Control (combined)',
            group2_name='Treatment (combined)'
        )
        
        result['control_groups'] = control_names
        result['treatment_groups'] = treatment_names
        result['comparison_type'] = 'aggregated_control_vs_treatment'
        
        return result
    
    def _confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return (np.nan, np.nan)
        
        n = len(data)
        mean = np.mean(data)
        se = self.stats.sem(data)
        
        # t-value for confidence interval
        t_val = self.stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * se
        
        return (float(mean - margin), float(mean + margin))
    
    def _correct_pvalues(self, p_values: List[float], method: str) -> List[float]:
        """Apply multiple comparison correction."""
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            return list(np.minimum(p_values * n, 1.0))
        
        elif method == 'holm':
            # Holm-Bonferroni step-down
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros(n)
            
            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = p_values[idx] * (n - rank)
            
            # Enforce monotonicity
            corrected = np.minimum.accumulate(corrected[np.argsort(sorted_indices)][::-1])[::-1]
            corrected = np.minimum(corrected, 1.0)
            return list(corrected)
        
        return list(p_values)
    
    def run_all_tests(
        self, 
        film_summary: pd.DataFrame, 
        group_by: str = None,
        metrics: List[str] = None,
        control_groups: List[str] = None
    ) -> Dict:
        """
        Run comprehensive statistical analysis.
        
        Args:
            film_summary: DataFrame with image-level data
            group_by: Column name for grouping
            metrics: List of metrics to analyze
            control_groups: List of control group names
            
        Returns:
            Dict with all statistical results
        """
        if metrics is None:
            metrics = ['rod_fraction', 'n_total', 'n_rod', 'n_normal', 
                      'total_iod', 'normal_mean_area', 'rod_mean_area']
        
        results = {
            'metrics': {},
            'summary': {},
            'control_vs_treatment': {}
        }
        
        if group_by is None or group_by not in film_summary.columns:
            return results
        
        # Get unique groups
        groups = [g for g in film_summary[group_by].unique() if g != 'ungrouped' and pd.notna(g)]
        
        if len(groups) < 2:
            return results
        
        # Auto-detect group types
        categories = categorize_groups(groups)
        results['group_categories'] = categories
        
        # Determine control groups if not specified
        if control_groups is None:
            control_groups = categories['negative_control'] + categories['positive_control']
        
        results['control_groups'] = control_groups
        results['n_groups'] = len(groups)
        
        for metric in metrics:
            if metric not in film_summary.columns:
                continue
            
            group_data = {
                g: film_summary[film_summary[group_by] == g][metric].dropna().values
                for g in groups
            }
            
            # Filter groups with insufficient data
            group_data = {k: v for k, v in group_data.items() if len(v) >= 2}
            
            if len(group_data) < 2:
                continue
            
            try:
                # Multi-group comparison
                results['metrics'][metric] = self.compare_multiple_groups(
                    group_data, 
                    correction='holm',
                    control_groups=control_groups if control_groups else None
                )
                
                # Control vs Treatment comparison
                if control_groups and len(control_groups) > 0:
                    results['control_vs_treatment'][metric] = self.compare_control_vs_treatment(
                        group_data
                    )
                    
            except Exception as e:
                results['metrics'][metric] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a human-readable summary of statistical results."""
        summary = {
            'significant_metrics': [],
            'large_effects': [],
            'recommendations': []
        }
        
        for metric, data in results.get('metrics', {}).items():
            if 'error' in data:
                continue
            
            if data.get('overall_significant'):
                summary['significant_metrics'].append({
                    'metric': metric,
                    'test': data.get('overall_test'),
                    'p_value': data.get('overall_p_value')
                })
            
            # Check for large effects in pairwise comparisons
            for pw in data.get('pairwise_comparisons', []):
                if pw.get('effect_size') == 'large':
                    summary['large_effects'].append({
                        'metric': metric,
                        'group1': pw.get('group1_name'),
                        'group2': pw.get('group2_name'),
                        'cohens_d': pw.get('cohens_d')
                    })
        
        # Add recommendations
        n_groups = results.get('n_groups', 0)
        if n_groups >= 3:
            summary['recommendations'].append(
                'Multiple groups detected. Use corrected p-values for pairwise comparisons.'
            )
        
        ctrl_groups = results.get('control_groups', [])
        if ctrl_groups:
            summary['recommendations'].append(
                f'Control groups identified: {", ".join(ctrl_groups)}. '
                'Control vs Treatment comparisons available.'
            )
        
        return summary


def analyze_groups(
    film_summary: pd.DataFrame,
    group_column: str,
    metrics: List[str] = None
) -> Dict:
    """
    Analyze groups in film summary data.
    
    Args:
        film_summary: DataFrame with image-level statistics
        group_column: Column name for grouping
        metrics: List of metric columns to analyze
        
    Returns:
        Dict with statistical results
    """
    analyzer = StatisticalAnalyzer()
    return analyzer.run_all_tests(film_summary, group_by=group_column, metrics=metrics)


def generate_statistics_report(
    film_summary: pd.DataFrame,
    group_column: str,
    metrics: List[str] = None
) -> Dict:
    """
    Generate comprehensive statistics report for grouped data.
    
    Args:
        film_summary: DataFrame with film-level summary
        group_column: Column name for grouping (e.g., 'condition', 'group')
        metrics: List of metric columns to analyze
        
    Returns:
        Dict with statistical results
    """
    if metrics is None:
        metrics = ['rod_fraction', 'n_total', 'n_rod', 'n_normal',
                   'total_iod', 'normal_mean_area', 'rod_mean_area']
    
    analyzer = StatisticalAnalyzer()
    results = {}
    
    # Get groups, excluding 'ungrouped'
    groups = [g for g in film_summary[group_column].unique() 
              if g != 'ungrouped' and pd.notna(g)]
    
    if len(groups) < 2:
        return {}  # Not enough groups for comparison
    
    for metric in metrics:
        if metric not in film_summary.columns:
            continue
        
        group_data = {
            g: film_summary[film_summary[group_column] == g][metric].dropna().values
            for g in groups
        }
        
        # Filter groups with insufficient data
        group_data = {k: v for k, v in group_data.items() if len(v) >= 2}
        
        if len(group_data) < 2:
            continue
        
        try:
            if len(group_data) == 2:
                names = list(group_data.keys())
                results[metric] = analyzer.compare_two_groups(
                    group_data[names[0]], group_data[names[1]],
                    group1_name=names[0], group2_name=names[1]
                )
            else:
                results[metric] = analyzer.compare_multiple_groups(group_data)
        except Exception as e:
            results[metric] = {'error': str(e)}
    
    return results


class SpatialStatistics:
    """Statistical analysis for spatial data."""
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
    
    def compare_clustering(
        self,
        group1_r_values: List[float],
        group2_r_values: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        """
        Compare Clark-Evans R values between two groups.
        Tests whether spatial clustering differs between conditions.
        """
        g1 = np.array(group1_r_values)
        g2 = np.array(group2_r_values)
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 3 or len(g2) < 3:
            return {'error': 'Insufficient samples'}
        
        # Mann-Whitney U (non-parametric, R often not normal)
        stat, p = self.stats.mannwhitneyu(g1, g2, alternative='two-sided')
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean_r': float(np.mean(g1)),
            'group2_mean_r': float(np.mean(g2)),
            'group1_n': len(g1),
            'group2_n': len(g2),
            'test_name': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha,
            'interpretation': self._interpret_r_difference(np.mean(g1), np.mean(g2))
        }
    
    def _interpret_r_difference(self, r1: float, r2: float) -> str:
        """Interpret difference in Clark-Evans R."""
        diff = r1 - r2
        if abs(diff) < 0.1:
            return 'similar_clustering'
        elif diff > 0:
            return 'group1_more_dispersed'
        else:
            return 'group2_more_dispersed'
    
    def compare_edge_preference(
        self,
        group1_edge_fractions: List[float],
        group2_edge_fractions: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        """
        Compare edge vs center preference between groups.
        """
        g1 = np.array(group1_edge_fractions)
        g2 = np.array(group2_edge_fractions)
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 3 or len(g2) < 3:
            return {'error': 'Insufficient samples'}
        
        stat, p = self.stats.mannwhitneyu(g1, g2, alternative='two-sided')
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean_edge_frac': float(np.mean(g1)),
            'group2_mean_edge_frac': float(np.mean(g2)),
            'test_name': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha
        }
    
    def compare_nnd(
        self,
        group1_nnd: List[float],
        group2_nnd: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        """
        Compare Nearest Neighbor Distances between groups.
        """
        g1 = np.array(group1_nnd)
        g2 = np.array(group2_nnd)
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 3 or len(g2) < 3:
            return {'error': 'Insufficient samples'}
        
        # Test normality
        _, p1 = self.stats.shapiro(g1[:5000])
        _, p2 = self.stats.shapiro(g2[:5000])
        
        if p1 > 0.05 and p2 > 0.05:
            stat, p = self.stats.ttest_ind(g1, g2)
            test_name = 't-test'
        else:
            stat, p = self.stats.mannwhitneyu(g1, g2, alternative='two-sided')
            test_name = 'Mann-Whitney U'
        
        # Effect size
        pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
        cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean_nnd': float(np.mean(g1)),
            'group2_mean_nnd': float(np.mean(g2)),
            'group1_std_nnd': float(np.std(g1)),
            'group2_std_nnd': float(np.std(g2)),
            'test_name': test_name,
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha,
            'cohens_d': float(cohens_d)
        }


def generate_spatial_statistics_report(
    spatial_results_by_group: Dict[str, List],  # group_name -> List[SpatialResult]
) -> Dict:
    """
    Generate statistical comparisons of spatial metrics between groups.
    
    Args:
        spatial_results_by_group: Dict mapping group names to lists of SpatialResult
        
    Returns:
        Dict with statistical comparison results
    """
    stats = SpatialStatistics()
    results = {}
    
    group_names = list(spatial_results_by_group.keys())
    
    if len(group_names) < 2:
        return {'error': 'Need at least 2 groups for comparison'}
    
    # For simplicity, compare first two groups
    # Could extend to multiple comparisons if needed
    g1_name, g2_name = group_names[0], group_names[1]
    g1_results = spatial_results_by_group[g1_name]
    g2_results = spatial_results_by_group[g2_name]
    
    # Clark-Evans R comparison
    g1_r = [r.clark_evans_r for r in g1_results if r.clark_evans_r > 0]
    g2_r = [r.clark_evans_r for r in g2_results if r.clark_evans_r > 0]
    results['clark_evans'] = stats.compare_clustering(g1_r, g2_r, g1_name, g2_name)
    
    # Edge fraction comparison
    g1_edge = [r.edge_fraction for r in g1_results]
    g2_edge = [r.edge_fraction for r in g2_results]
    results['edge_fraction'] = stats.compare_edge_preference(g1_edge, g2_edge, g1_name, g2_name)
    
    # NND comparison
    g1_nnd = [r.nnd_mean for r in g1_results if r.nnd_mean > 0]
    g2_nnd = [r.nnd_mean for r in g2_results if r.nnd_mean > 0]
    results['nnd'] = stats.compare_nnd(g1_nnd, g2_nnd, g1_name, g2_name)
    
    return results
