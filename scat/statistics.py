"""
Statistical analysis module for SCAT.
Provides group comparisons, normality tests, and effect sizes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# scipy is imported lazily when needed


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
        correction: str = 'holm'
    ) -> Dict:
        """
        Compare multiple groups with correction for multiple comparisons.
        
        Args:
            groups: Dict mapping group names to data arrays
            correction: 'holm', 'bonferroni', or 'none'
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
        pairs = list(combinations(group_names, 2))
        
        for name1, name2 in pairs:
            result = self.compare_two_groups(
                valid_groups[name1], valid_groups[name2],
                group1_name=name1, group2_name=name2
            )
            pairwise.append(result)
        
        # Apply correction (only for valid results with p_value)
        if correction != 'none' and pairwise:
            # Filter results that have p_value (not error results)
            valid_results = [r for r in pairwise if 'p_value' in r]
            
            if valid_results:
                p_values = [r['p_value'] for r in valid_results]
                corrected_p = self._correct_pvalues(p_values, correction)
                
                for result, p_corr in zip(valid_results, corrected_p):
                    result['p_value_corrected'] = p_corr
                    result['significant_corrected'] = p_corr < self.alpha
        
        return {
            'overall_test': overall_test,
            'overall_statistic': float(stat),
            'overall_p_value': float(p),
            'overall_significant': p < self.alpha,
            'n_groups': len(valid_groups),
            'group_names': group_names,
            'correction_method': correction,
            'pairwise_comparisons': pairwise
        }
    
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
            corrected = np.minimum.accumulate(corrected[sorted_indices][::-1])[::-1]
            result = np.zeros(n)
            for rank, idx in enumerate(sorted_indices):
                result[idx] = min(corrected[rank], 1.0)
            
            return list(result)
        
        return list(p_values)
    
    def distribution_comparison(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        group1_name: str = 'Group1',
        group2_name: str = 'Group2'
    ) -> Dict:
        """
        Compare distributions using Kolmogorov-Smirnov test.
        Useful for hue (pH) distribution comparisons.
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        stat, p = self.stats.ks_2samp(group1, group2)
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'n1': len(group1),
            'n2': len(group2),
            'test_name': 'Kolmogorov-Smirnov',
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha
        }


def generate_statistics_report(
    film_summary: pd.DataFrame,
    group_column: str,
    metrics: List[str] = None
) -> Dict:
    """
    Generate comprehensive statistics report for grouped data.
    
    Args:
        film_summary: DataFrame with film-level summary
        group_column: Column name for grouping (e.g., 'condition', 'mating_status')
        metrics: List of metric columns to analyze
        
    Returns:
        Dict with statistical results
    """
    if metrics is None:
        metrics = ['rod_fraction', 'n_total', 'total_iod', 'normal_mean_area', 'rod_mean_area']
    
    analyzer = StatisticalAnalyzer()
    results = {}
    
    # Get groups, excluding 'ungrouped'
    groups = [g for g in film_summary[group_column].unique() if g != 'ungrouped']
    
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
            # Skip metrics that fail
            results[metric] = {'error': str(e)}
    
    return results


class SpatialStatistics:
    """Statistical analysis for spatial data."""
    
    def __init__(self, alpha: float = 0.05):
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
        from scipy import stats
        
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
        from scipy import stats
        
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
        from scipy import stats
        
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
