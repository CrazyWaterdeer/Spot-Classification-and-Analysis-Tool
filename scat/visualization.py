"""
Visualization module for SCAT.
Provides PCA, clustering, density plots, and comparison charts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

# Lazy loading flags
_viz_libs_loaded = False
_plt = None
_mpatches = None
_to_rgba = None
_sns = None
_PCA = None
_StandardScaler = None
_KMeans = None
_AgglomerativeClustering = None

HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_SKLEARN = False


def _load_viz_libs():
    """Lazy load visualization libraries."""
    global _viz_libs_loaded, _plt, _mpatches, _to_rgba, _sns
    global _PCA, _StandardScaler, _KMeans, _AgglomerativeClustering
    global HAS_MATPLOTLIB, HAS_SEABORN, HAS_SKLEARN
    
    if _viz_libs_loaded:
        return
    
    # Load matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import to_rgba
        _plt = plt
        _mpatches = mpatches
        _to_rgba = to_rgba
        HAS_MATPLOTLIB = True
    except ImportError:
        warnings.warn("matplotlib not installed. Visualization features disabled.")
    
    # Load seaborn
    try:
        import seaborn as sns
        _sns = sns
        HAS_SEABORN = True
    except ImportError:
        pass
    
    # Load sklearn
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, AgglomerativeClustering
        _PCA = PCA
        _StandardScaler = StandardScaler
        _KMeans = KMeans
        _AgglomerativeClustering = AgglomerativeClustering
        HAS_SKLEARN = True
    except ImportError:
        pass
    
    _viz_libs_loaded = True


class Visualizer:
    """Generate visualizations for excreta analysis."""
    
    def __init__(self, output_dir: Path, style: str = 'whitegrid'):
        _load_viz_libs()  # Lazy load visualization libraries
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB and HAS_SEABORN:
            _sns.set_style(style)
            _plt.rcParams['figure.figsize'] = (10, 8)
            _plt.rcParams['figure.dpi'] = 150
    
    def pca_plot(
        self,
        film_summary: pd.DataFrame,
        features: List[str] = None,
        color_by: str = None,
        title: str = "PCA of Film Samples",
        filename: str = "pca_plot.png"
    ) -> Optional[str]:
        """
        Generate PCA plot of film samples.
        
        Args:
            film_summary: DataFrame with film-level data
            features: Columns to use for PCA (default: numeric columns)
            color_by: Column for color coding (e.g., 'condition')
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            warnings.warn("PCA plot requires matplotlib and scikit-learn")
            return None
        
        # Select features
        if features is None:
            features = [
                'n_normal', 'n_rod', 'rod_fraction',
                'normal_mean_area', 'rod_mean_area',
                'normal_mean_iod', 'total_iod',
                'normal_mean_hue', 'normal_mean_lightness'
            ]
        
        # Filter available features
        features = [f for f in features if f in film_summary.columns]
        
        if len(features) < 2:
            warnings.warn("Not enough features for PCA")
            return None
        
        # Prepare data
        df = film_summary[features].dropna()
        if len(df) < 3:
            warnings.warn("Not enough samples for PCA")
            return None
        
        # Standardize and fit PCA
        scaler = _StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        pca = _PCA(n_components=min(2, len(features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        fig, ax = _plt.subplots(figsize=(10, 8))
        
        if color_by and color_by in film_summary.columns:
            # Get matching indices
            valid_idx = df.index
            groups = film_summary.loc[valid_idx, color_by]
            unique_groups = groups.unique()
            colors = _plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
            
            for group, color in zip(unique_groups, colors):
                mask = groups == group
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    c=[color], label=group, s=100, alpha=0.7, edgecolors='black'
                )
            ax.legend(title=color_by)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add loading vectors
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        scale = 2
        for i, feature in enumerate(features):
            ax.annotate(
                '', xy=(loadings[i, 0]*scale, loadings[i, 1]*scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5)
            )
            ax.text(
                loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1,
                feature, fontsize=8, color='red', alpha=0.7
            )
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def hue_density_plot(
        self,
        deposits_data: Dict[str, pd.DataFrame],
        title: str = "Hue Distribution by Condition",
        filename: str = "hue_density.png",
        bandwidth: float = 7
    ) -> Optional[str]:
        """
        Generate hue (pH proxy) density plot.
        
        Args:
            deposits_data: Dict mapping condition names to deposit DataFrames
            title: Plot title
            filename: Output filename
            bandwidth: KDE bandwidth
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        fig, ax = _plt.subplots(figsize=(12, 6))
        
        colors = _plt.cm.Set1(np.linspace(0, 1, len(deposits_data)))
        
        for (name, df), color in zip(deposits_data.items(), colors):
            if 'mean_hue' in df.columns:
                hue_values = df['mean_hue'].dropna()
                if len(hue_values) > 10:
                    _sns.kdeplot(
                        data=hue_values, ax=ax, label=f"{name} (n={len(hue_values)})",
                        color=color, linewidth=2, bw_adjust=bandwidth/10
                    )
        
        ax.set_xlabel('Hue (degrees)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 360)
        
        # Add pH reference zones for Bromophenol Blue
        ax.axvspan(0, 60, alpha=0.1, color='yellow', label='Acidic')
        ax.axvspan(60, 150, alpha=0.1, color='green', label='Transitional')
        ax.axvspan(150, 360, alpha=0.1, color='blue', label='Basic')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def violin_comparison(
        self,
        film_summary: pd.DataFrame,
        metric: str,
        group_by: str,
        title: str = None,
        filename: str = None,
        ylabel: str = None
    ) -> Optional[str]:
        """
        Generate violin plot comparing groups.
        
        Args:
            film_summary: DataFrame with film-level data
            metric: Column to plot (e.g., 'rod_fraction')
            group_by: Column for grouping
            title: Plot title
            filename: Output filename
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        if metric not in film_summary.columns or group_by not in film_summary.columns:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 6))
        
        _sns.violinplot(
            data=film_summary, x=group_by, y=metric,
            ax=ax, inner='box', palette='Set2'
        )
        _sns.stripplot(
            data=film_summary, x=group_by, y=metric,
            ax=ax, color='black', alpha=0.5, size=4
        )
        
        ax.set_title(title or f'{metric} by {group_by}')
        ax.set_ylabel(ylabel or metric)
        ax.set_xlabel(group_by)
        
        _plt.tight_layout()
        filepath = self.output_dir / (filename or f'violin_{metric}_by_{group_by}.png')
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def box_comparison(
        self,
        film_summary: pd.DataFrame,
        metrics: List[str],
        group_by: str,
        title: str = "Metrics Comparison",
        filename: str = "box_comparison.png"
    ) -> Optional[str]:
        """
        Generate grouped box plots for multiple metrics.
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        metrics = [m for m in metrics if m in film_summary.columns]
        if not metrics:
            return None
        
        # Melt data for grouped plotting
        df_melted = film_summary.melt(
            id_vars=[group_by],
            value_vars=metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        fig, ax = _plt.subplots(figsize=(12, 6))
        
        _sns.boxplot(
            data=df_melted, x='Metric', y='Value', hue=group_by,
            ax=ax, palette='Set2'
        )
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        _plt.xticks(rotation=45, ha='right')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def heatmap(
        self,
        film_summary: pd.DataFrame,
        features: List[str] = None,
        row_label: str = 'filename',
        title: str = "Feature Heatmap",
        filename: str = "heatmap.png",
        cluster_rows: bool = True
    ) -> Optional[str]:
        """
        Generate heatmap of features across samples.
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        if features is None:
            features = [
                'n_normal', 'n_rod', 'rod_fraction',
                'normal_mean_area', 'normal_mean_iod', 'total_iod'
            ]
        
        features = [f for f in features if f in film_summary.columns]
        if not features:
            return None
        
        # Prepare data
        df = film_summary[features].copy()
        
        # Standardize for visualization
        df_scaled = (df - df.mean()) / df.std()
        
        # Set index
        if row_label in film_summary.columns:
            df_scaled.index = film_summary[row_label]
        
        # Create heatmap
        fig, ax = _plt.subplots(figsize=(12, max(8, len(df_scaled) * 0.3)))
        
        if cluster_rows and HAS_SKLEARN:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            
            # Cluster rows
            if len(df_scaled) > 2:
                linkage_matrix = linkage(pdist(df_scaled.fillna(0)), method='ward')
                dendro = dendrogram(linkage_matrix, no_plot=True)
                df_scaled = df_scaled.iloc[dendro['leaves']]
        
        _sns.heatmap(
            df_scaled, ax=ax, cmap='RdBu_r', center=0,
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': 'Z-score'}
        )
        
        ax.set_title(title)
        _plt.xticks(rotation=45, ha='right')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def scatter_matrix(
        self,
        film_summary: pd.DataFrame,
        features: List[str] = None,
        color_by: str = None,
        filename: str = "scatter_matrix.png"
    ) -> Optional[str]:
        """
        Generate scatter matrix (pair plot) of features.
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        if features is None:
            features = ['n_total', 'rod_fraction', 'total_iod', 'normal_mean_area']
        
        features = [f for f in features if f in film_summary.columns]
        if len(features) < 2:
            return None
        
        plot_df = film_summary[features].copy()
        if color_by and color_by in film_summary.columns:
            plot_df[color_by] = film_summary[color_by]
            g = _sns.pairplot(plot_df, hue=color_by, palette='Set1', diag_kind='kde')
        else:
            g = _sns.pairplot(plot_df, diag_kind='kde')
        
        g.fig.suptitle('Feature Relationships', y=1.02)
        
        filepath = self.output_dir / filename
        g.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def area_iod_scatter(
        self,
        deposits_df: pd.DataFrame,
        color_by_label: bool = True,
        title: str = "Area vs IOD by Deposit Type",
        filename: str = "area_iod_scatter.png"
    ) -> Optional[str]:
        """
        Scatter plot of area vs IOD for individual deposits.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 8))
        
        if color_by_label and 'label' in deposits_df.columns:
            colors = {'normal': 'green', 'rod': 'red', 'artifact': 'gray'}
            for label in ['normal', 'rod', 'artifact']:
                mask = deposits_df['label'] == label
                if mask.any():
                    ax.scatter(
                        deposits_df.loc[mask, 'area_px'],
                        deposits_df.loc[mask, 'iod'],
                        c=colors.get(label, 'blue'),
                        label=label.capitalize(),
                        alpha=0.5, s=30
                    )
            ax.legend()
        else:
            ax.scatter(deposits_df['area_px'], deposits_df['iod'], alpha=0.5)
        
        ax.set_xlabel('Area (pixels)')
        ax.set_ylabel('IOD')
        ax.set_title(title)
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def summary_dashboard(
        self,
        film_summary: pd.DataFrame,
        group_by: str = None,
        filename: str = "dashboard.png"
    ) -> Optional[str]:
        """
        Generate summary dashboard with multiple plots.
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        fig, axes = _plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. ROD fraction distribution
        ax = axes[0, 0]
        if group_by and group_by in film_summary.columns:
            _sns.boxplot(data=film_summary, x=group_by, y='rod_fraction', ax=ax, palette='Set2')
            _sns.stripplot(data=film_summary, x=group_by, y='rod_fraction', ax=ax, color='black', alpha=0.5)
        else:
            _sns.histplot(film_summary['rod_fraction'], ax=ax, kde=True)
        ax.set_title('ROD Fraction Distribution')
        ax.set_ylabel('ROD Fraction')
        
        # 2. Total deposits
        ax = axes[0, 1]
        if group_by and group_by in film_summary.columns:
            _sns.barplot(data=film_summary, x=group_by, y='n_total', ax=ax, palette='Set2', errorbar='sd')
        else:
            _sns.histplot(film_summary['n_total'], ax=ax, kde=True)
        ax.set_title('Total Deposits per Film')
        ax.set_ylabel('Count')
        
        # 3. Normal vs ROD counts
        ax = axes[1, 0]
        if 'n_normal' in film_summary.columns and 'n_rod' in film_summary.columns:
            plot_df = film_summary.melt(
                id_vars=[group_by] if group_by else [],
                value_vars=['n_normal', 'n_rod'],
                var_name='Type', value_name='Count'
            )
            plot_df['Type'] = plot_df['Type'].map({'n_normal': 'Normal', 'n_rod': 'ROD'})
            _sns.barplot(data=plot_df, x='Type', y='Count', hue=group_by if group_by else None, ax=ax, palette='Set2', errorbar='sd')
        ax.set_title('Deposit Counts by Type')
        
        # 4. Total IOD
        ax = axes[1, 1]
        if 'total_iod' in film_summary.columns:
            if group_by and group_by in film_summary.columns:
                _sns.boxplot(data=film_summary, x=group_by, y='total_iod', ax=ax, palette='Set2')
            else:
                _sns.histplot(film_summary['total_iod'], ax=ax, kde=True)
        ax.set_title('Total IOD Distribution')
        ax.set_ylabel('Total IOD')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)


def generate_all_visualizations(
    film_summary: pd.DataFrame,
    deposits_df: pd.DataFrame,
    output_dir: Path,
    group_by: str = None
) -> Dict[str, str]:
    """
    Generate all available visualizations.
    
    Returns:
        Dict mapping visualization name to filepath
    """
    viz = Visualizer(output_dir)
    results = {}
    
    # Dashboard
    path = viz.summary_dashboard(film_summary, group_by)
    if path:
        results['dashboard'] = path
    
    # PCA
    path = viz.pca_plot(film_summary, color_by=group_by)
    if path:
        results['pca'] = path
    
    # Heatmap
    path = viz.heatmap(film_summary)
    if path:
        results['heatmap'] = path
    
    # Violin plots for key metrics
    for metric in ['rod_fraction', 'total_iod', 'n_total']:
        if metric in film_summary.columns and group_by:
            path = viz.violin_comparison(film_summary, metric, group_by)
            if path:
                results[f'violin_{metric}'] = path
    
    # Scatter matrix
    path = viz.scatter_matrix(film_summary, color_by=group_by)
    if path:
        results['scatter_matrix'] = path
    
    # Area vs IOD scatter
    if deposits_df is not None and len(deposits_df) > 0:
        path = viz.area_iod_scatter(deposits_df)
        if path:
            results['area_iod'] = path
    
    return results


class SpatialVisualizer:
    """Generate spatial analysis visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def density_heatmap(
        self,
        density_map: np.ndarray,
        title: str = "Deposit Density Map",
        filename: str = "density_map.png"
    ) -> Optional[str]:
        """Generate density heatmap from spatial analysis."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = _plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(density_map, cmap='YlOrRd', interpolation='bilinear')
        _plt.colorbar(im, ax=ax, label='Deposit Density')
        
        ax.set_title(title)
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def nnd_histogram(
        self,
        nnd_values: np.ndarray,
        mean_nnd: float = None,
        title: str = "Nearest Neighbor Distance Distribution",
        filename: str = "nnd_histogram.png"
    ) -> Optional[str]:
        """Generate NND histogram."""
        if not HAS_MATPLOTLIB or len(nnd_values) == 0:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 5))
        
        ax.hist(nnd_values, bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
        
        if mean_nnd is not None:
            ax.axvline(mean_nnd, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_nnd:.1f}')
            ax.legend()
        
        ax.set_xlabel('Nearest Neighbor Distance (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def quadrant_plot(
        self,
        quadrant_counts: Dict[str, int],
        title: str = "Quadrant Distribution",
        filename: str = "quadrant_plot.png"
    ) -> Optional[str]:
        """Generate quadrant distribution plot."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = _plt.subplots(figsize=(8, 8))
        
        # Create 2x2 grid
        data = np.array([
            [quadrant_counts['Q1'], quadrant_counts['Q2']],
            [quadrant_counts['Q3'], quadrant_counts['Q4']]
        ])
        
        im = ax.imshow(data, cmap='Blues')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{data[i, j]}', 
                              ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Left', 'Right'])
        ax.set_yticklabels(['Top', 'Bottom'])
        ax.set_title(title)
        
        _plt.colorbar(im, ax=ax, label='Deposit Count')
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def spatial_scatter(
        self,
        centroids: np.ndarray,
        labels: List[str] = None,
        image_shape: Tuple[int, int] = None,
        title: str = "Deposit Spatial Distribution",
        filename: str = "spatial_scatter.png"
    ) -> Optional[str]:
        """Generate scatter plot of deposit locations."""
        if not HAS_MATPLOTLIB or len(centroids) == 0:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 10))
        
        if labels is not None:
            colors = {'normal': 'green', 'rod': 'red', 'artifact': 'gray', 'unknown': 'yellow'}
            for label in set(labels):
                mask = np.array(labels) == label
                if mask.any():
                    ax.scatter(
                        centroids[mask, 0], centroids[mask, 1],
                        c=colors.get(label, 'blue'),
                        label=label.capitalize(),
                        alpha=0.6, s=50
                    )
            ax.legend()
        else:
            ax.scatter(centroids[:, 0], centroids[:, 1], alpha=0.6, s=50)
        
        if image_shape:
            ax.set_xlim(0, image_shape[1])
            ax.set_ylim(image_shape[0], 0)  # Invert Y for image coordinates
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def clark_evans_summary(
        self,
        r_values: List[float],
        interpretations: List[str],
        title: str = "Clark-Evans Clustering Index",
        filename: str = "clark_evans_summary.png"
    ) -> Optional[str]:
        """Generate summary of Clark-Evans R values."""
        if not HAS_MATPLOTLIB or len(r_values) == 0:
            return None
        
        fig, axes = _plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of R values
        ax = axes[0]
        ax.hist(r_values, bins=20, color='#9C27B0', edgecolor='white', alpha=0.8)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Random (R=1)')
        ax.axvline(np.mean(r_values), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(r_values):.2f}')
        ax.set_xlabel('Clark-Evans R')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of R Values')
        ax.legend()
        
        # Interpretation pie chart
        ax = axes[1]
        interp_counts = {}
        for interp in interpretations:
            if 'cluster' in interp:
                key = 'Clustered'
            elif interp == 'random':
                key = 'Random'
            elif 'dispers' in interp:
                key = 'Dispersed'
            else:
                key = 'Other'
            interp_counts[key] = interp_counts.get(key, 0) + 1
        
        if interp_counts:
            colors = {'Clustered': '#F44336', 'Random': '#4CAF50', 
                     'Dispersed': '#2196F3', 'Other': '#9E9E9E'}
            ax.pie(
                interp_counts.values(), 
                labels=interp_counts.keys(),
                colors=[colors.get(k, 'gray') for k in interp_counts.keys()],
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Clustering Interpretation')
        
        _plt.suptitle(title)
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)


def generate_spatial_visualizations(
    spatial_results: List,  # List of SpatialResult
    output_dir: Path,
    deposits_by_image: Dict[str, Tuple[np.ndarray, List[str], Tuple[int, int]]] = None
) -> Dict[str, str]:
    """
    Generate all spatial visualizations.
    
    Args:
        spatial_results: List of SpatialResult objects
        output_dir: Output directory
        deposits_by_image: Dict mapping filename to (centroids, labels, image_shape)
    """
    viz = SpatialVisualizer(output_dir)
    results = {}
    
    if not spatial_results:
        return results
    
    # Aggregate NND values
    all_nnd = np.concatenate([r.nnd_values for r in spatial_results if len(r.nnd_values) > 0])
    if len(all_nnd) > 0:
        mean_nnd = np.mean([r.nnd_mean for r in spatial_results if r.nnd_mean > 0])
        path = viz.nnd_histogram(all_nnd, mean_nnd)
        if path:
            results['nnd_histogram'] = path
    
    # Clark-Evans summary
    r_values = [r.clark_evans_r for r in spatial_results if r.clark_evans_r > 0]
    interpretations = [r.clustering_interpretation for r in spatial_results 
                       if r.clustering_interpretation != 'insufficient_data']
    if r_values:
        path = viz.clark_evans_summary(r_values, interpretations)
        if path:
            results['clark_evans'] = path
    
    # Aggregate density map
    if spatial_results:
        agg_density = np.mean([r.density_map for r in spatial_results], axis=0)
        path = viz.density_heatmap(agg_density, title="Average Deposit Density")
        if path:
            results['density_map'] = path
    
    # Aggregate quadrant counts
    total_quadrants = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    for r in spatial_results:
        for q, count in r.quadrant_counts.items():
            total_quadrants[q] += count
    
    path = viz.quadrant_plot(total_quadrants, title="Total Quadrant Distribution")
    if path:
        results['quadrant_plot'] = path
    
    return results
