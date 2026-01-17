"""
Report generation module for SCAT.
Generates HTML and PDF reports from analysis results.
"""

import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from io import BytesIO

# Check for optional dependencies
try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ReportGenerator:
    """Generate HTML/PDF reports from analysis results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        film_summary: pd.DataFrame,
        deposit_data: pd.DataFrame = None,
        spatial_stats: Dict = None,
        statistical_results: Dict = None,
        visualization_paths: Dict[str, str] = None,
        metadata: pd.DataFrame = None,
        group_by: str = None,
        title: str = "SCAT Analysis Report"
    ) -> str:
        """
        Generate comprehensive HTML report.
        
        Returns:
            Path to generated HTML file
        """
        # Calculate summary statistics
        summary = self._calculate_summary(film_summary)
        
        # Generate inline plots
        inline_plots = {}
        if HAS_MATPLOTLIB:
            inline_plots['overview_bar'] = self._generate_overview_bar(film_summary)
            inline_plots['rod_distribution'] = self._generate_rod_histogram(film_summary)
            if group_by and group_by in film_summary.columns:
                inline_plots['group_comparison'] = self._generate_group_comparison(
                    film_summary, group_by
                )
        
        # Build HTML
        html_content = self._build_html(
            title=title,
            summary=summary,
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            inline_plots=inline_plots,
            group_by=group_by
        )
        
        # Save
        output_path = self.output_dir / 'report.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _calculate_summary(self, film_summary: pd.DataFrame) -> Dict:
        """Calculate overall summary statistics."""
        def safe_sum(col):
            return int(film_summary[col].sum()) if col in film_summary.columns else 0
        
        def safe_mean(col):
            return float(film_summary[col].mean()) if col in film_summary.columns else 0.0
        
        def safe_std(col):
            return float(film_summary[col].std()) if col in film_summary.columns else 0.0
        
        return {
            'total_films': len(film_summary),
            'total_deposits': safe_sum('n_total'),
            'total_normal': safe_sum('n_normal'),
            'total_rod': safe_sum('n_rod'),
            'total_artifact': safe_sum('n_artifact'),
            'mean_rod_fraction': safe_mean('rod_fraction'),
            'std_rod_fraction': safe_std('rod_fraction'),
            'mean_total_iod': safe_mean('total_iod'),
            'mean_normal_area': safe_mean('normal_mean_area'),
            'mean_rod_area': safe_mean('rod_mean_area'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_overview_bar(self, film_summary: pd.DataFrame) -> str:
        """Generate overview bar chart as base64."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        def safe_sum(col):
            return film_summary[col].sum() if col in film_summary.columns else 0
        
        totals = {
            'Normal': safe_sum('n_normal'),
            'ROD': safe_sum('n_rod'),
            'Artifact': safe_sum('n_artifact')
        }
        
        colors = ['#4CAF50', '#F44336', '#9E9E9E']
        bars = ax.bar(totals.keys(), totals.values(), color=colors)
        
        ax.set_ylabel('Count')
        ax.set_title('Total Deposits by Classification')
        
        # Add value labels
        for bar, val in zip(bars, totals.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_rod_histogram(self, film_summary: pd.DataFrame) -> str:
        """Generate ROD fraction histogram as base64."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.hist(film_summary['rod_fraction'] * 100, bins=15, 
                color='#2196F3', edgecolor='white', alpha=0.8)
        ax.axvline(film_summary['rod_fraction'].mean() * 100, 
                   color='red', linestyle='--', linewidth=2, label='Mean')
        
        ax.set_xlabel('ROD Fraction (%)')
        ax.set_ylabel('Number of Films')
        ax.set_title('Distribution of ROD Fraction')
        ax.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_group_comparison(
        self, 
        film_summary: pd.DataFrame, 
        group_by: str
    ) -> str:
        """Generate group comparison box plot as base64."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        groups = film_summary[group_by].unique()
        data = [film_summary[film_summary[group_by] == g]['rod_fraction'] * 100 
                for g in groups]
        
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('ROD Fraction (%)')
        ax.set_xlabel(group_by)
        ax.set_title(f'ROD Fraction by {group_by}')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _build_html(
        self,
        title: str,
        summary: Dict,
        film_summary: pd.DataFrame,
        deposit_data: pd.DataFrame,
        spatial_stats: Dict,
        statistical_results: Dict,
        visualization_paths: Dict,
        inline_plots: Dict,
        group_by: str
    ) -> str:
        """Build complete HTML document."""
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #1a237e;
            border-bottom: 3px solid #3949ab;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3949ab;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #1a237e;
        }}
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .stat-card.normal {{ border-left-color: #4CAF50; }}
        .stat-card.normal .value {{ color: #4CAF50; }}
        .stat-card.rod {{ border-left-color: #F44336; }}
        .stat-card.rod .value {{ color: #F44336; }}
        .stat-card.artifact {{ border-left-color: #9E9E9E; }}
        .stat-card.artifact .value {{ color: #9E9E9E; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #1a237e;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Generated: {summary['generated_at']}</div>
    </div>
    
    <!-- Summary Section -->
    <div class="section">
        <h2>📊 Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{summary['total_films']}</div>
                <div class="label">Total Films</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['total_deposits']}</div>
                <div class="label">Total Deposits</div>
            </div>
            <div class="stat-card normal">
                <div class="value">{summary['total_normal']}</div>
                <div class="label">Normal</div>
            </div>
            <div class="stat-card rod">
                <div class="value">{summary['total_rod']}</div>
                <div class="label">ROD</div>
            </div>
            <div class="stat-card artifact">
                <div class="value">{summary['total_artifact']}</div>
                <div class="label">Artifact</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['mean_rod_fraction']*100:.1f}%</div>
                <div class="label">Mean ROD Fraction (±{summary['std_rod_fraction']*100:.1f}%)</div>
            </div>
        </div>
'''
        
        # Add inline plots
        if 'overview_bar' in inline_plots:
            html += f'''
        <div class="two-column">
            <div class="plot-container">
                <img src="data:image/png;base64,{inline_plots['overview_bar']}" alt="Overview">
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{inline_plots['rod_distribution']}" alt="ROD Distribution">
            </div>
        </div>
'''
        
        html += '    </div>\n'
        
        # Group comparison
        if group_by and 'group_comparison' in inline_plots:
            html += f'''
    <div class="section">
        <h2>📈 Condition Comparison ({group_by})</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{inline_plots['group_comparison']}" alt="Group Comparison">
        </div>
    </div>
'''
        
        # Statistical results
        if statistical_results and isinstance(statistical_results, dict):
            html += '''
    <div class="section">
        <h2>📉 Statistical Analysis</h2>
'''
            for metric, result in statistical_results.items():
                # Skip if result is not a dict or has error
                if not isinstance(result, dict):
                    continue
                if 'error' in result:
                    continue
                
                html += f'        <h3 style="color:#3949ab; margin-top:20px;">{metric}</h3>\n'
                
                if 'overall_test' in result:
                    html += f'''        <p><strong>Test:</strong> {result['overall_test']}</p>
        <p><strong>p-value:</strong> {result['overall_p_value']:.4f} 
        {'✅ Significant' if result['overall_significant'] else '❌ Not significant'}</p>
'''
                else:
                    html += f'''        <p><strong>Test:</strong> {result.get('test_name', 'N/A')}</p>
        <p><strong>{result.get('group1_name', 'Group1')}:</strong> {result.get('mean1', 0):.3f} ± {result.get('std1', 0):.3f}</p>
        <p><strong>{result.get('group2_name', 'Group2')}:</strong> {result.get('mean2', 0):.3f} ± {result.get('std2', 0):.3f}</p>
        <p><strong>p-value:</strong> {result.get('p_value', 0):.4f}</p>
        <p><strong>Effect size (Cohen's d):</strong> {result.get('cohens_d', 0):.2f} ({result.get('effect_size', 'N/A')})</p>
'''
            html += '    </div>\n'
        
        # Spatial statistics
        if spatial_stats:
            html += f'''
    <div class="section">
        <h2>🗺️ Spatial Analysis</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{spatial_stats.get('mean_nnd', 0):.1f}</div>
                <div class="label">Mean NND (pixels)</div>
            </div>
            <div class="stat-card">
                <div class="value">{spatial_stats.get('mean_clark_evans', 0):.2f}</div>
                <div class="label">Clark-Evans R</div>
            </div>
            <div class="stat-card">
                <div class="value">{spatial_stats.get('mean_edge_fraction', 0)*100:.1f}%</div>
                <div class="label">Edge Fraction</div>
            </div>
        </div>
        <div class="highlight">
            <strong>Clustering:</strong> 
            {spatial_stats.get('n_clustered', 0)} clustered, 
            {spatial_stats.get('n_random', 0)} random, 
            {spatial_stats.get('n_dispersed', 0)} dispersed 
            (out of {spatial_stats.get('n_images', 0)} images)
        </div>
    </div>
'''
        
        # Film summary table
        html += '''
    <div class="section">
        <h2>📋 Film Summary</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Normal</th>
                        <th>ROD</th>
                        <th>Artifact</th>
                        <th>ROD %</th>
                        <th>Total IOD</th>
                    </tr>
                </thead>
                <tbody>
'''
        for _, row in film_summary.iterrows():
            n_normal = int(row.get('n_normal', 0)) if 'n_normal' in row.index else 0
            n_rod = int(row.get('n_rod', 0)) if 'n_rod' in row.index else 0
            n_artifact = int(row.get('n_artifact', 0)) if 'n_artifact' in row.index else 0
            rod_fraction = row.get('rod_fraction', 0) if 'rod_fraction' in row.index else 0
            total_iod = row.get('total_iod', 0) if 'total_iod' in row.index else 0
            
            html += f'''                    <tr>
                        <td>{row.get('filename', 'N/A')}</td>
                        <td>{n_normal}</td>
                        <td>{n_rod}</td>
                        <td>{n_artifact}</td>
                        <td>{rod_fraction*100:.1f}%</td>
                        <td>{total_iod:.0f}</td>
                    </tr>
'''
        
        html += '''                </tbody>
            </table>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated by SCAT - Spot Classification and Analysis Tool</p>
        <p>Anthropic Claude AI Assistant</p>
    </div>
</body>
</html>'''
        
        return html
    
    def generate_pdf_report(
        self,
        film_summary: pd.DataFrame,
        **kwargs
    ) -> Optional[str]:
        """
        Generate PDF report (requires weasyprint or pdfkit).
        Falls back to HTML if PDF libraries not available.
        """
        # First generate HTML
        html_path = self.generate_html_report(film_summary, **kwargs)
        
        # Try to convert to PDF
        pdf_path = self.output_dir / 'report.pdf'
        
        try:
            import weasyprint
            weasyprint.HTML(html_path).write_pdf(str(pdf_path))
            return str(pdf_path)
        except ImportError:
            pass
        
        try:
            import pdfkit
            pdfkit.from_file(html_path, str(pdf_path))
            return str(pdf_path)
        except ImportError:
            pass
        
        print("PDF generation requires 'weasyprint' or 'pdfkit'. HTML report generated instead.")
        return html_path


def generate_report(
    film_summary: pd.DataFrame,
    output_dir: Union[str, Path],
    deposit_data: pd.DataFrame = None,
    spatial_stats: Dict = None,
    statistical_results: Dict = None,
    visualization_paths: Dict = None,
    group_by: str = None,
    format: str = 'html'
) -> str:
    """
    Convenience function to generate report.
    
    Args:
        film_summary: Film-level summary DataFrame
        output_dir: Output directory
        format: 'html' or 'pdf'
        
    Returns:
        Path to generated report
    """
    generator = ReportGenerator(output_dir)
    
    if format == 'pdf':
        return generator.generate_pdf_report(
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            group_by=group_by
        )
    else:
        return generator.generate_html_report(
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            group_by=group_by
        )
