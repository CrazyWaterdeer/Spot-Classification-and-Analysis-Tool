"""
Analysis pipeline - combines detection, feature extraction, and classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

from PIL import Image
import cv2

from .detector import DepositDetector, Deposit
from .features import FeatureExtractor
from .classifier import ThresholdClassifier, get_classifier, ClassifierConfig


class AnalysisResult:
    """Container for analysis results of a single image."""
    
    def __init__(self, filename: str, deposits: List[Deposit], dpi: float):
        self.filename = filename
        self.deposits = deposits
        self.dpi = dpi
        self.timestamp = datetime.now().isoformat()
    
    @property
    def n_total(self) -> int:
        return len(self.deposits)
    
    @property
    def n_rod(self) -> int:
        return sum(1 for d in self.deposits if d.label == "rod")
    
    @property
    def n_normal(self) -> int:
        return sum(1 for d in self.deposits if d.label == "normal")
    
    @property
    def n_artifact(self) -> int:
        return sum(1 for d in self.deposits if d.label == "artifact")
    
    @property
    def rod_fraction(self) -> float:
        valid = self.n_rod + self.n_normal
        return self.n_rod / valid if valid > 0 else 0.0
    
    def get_summary(self) -> Dict:
        valid_deposits = [d for d in self.deposits if d.label in ["rod", "normal"]]
        normal_deposits = [d for d in self.deposits if d.label == "normal"]
        rod_deposits = [d for d in self.deposits if d.label == "rod"]
        
        summary = {
            'filename': self.filename,
            'n_total': self.n_total,
            # Order: Normal → ROD → Artifact
            'n_normal': self.n_normal,
            'n_rod': self.n_rod,
            'n_artifact': self.n_artifact,
            'rod_fraction': self.rod_fraction,
        }
        
        # Normal statistics
        if normal_deposits:
            summary['normal_mean_area'] = np.mean([d.area for d in normal_deposits])
            summary['normal_std_area'] = np.std([d.area for d in normal_deposits])
            summary['normal_mean_iod'] = np.mean([d.iod for d in normal_deposits])
            summary['normal_total_iod'] = sum(d.iod for d in normal_deposits)
            summary['normal_mean_hue'] = np.mean([d.mean_hue for d in normal_deposits])
            summary['normal_mean_lightness'] = np.mean([d.mean_lightness for d in normal_deposits])
            summary['normal_mean_circularity'] = np.mean([d.circularity for d in normal_deposits])
        else:
            summary['normal_mean_area'] = np.nan
            summary['normal_std_area'] = np.nan
            summary['normal_mean_iod'] = np.nan
            summary['normal_total_iod'] = 0
            summary['normal_mean_hue'] = np.nan
            summary['normal_mean_lightness'] = np.nan
            summary['normal_mean_circularity'] = np.nan
        
        # ROD statistics
        if rod_deposits:
            summary['rod_mean_area'] = np.mean([d.area for d in rod_deposits])
            summary['rod_std_area'] = np.std([d.area for d in rod_deposits])
            summary['rod_mean_iod'] = np.mean([d.iod for d in rod_deposits])
            summary['rod_total_iod'] = sum(d.iod for d in rod_deposits)
            summary['rod_mean_hue'] = np.mean([d.mean_hue for d in rod_deposits])
            summary['rod_mean_lightness'] = np.mean([d.mean_lightness for d in rod_deposits])
            summary['rod_mean_circularity'] = np.mean([d.circularity for d in rod_deposits])
        else:
            summary['rod_mean_area'] = np.nan
            summary['rod_std_area'] = np.nan
            summary['rod_mean_iod'] = np.nan
            summary['rod_total_iod'] = 0
            summary['rod_mean_hue'] = np.nan
            summary['rod_mean_lightness'] = np.nan
            summary['rod_mean_circularity'] = np.nan
        
        # Total statistics
        if valid_deposits:
            summary['total_iod'] = sum(d.iod for d in valid_deposits)
            summary['mean_area'] = np.mean([d.area for d in valid_deposits])
            summary['mean_iod'] = np.mean([d.iod for d in valid_deposits])
        else:
            summary['total_iod'] = 0
            summary['mean_area'] = np.nan
            summary['mean_iod'] = np.nan
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        extractor = FeatureExtractor(dpi=self.dpi)
        records = [extractor.to_feature_dict(d) for d in self.deposits]
        df = pd.DataFrame(records)
        df['filename'] = self.filename
        return df


class Analyzer:
    """Main analysis pipeline."""
    
    def __init__(
        self,
        detector: Optional[DepositDetector] = None,
        classifier_config: Optional[ClassifierConfig] = None,
        dpi: float = 600.0
    ):
        self.detector = detector or DepositDetector()
        self.classifier_config = classifier_config or ClassifierConfig()
        self.classifier = get_classifier(self.classifier_config)
        self.dpi = dpi
        self.extractor = FeatureExtractor(dpi=dpi)
    
    def analyze_image(self, image_path: Union[str, Path], n_flies: int = 1) -> AnalysisResult:
        image_path = Path(image_path)
        img = Image.open(image_path)
        image = np.array(img)
        
        dpi = img.info.get('dpi', (self.dpi, self.dpi))[0]
        self.extractor = FeatureExtractor(dpi=dpi)
        
        deposits = self.detector.detect(image)
        deposits = self.extractor.extract_features(image, deposits)
        
        # Call predict for each classifier
        from .classifier import ThresholdClassifier, RandomForestClassifier, CNNClassifier
        if isinstance(self.classifier, (ThresholdClassifier, RandomForestClassifier)):
            deposits = self.classifier.predict(deposits)
        elif isinstance(self.classifier, CNNClassifier):
            deposits = self.classifier.predict(deposits, image)
        else:
            # Fallback
            deposits = self.classifier.predict(deposits)
        
        return AnalysisResult(filename=image_path.name, deposits=deposits, dpi=dpi)
    
    def analyze_batch(
        self, image_paths: List[Union[str, Path]],
        metadata: Optional[pd.DataFrame] = None, progress_callback=None
    ) -> List[AnalysisResult]:
        results = []
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
            results.append(self.analyze_image(path))
        return results
    
    def generate_annotated_image(
        self, image: np.ndarray, deposits: List[Deposit], show_labels: bool = True
    ) -> np.ndarray:
        result = image.copy()
        colors = {'rod': (255, 0, 0), 'normal': (0, 255, 0), 'artifact': (128, 128, 128), 'unknown': (255, 255, 0)}
        
        for d in deposits:
            color = colors.get(d.label, colors['unknown'])
            cv2.drawContours(result, [d.contour], -1, color, 2)
            if show_labels:
                cv2.putText(result, f"{d.id}", (d.centroid[0] + 5, d.centroid[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return result


class ReportGenerator:
    """Generate analysis reports."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.deposits_dir = self.output_dir / 'deposits'
        self.deposits_dir.mkdir(exist_ok=True)
    
    def generate_film_summary(self, results: List[AnalysisResult], metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = pd.DataFrame([r.get_summary() for r in results])
        if metadata is not None:
            df = df.merge(metadata, on='filename', how='left')
        return df
    
    def generate_condition_summary(self, film_summary: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
        numeric_cols = film_summary.select_dtypes(include=[np.number]).columns
        agg_funcs = {col: ['mean', 'std', 'count'] for col in numeric_cols if col not in group_by}
        condition_summary = film_summary.groupby(group_by).agg(agg_funcs)
        condition_summary.columns = ['_'.join(col).strip() for col in condition_summary.columns]
        return condition_summary.reset_index()
    
    def generate_deposit_data(self, results: List[AnalysisResult], metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = pd.concat([r.to_dataframe() for r in results], ignore_index=True)
        if metadata is not None:
            df = df.merge(metadata, on='filename', how='left')
        return df
    
    def save_individual_deposits(self, results: List[AnalysisResult], metadata: Optional[pd.DataFrame] = None, save_json: bool = True):
        """Save individual CSV file for each image."""
        for result in results:
            df = result.to_dataframe()
            if metadata is not None:
                df = df.merge(metadata, on='filename', how='left')
            
            # Reorder columns: id, position, Normal→ROD→Artifact related, then rest
            priority_cols = ['id', 'filename', 'x', 'y', 'width', 'height', 'label', 'confidence',
                           'area_px', 'area_um2', 'circularity', 'aspect_ratio',
                           'mean_hue', 'mean_saturation', 'mean_lightness', 'iod']
            
            existing_priority = [c for c in priority_cols if c in df.columns]
            other_cols = [c for c in df.columns if c not in priority_cols]
            df = df[existing_priority + other_cols]
            
            # Save with image name
            image_stem = Path(result.filename).stem
            filepath = self.deposits_dir / f'{image_stem}_deposits.csv'
            df.to_csv(filepath, index=False)
            
            # Optionally save contour data as JSON for retraining
            if save_json:
                self._save_contour_json(result, image_stem)
    
    def _save_contour_json(self, result: AnalysisResult, image_stem: str):
        """Save deposit contours as JSON (unified format with labeling)."""
        import json
        
        deposits_data = []
        for d in result.deposits:
            deposit_dict = {
                'id': d.id,
                'contour': d.contour.squeeze().tolist() if d.contour is not None else [],
                'x': d.centroid[0],
                'y': d.centroid[1],
                'width': d.width,
                'height': d.height,
                'area': float(d.area),
                'circularity': float(d.circularity),
                'label': d.label,
                'confidence': float(d.confidence),
                'merged': getattr(d, 'merged', False),
                'group_id': getattr(d, 'group_id', None)
            }
            deposits_data.append(deposit_dict)
        
        # Use unified format: *.labels.json
        json_path = self.deposits_dir / f'{image_stem}.labels.json'
        with open(json_path, 'w') as f:
            json.dump({
                'image_file': result.filename,
                'next_group_id': 1,
                'deposits': deposits_data
            }, f, indent=2)
    
    def save_all(
        self, 
        results: List[AnalysisResult], 
        metadata: Optional[pd.DataFrame] = None, 
        group_by: Optional[List[str]] = None,
        save_individual: bool = True,
        save_json: bool = True
    ) -> Dict:
        """
        Save all reports.
        
        Args:
            results: List of analysis results
            metadata: Optional metadata DataFrame
            group_by: Columns for condition grouping
            save_individual: Whether to save individual deposit files per image
            save_json: Whether to save JSON files for retraining
        """
        # Film summary
        film_summary = self.generate_film_summary(results, metadata)
        film_summary.to_csv(self.output_dir / 'film_summary.csv', index=False)
        
        # Condition summary
        if group_by and metadata is not None:
            condition_summary = self.generate_condition_summary(film_summary, group_by)
            condition_summary.to_csv(self.output_dir / 'condition_summary.csv', index=False)
        
        # Individual deposit files per image
        if save_individual:
            self.save_individual_deposits(results, metadata, save_json=save_json)
        
        # Combined deposit data (optional, for convenience)
        deposit_data = self.generate_deposit_data(results, metadata)
        deposit_data.to_csv(self.output_dir / 'all_deposits.csv', index=False)
        
        return {
            'film_summary': film_summary, 
            'deposit_data': deposit_data,
            'deposits_dir': str(self.deposits_dir)
        }
