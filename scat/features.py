"""
Feature extraction module for deposit analysis.
"""

import cv2
import numpy as np
from typing import List
from .detector import Deposit


class FeatureExtractor:
    """Extracts color and morphological features from detected deposits."""
    
    def __init__(self, dpi: float = 600.0):
        self.dpi = dpi
        self.pixels_per_um = dpi / 25400
    
    def extract_features(self, image: np.ndarray, deposits: List[Deposit]) -> List[Deposit]:
        """Extract color features for all deposits.
        
        Optimized: Reuses mask array instead of reallocating for each deposit.
        """
        if not deposits:
            return deposits
        
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        
        # Pre-allocate mask once (memory optimization)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for deposit in deposits:
            # Clear and reuse mask instead of reallocating
            mask.fill(0)
            cv2.drawContours(mask, [deposit.contour], -1, 255, -1)
            
            # Boolean mask for indexing
            mask_bool = mask > 0
            
            pixels_rgb = image[mask_bool]
            if len(pixels_rgb) > 0:
                deposit.mean_r = pixels_rgb[:, 0].mean() / 255.0
                deposit.mean_g = pixels_rgb[:, 1].mean() / 255.0
                deposit.mean_b = pixels_rgb[:, 2].mean() / 255.0
            
            pixels_hls = hls[mask_bool]
            if len(pixels_hls) > 0:
                deposit.mean_hue = pixels_hls[:, 0].mean() * 2
                deposit.mean_lightness = pixels_hls[:, 1].mean() / 255.0
                deposit.mean_saturation = pixels_hls[:, 2].mean() / 255.0
            
            deposit.iod = deposit.area * (1 - deposit.mean_lightness)
        
        return deposits
    
    def area_to_um2(self, area_pixels: float) -> float:
        """Convert area from pixels to square micrometers."""
        return area_pixels / (self.pixels_per_um ** 2)
    
    def to_feature_dict(self, deposit: Deposit) -> dict:
        """Convert deposit to feature dictionary."""
        return {
            'id': deposit.id,
            'x': deposit.centroid[0],
            'y': deposit.centroid[1],
            'width': deposit.width,
            'height': deposit.height,
            'area_px': deposit.area,
            'area_um2': self.area_to_um2(deposit.area),
            'perimeter': deposit.perimeter,
            'circularity': deposit.circularity,
            'aspect_ratio': deposit.aspect_ratio,
            'mean_hue': deposit.mean_hue,
            'mean_saturation': deposit.mean_saturation,
            'mean_lightness': deposit.mean_lightness,
            'mean_r': deposit.mean_r,
            'mean_g': deposit.mean_g,
            'mean_b': deposit.mean_b,
            'iod': deposit.iod,
            'label': deposit.label,
            'confidence': deposit.confidence
        }


def estimate_ph(hue: float) -> str:
    """Estimate pH category from BPB hue value."""
    if hue < 60:
        return "acidic"
    elif hue < 150:
        return "transitional"
    else:
        return "neutral/basic"
