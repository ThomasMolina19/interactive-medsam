"""
Módulo de detección automática del húmero
Fase 2: Detección por circularidad, intensidad y scoring
"""

from .humerus_detector import (
    detect_humerus_by_circularity,
    detect_humerus_by_intensity,
    score_humerus_candidate,
    select_best_candidate,
    generate_bounding_box,
    detect_humerus_automatic,
    filter_candidates_by_proximity,
    correct_first_frame,
    correct_outlier_detection
)

__all__ = [
    'detect_humerus_by_circularity',
    'detect_humerus_by_intensity',
    'score_humerus_candidate',
    'select_best_candidate',
    'generate_bounding_box',
    'detect_humerus_automatic'
]
