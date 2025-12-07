"""
Módulo de segmentación con SAM usando propagación de centros.
"""

from .config import SIMILARITY_THRESHOLD, WARNING_THRESHOLD, IMAGE_EXTENSIONS
from .model import SAMModel
from .mask_utils import (
    refine_medical_mask,
    calculate_mask_center,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_negative_point
)
from .segmentation import segment_with_point, segment_with_points, segment_with_retry
from .propagation import propagate_direction, read_image_file
from .visualization import save_segmentation_result, save_summary
from .ui import get_user_paths
from .interactive import interactive_segment

__all__ = [
    'SIMILARITY_THRESHOLD',
    'WARNING_THRESHOLD',
    'IMAGE_EXTENSIONS',
    'SAMModel',
    'refine_medical_mask',
    'calculate_mask_center',
    'calculate_dice_coefficient',
    'calculate_iou',
    'calculate_negative_point',
    'segment_with_point',
    'segment_with_points',
    'segment_with_retry',
    'propagate_direction',
    'read_image_file',
    'save_segmentation_result',
    'save_summary',
    'get_user_paths',
    'interactive_segment',
]
