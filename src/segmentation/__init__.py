"""
Módulo de segmentación automática completa
Pipeline integrado: Fase 1 + Fase 2 + SAM
"""

from .automatic_segmentation import automatic_humerus_segmentation

__all__ = ['automatic_humerus_segmentation']
