"""Utilidades para procesamiento de máscaras de segmentación."""

import numpy as np
from scipy import ndimage
from skimage import morphology

from .config import MIN_MASK_SIZE, DISK_RADIUS


def refine_medical_mask(mask, min_size=None, disk_radius=None):
    """
    Limpia y refina una máscara de segmentación médica.
    
    Args:
        mask: Máscara binaria numpy array
        min_size: Tamaño mínimo de objetos a mantener
        disk_radius: Radio del disco para operaciones morfológicas
        
    Returns:
        Máscara binaria refinada
    """
    if min_size is None:
        min_size = MIN_MASK_SIZE
    if disk_radius is None:
        disk_radius = DISK_RADIUS
        
    if np.sum(mask) == 0:
        return mask
    
    # Remover objetos pequeños
    mask_clean = morphology.remove_small_objects(mask, min_size=min_size)
    
    # Rellenar huecos
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Suavizar bordes con operaciones morfológicas
    kernel = morphology.disk(disk_radius)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth


def calculate_mask_center(mask):
    """
    Calcula el centroide de una máscara binaria.
    
    Args:
        mask: Máscara binaria numpy array
        
    Returns:
        list: [center_x, center_y] o None si la máscara está vacía
    """
    if np.sum(mask) == 0:
        return None
    
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    center_x = float(np.mean(x_coords))
    center_y = float(np.mean(y_coords))
    
    return [center_x, center_y]


def calculate_dice_coefficient(mask1, mask2):
    """
    Calcula el coeficiente Dice entre dos máscaras.
    
    Args:
        mask1: Primera máscara binaria
        mask2: Segunda máscara binaria
        
    Returns:
        float: Coeficiente Dice (0-1)
    """
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / sum_masks


def calculate_iou(mask1, mask2):
    """
    Calcula Intersection over Union entre dos máscaras.
    
    Args:
        mask1: Primera máscara binaria
        mask2: Segunda máscara binaria
        
    Returns:
        float: IoU (0-1)
    """
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def calculate_negative_point(mask, center, distance_factor=0.30):
    """
    Calcula un punto negativo fuera de la máscara a una distancia proporcional.
    
    Args:
        mask: Máscara binaria numpy array
        center: Centro de la máscara [x, y]
        distance_factor: Factor de distancia desde el borde (0.30 = 30% más allá)
        
    Returns:
        list: [x, y] del punto negativo o None si no se encuentra
    """
    if mask is None or np.sum(mask) == 0 or center is None:
        return None
    
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    # Calcular dimensiones de la máscara
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    radius = max(width, height) / 2
    
    # Calcular offset desde el centro
    offset = radius * (1 + distance_factor)
    
    h, w = mask.shape
    center_x, center_y = center
    
    # Direcciones a probar
    directions = [
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
    ]
    
    # Probar con diferentes multiplicadores
    for multiplier in [1.0, 1.5]:
        for dx, dy in directions:
            neg_x = center_x + dx * offset * multiplier
            neg_y = center_y + dy * offset * multiplier
            
            # Verificar que está dentro de la imagen y fuera de la máscara
            if 0 <= neg_x < w and 0 <= neg_y < h:
                if not mask[int(neg_y), int(neg_x)]:
                    return [neg_x, neg_y]
    
    return None
