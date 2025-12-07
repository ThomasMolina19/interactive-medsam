"""Lógica de segmentación con SAM."""

import numpy as np
import cv2
from scipy import ndimage

from .config import IMAGE_ENHANCE_ALPHA, IMAGE_ENHANCE_BETA
from .mask_utils import refine_medical_mask


def segment_with_point(model, img, point, label=1, verbose=False):
    """
    Segmenta una imagen usando un punto.
    
    Args:
        model: Instancia de SAMModel
        img: Imagen RGB numpy array
        point: Coordenadas [x, y] del punto
        label: 1 para positivo, 0 para negativo
        verbose: Si imprime información de debug
        
    Returns:
        tuple: (mask, score) o (None, 0.0) si falla
    """
    # Mejorar contraste de la imagen
    img_enhanced = cv2.convertScaleAbs(img, alpha=IMAGE_ENHANCE_ALPHA, beta=IMAGE_ENHANCE_BETA)
    model.set_image(img_enhanced)
    
    input_point = np.array([point])
    input_label = np.array([label])
    
    try:
        masks, scores, _ = model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        raw_area = np.sum(best_mask)
        if verbose:
            print(f"    🔍 SAM raw mask area: {raw_area} px, score: {best_score:.3f}")
        
        if raw_area == 0:
            if verbose:
                print(f"    ⚠️ SAM no generó máscara (punto fuera del objeto?)")
            return None, 0.0
        
        # Refinar máscara
        refined_mask = refine_medical_mask(best_mask)
        refined_area = np.sum(refined_mask)
        
        # Si el refinamiento eliminó todo, usar máscara raw con fill
        if refined_area == 0 and raw_area > 0:
            if verbose:
                print(f"    ⚠️ Refinamiento eliminó máscara ({raw_area}px < min_size), usando raw")
            refined_mask = ndimage.binary_fill_holes(best_mask)
        
        return refined_mask, best_score
        
    except Exception as e:
        print(f"    ⚠️ Error in segmentation: {e}")
        return None, 0.0


def segment_with_points(model, img, positive_points, negative_points):
    """
    Segmenta una imagen usando múltiples puntos positivos y negativos.
    
    Args:
        model: Instancia de SAMModel
        img: Imagen RGB numpy array
        positive_points: Lista de puntos [x, y] dentro del objeto
        negative_points: Lista de puntos [x, y] fuera del objeto
        
    Returns:
        tuple: (mask, score) o (None, 0.0) si falla
    """
    img_enhanced = cv2.convertScaleAbs(img, alpha=IMAGE_ENHANCE_ALPHA, beta=IMAGE_ENHANCE_BETA)
    model.set_image(img_enhanced)
    
    # Construir arrays de puntos y labels
    input_points = []
    input_labels = []
    
    for point in positive_points:
        input_points.append(point)
        input_labels.append(1)
    
    for point in negative_points:
        input_points.append(point)
        input_labels.append(0)
    
    if len(input_points) == 0:
        return None, 0.0
    
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)
    
    try:
        masks, scores, _ = model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        refined_mask = refine_medical_mask(best_mask)
        
        return refined_mask, best_score
        
    except Exception as e:
        print(f"⚠️  Error in segmentation: {e}")
        return None, 0.0


def segment_with_retry(model, img, reference_center, offsets, verbose=False):
    """
    Intenta segmentar con offsets si falla el punto original.
    
    Args:
        model: Instancia de SAMModel
        img: Imagen RGB numpy array
        reference_center: Punto inicial [x, y]
        offsets: Lista de offsets [(dx, dy), ...] a probar
        verbose: Si imprime información de debug
        
    Returns:
        tuple: (mask, score, used_point) o (None, 0.0, None) si falla todo
    """
    # Intentar con el punto original
    mask, score = segment_with_point(model, img, reference_center, verbose=verbose)
    
    if mask is not None and np.sum(mask) > 0:
        return mask, score, reference_center
    
    # Si falla, intentar con offsets
    if verbose:
        print(f"    🔄 Intentando con centro ajustado...")
    
    h, w = img.shape[:2]
    for dx, dy in offsets[1:]:  # Saltar (0,0) que ya probamos
        adjusted_point = [reference_center[0] + dx, reference_center[1] + dy]
        
        # Verificar que el punto está dentro de la imagen
        if 0 <= adjusted_point[0] < w and 0 <= adjusted_point[1] < h:
            mask, score = segment_with_point(model, img, adjusted_point, verbose=False)
            if mask is not None and np.sum(mask) > 0:
                print(f"    ✅ Encontrada segmentación con offset ({dx}, {dy})")
                return mask, score, adjusted_point
    
    return None, 0.0, None
