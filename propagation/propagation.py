"""Lógica de propagación de segmentación a través de imágenes."""

import os
import numpy as np
from PIL import Image

from .config import SIMILARITY_THRESHOLD, WARNING_THRESHOLD, RETRY_OFFSETS, NEGATIVE_POINT_DISTANCE_FACTOR
from .mask_utils import (
    calculate_mask_center,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_negative_point
)
from .segmentation import segment_with_retry, segment_with_points
from .visualization import save_segmentation_result


def read_image_file(filepath):
    """
    Lee un archivo de imagen y retorna array RGB.
    
    Args:
        filepath: Ruta al archivo de imagen
        
    Returns:
        numpy array RGB o None si falla
    """
    try:
        img = np.array(Image.open(filepath).convert("RGB"))
        return img
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return None


def propagate_direction(model, files, start_idx, reference_mask, reference_center,
                        output_dir, direction='forward'):
    """
    Propaga segmentación en una dirección (hacia arriba o abajo).
    
    Args:
        model: Instancia de SAMModel
        files: Lista de rutas de archivos ordenados
        start_idx: Índice de la imagen inicial (medio)
        reference_mask: Máscara de referencia inicial
        reference_center: Centro de la máscara inicial [x, y]
        output_dir: Directorio de salida
        direction: 'forward' (hacia abajo) o 'backward' (hacia arriba)
        
    Returns:
        dict: Diccionario de segmentaciones {idx: {...}}
    """
    segmentations = {}
    
    # Configurar dirección
    if direction == 'backward':
        indices = range(start_idx - 1, -1, -1)
        label = "ARRIBA"
    else:
        indices = range(start_idx + 1, len(files))
        label = "ABAJO"
    
    print(f"\n📤 PROPAGACIÓN HACIA {label}...")
    
    current_ref_mask = reference_mask
    current_ref_center = reference_center
    
    for idx in indices:
        file_path = files[idx]
        filename = os.path.basename(file_path).split('.')[0]
        
        print(f"\n  [{idx+1}/{len(files)}] Procesando {os.path.basename(file_path)}...")
        
        # Leer imagen
        img = read_image_file(file_path)
        if img is None:
            print(f"    ❌ Error leyendo imagen. Saltando...")
            continue
        
        # Intentar segmentación
        mask, score, used_point = segment_with_retry(
            model, img, current_ref_center, RETRY_OFFSETS, verbose=True
        )
        
        if mask is None:
            print(f"    ❌ Segmentación falló en todos los intentos. Saltando imagen...")
            continue
        
        # Calcular métricas de similitud
        dice = calculate_dice_coefficient(current_ref_mask, mask)
        iou = calculate_iou(current_ref_mask, mask)
        difference = 1.0 - dice
        
        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        
        # Reintentar con punto negativo si hay diferencia severa
        used_neg_point = None
        if difference > WARNING_THRESHOLD:
            mask, score, dice, iou, used_neg_point = _retry_with_negative_point(
                model, img, mask, score, current_ref_mask, current_ref_center, dice
            )
            difference = 1.0 - dice
        elif difference > SIMILARITY_THRESHOLD:
            print(f"    ⚠️  Advertencia: Diferencia ({difference*100:.1f}%) > umbral ({SIMILARITY_THRESHOLD*100:.0f}%)")
        
        # Calcular nuevo centro
        center = calculate_mask_center(mask)
        if center is None:
            print(f"    ⚠️  No se pudo calcular centro. Usando centro anterior.")
            center = current_ref_center
        
        print(f"    ✅ Válida. Centro: ({center[0]:.0f}, {center[1]:.0f})")
        
        # Guardar segmentación
        used_seg_point = list(current_ref_center)
        
        segmentations[idx] = {
            'mask': mask,
            'center': center,
            'seg_point': used_seg_point,
            'score': score,
            'area': np.sum(mask),
            'dice': dice,
            'iou': iou
        }
        
        # Guardar visualización
        save_segmentation_result(
            img, mask, filename, output_dir,
            center=center, seg_point=used_seg_point, neg_point=used_neg_point,
            info=f"Dice: {dice:.3f} | Score: {score:.3f}"
        )
        
        # Actualizar referencia para siguiente iteración
        current_ref_mask = mask
        current_ref_center = center
    
    return segmentations


def _retry_with_negative_point(model, img, mask, score, ref_mask, ref_center, dice):
    """
    Reintenta segmentación con punto negativo para mejorar resultado.
    
    Args:
        model: Instancia de SAMModel
        img: Imagen RGB numpy array
        mask: Máscara actual
        score: Score actual
        ref_mask: Máscara de referencia
        ref_center: Centro de referencia
        dice: Dice actual
        
    Returns:
        tuple: (mask, score, dice, iou, neg_point_used)
    """
    print(f"    🚨 ADVERTENCIA SEVERA: Diferencia ({(1-dice)*100:.1f}%) > {WARNING_THRESHOLD*100:.0f}%")
    print(f"    🔄 Reintentando con punto negativo...")
    
    neg_point = calculate_negative_point(ref_mask, ref_center, distance_factor=NEGATIVE_POINT_DISTANCE_FACTOR)
    used_neg_point = None
    
    if neg_point is not None:
        print(f"    📍 Punto negativo: ({neg_point[0]:.0f}, {neg_point[1]:.0f})")
        
        mask_retry, score_retry = segment_with_points(
            model, img,
            positive_points=[ref_center],
            negative_points=[neg_point]
        )
        
        if mask_retry is not None and np.sum(mask_retry) > 0:
            dice_retry = calculate_dice_coefficient(ref_mask, mask_retry)
            iou_retry = calculate_iou(ref_mask, mask_retry)
            difference_retry = 1.0 - dice_retry
            
            print(f"    📊 Nuevo Dice: {dice_retry:.3f}, IoU: {iou_retry:.3f}, Diferencia: {difference_retry*100:.1f}%")
            
            if dice_retry > dice:
                print(f"    ✅ Segmentación mejoró de Dice {dice:.3f} a {dice_retry:.3f}")
                return mask_retry, score_retry, dice_retry, iou_retry, list(neg_point)
            else:
                print(f"    ⚠️ La nueva segmentación no mejoró, manteniendo original")
        else:
            print(f"    ⚠️ Reintento falló, manteniendo segmentación original")
    else:
        print(f"    ⚠️ No se pudo calcular punto negativo válido")
    
    iou = calculate_iou(ref_mask, mask)
    return mask, score, dice, iou, used_neg_point
