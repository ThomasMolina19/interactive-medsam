"""
Pipeline de Segmentación Automática Completa
Integra: Fase 1 (Preprocesamiento) + Fase 2 (Detección) + SAM (Segmentación)
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from scipy import ndimage
from skimage import morphology

# Importar fases anteriores
from src.preprocessing import enhance_bone_contrast, detect_bone_edges
from src.detection import detect_humerus_automatic


def refine_medical_mask(mask: np.ndarray) -> np.ndarray:
    """
    Refina la máscara de segmentación con operaciones morfológicas.
    
    Args:
        mask: Máscara binaria de SAM
    
    Returns:
        Máscara refinada
    """
    # Remover objetos pequeños
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    
    # Rellenar huecos
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Suavizado morfológico
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth


def automatic_humerus_segmentation(
    img: np.ndarray,
    predictor,
    use_preprocessing: bool = True,
    use_auto_detection: bool = True,
    fallback_box: Optional[np.ndarray] = None,
    return_intermediate: bool = False
) -> Dict:
    """
    Pipeline completo de segmentación automática del húmero.
    
    Flujo:
    1. Fase 1: Preprocesamiento (CLAHE + bordes)
    2. Fase 2: Detección automática (Hough + scoring)
    3. SAM: Segmentación con caja automática
    4. Post-procesamiento: Refinamiento de máscara
    
    Args:
        img: Imagen de entrada (RGB)
        predictor: Predictor de SAM ya inicializado
        use_preprocessing: Aplicar Fase 1
        use_auto_detection: Aplicar Fase 2
        fallback_box: Caja de respaldo si falla detección
        return_intermediate: Retornar resultados intermedios
    
    Returns:
        Diccionario con:
        - 'success': bool
        - 'mask': Máscara final
        - 'box': Bounding box usada
        - 'confidence': Confianza de detección
        - 'method': Método usado ('automatic' o 'fallback')
        - 'intermediate': Resultados intermedios (opcional)
        
    Example:
        >>> from segment_anything import sam_model_registry, SamPredictor
        >>> sam = sam_model_registry["vit_b"]()
        >>> sam.load_state_dict(torch.load("medsam_vit_b.pth"))
        >>> predictor = SamPredictor(sam)
        >>> 
        >>> result = automatic_humerus_segmentation(img, predictor)
        >>> if result['success']:
        >>>     mask = result['mask']
        >>>     confidence = result['confidence']
    """
    result = {
        'success': False,
        'mask': None,
        'box': None,
        'confidence': 0.0,
        'method': None,
        'sam_score': 0.0,
        'intermediate': {} if return_intermediate else None
    }
    
    # ========== FASE 1: PREPROCESAMIENTO ==========
    if use_preprocessing:
        img_enhanced = enhance_bone_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8))
        if return_intermediate:
            result['intermediate']['img_enhanced'] = img_enhanced
            result['intermediate']['edges'] = detect_bone_edges(img_enhanced)
    else:
        img_enhanced = img.copy()
    
    # ========== FASE 2: DETECCIÓN AUTOMÁTICA ==========
    box = None
    detection_confidence = 0.0
    
    if use_auto_detection:
        detection = detect_humerus_automatic(
            img, 
            img_enhanced, 
            method='combined',
            return_all_candidates=return_intermediate
        )
        
        if detection['success']:
            box = detection['box']
            detection_confidence = detection['confidence']
            result['method'] = 'automatic'
            
            if return_intermediate:
                result['intermediate']['detection'] = detection
        else:
            # Detección falló, usar fallback si está disponible
            if fallback_box is not None:
                box = fallback_box
                result['method'] = 'fallback'
            else:
                # Sin fallback, usar región central por defecto
                h, w = img.shape[:2]
                margin = min(w, h) // 4
                box = np.array([margin, margin, w - margin, h - margin])
                result['method'] = 'default'
    else:
        # No usar detección automática
        if fallback_box is not None:
            box = fallback_box
            result['method'] = 'manual'
        else:
            # Región central por defecto
            h, w = img.shape[:2]
            margin = min(w, h) // 4
            box = np.array([margin, margin, w - margin, h - margin])
            result['method'] = 'default'
    
    result['box'] = box
    result['confidence'] = detection_confidence
    
    # ========== SAM: SEGMENTACIÓN ==========
    try:
        # Configurar imagen en predictor
        predictor.set_image(img_enhanced)
        
        # Predecir con la caja
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=True
        )
        
        # Seleccionar mejor máscara
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        if return_intermediate:
            result['intermediate']['all_masks'] = masks
            result['intermediate']['all_scores'] = scores
        
        # ========== POST-PROCESAMIENTO ==========
        refined_mask = refine_medical_mask(best_mask)
        
        result['success'] = True
        result['mask'] = refined_mask
        result['sam_score'] = float(best_score)
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
    
    return result


def batch_automatic_segmentation(
    image_paths: list,
    predictor,
    save_masks: bool = True,
    output_dir: Optional[str] = None
) -> list:
    """
    Segmentación automática en lote de múltiples imágenes.
    
    Args:
        image_paths: Lista de rutas a imágenes
        predictor: Predictor de SAM
        save_masks: Guardar máscaras
        output_dir: Directorio de salida
    
    Returns:
        Lista de resultados
    """
    from PIL import Image
    from pathlib import Path
    
    results = []
    
    if save_masks and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for img_path in image_paths:
        try:
            # Cargar imagen
            img = np.array(Image.open(img_path).convert("RGB"))
            
            # Segmentar
            result = automatic_humerus_segmentation(img, predictor)
            
            result['image_path'] = str(img_path)
            result['image_name'] = Path(img_path).name
            
            # Guardar máscara si se solicita
            if save_masks and output_dir and result['success']:
                mask_path = Path(output_dir) / f"mask_{Path(img_path).stem}.png"
                mask_img = Image.fromarray((result['mask'] * 255).astype(np.uint8))
                mask_img.save(mask_path)
                result['mask_path'] = str(mask_path)
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'success': False,
                'image_path': str(img_path),
                'image_name': Path(img_path).name,
                'error': str(e)
            })
    
    return results
