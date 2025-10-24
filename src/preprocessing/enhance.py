"""
Módulo de mejora de imágenes médicas
Implementa técnicas avanzadas de preprocesamiento para resaltar estructuras óseas
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# TAREA 1.1: CLAHE para Contraste
# ============================================================================

def enhance_bone_contrast(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    convert_to_rgb: bool = True
) -> np.ndarray:
    """
    Mejora el contraste de la imagen usando CLAHE (Contrast Limited Adaptive 
    Histogram Equalization) para resaltar mejor el húmero y sus bordes.
    
    CLAHE es superior al ajuste lineal porque:
    - Mejora el contraste local en lugar de global
    - Evita la sobre-amplificación del ruido
    - Preserva mejor los detalles en regiones oscuras y claras
    
    Args:
        img: Imagen de entrada (RGB o escala de grises)
        clip_limit: Límite de contraste para evitar sobre-amplificación (default: 2.0)
        tile_grid_size: Tamaño de la cuadrícula para procesamiento local (default: 8x8)
        convert_to_rgb: Si True, convierte resultado a RGB (default: True)
    
    Returns:
        Imagen con contraste mejorado
        
    Example:
        >>> img = cv2.imread('medical_image.png')
        >>> enhanced = enhance_bone_contrast(img)
    """
    # Convertir a escala de grises si es necesario
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Crear objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Aplicar CLAHE
    enhanced = clahe.apply(gray)
    
    # Convertir de vuelta a RGB si se solicita (para compatibilidad con SAM)
    if convert_to_rgb:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced


def enhance_bone_contrast_color(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica CLAHE preservando información de color (si existe).
    Útil para imágenes médicas con mapas de color.
    
    Args:
        img: Imagen RGB de entrada
        clip_limit: Límite de contraste
        tile_grid_size: Tamaño de cuadrícula
    
    Returns:
        Imagen RGB con contraste mejorado
    """
    if len(img.shape) == 2:
        # Si es escala de grises, usar función simple
        return enhance_bone_contrast(img, clip_limit, tile_grid_size, convert_to_rgb=True)
    
    # Convertir a espacio LAB (mejor para procesamiento de contraste)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Separar canales
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE solo al canal de luminancia (L)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Recombinar canales
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convertir de vuelta a RGB
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced


# ============================================================================
# TAREA 1.2: Detección de Bordes
# ============================================================================

def detect_bone_edges(
    img_enhanced: np.ndarray,
    gaussian_kernel: Tuple[int, int] = (5, 5),
    gaussian_sigma: float = 0,
    canny_threshold1: int = 30,
    canny_threshold2: int = 100
) -> np.ndarray:
    """
    Detecta bordes del húmero usando filtro Canny con suavizado Gaussiano previo.
    
    El proceso es:
    1. Suavizado Gaussiano para reducir ruido
    2. Detección de bordes Canny
    3. Los bordes del húmero deben aparecer claramente
    
    Args:
        img_enhanced: Imagen mejorada (de enhance_bone_contrast)
        gaussian_kernel: Tamaño del kernel Gaussiano (debe ser impar)
        gaussian_sigma: Desviación estándar del Gaussiano (0 = auto)
        canny_threshold1: Umbral inferior para Canny
        canny_threshold2: Umbral superior para Canny
    
    Returns:
        Imagen binaria con bordes detectados (255 = borde, 0 = fondo)
        
    Example:
        >>> enhanced = enhance_bone_contrast(img)
        >>> edges = detect_bone_edges(enhanced)
    """
    # Convertir a escala de grises si es necesario
    if len(img_enhanced.shape) == 3:
        gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_enhanced.copy()
    
    # Suavizado Gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, gaussian_kernel, gaussian_sigma)
    
    # Detección de bordes Canny
    edges = cv2.Canny(blurred, threshold1=canny_threshold1, threshold2=canny_threshold2)
    
    return edges


def detect_bone_edges_advanced(
    img_enhanced: np.ndarray,
    use_morphology: bool = True,
    morph_kernel_size: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detección avanzada de bordes con post-procesamiento morfológico.
    
    Args:
        img_enhanced: Imagen mejorada
        use_morphology: Aplicar operaciones morfológicas para limpiar bordes
        morph_kernel_size: Tamaño del kernel morfológico
    
    Returns:
        Tuple de (edges_raw, edges_cleaned)
    """
    # Detección básica
    edges = detect_bone_edges(img_enhanced)
    
    if not use_morphology:
        return edges, edges
    
    # Operaciones morfológicas para limpiar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    
    # Cerrar gaps pequeños en los bordes
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Eliminar ruido pequeño
    edges_cleaned = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return edges, edges_cleaned


# ============================================================================
# TAREA 1.3: Normalización Adaptativa
# ============================================================================

def normalize_adaptive(
    img: np.ndarray,
    block_size: int = 51,
    c: int = 2,
    method: str = 'gaussian'
) -> np.ndarray:
    """
    Normaliza la intensidad de la imagen de forma adaptativa por regiones.
    Compensa variaciones de iluminación en diferentes partes de la imagen.
    
    Args:
        img: Imagen de entrada
        block_size: Tamaño de la región para normalización local (debe ser impar)
        c: Constante sustraída de la media (ajuste fino)
        method: 'gaussian' o 'mean' para el tipo de umbralización adaptativa
    
    Returns:
        Imagen con intensidad normalizada
        
    Example:
        >>> img = cv2.imread('medical_image.png', 0)
        >>> normalized = normalize_adaptive(img)
    """
    # Convertir a escala de grises si es necesario
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Asegurar que block_size es impar
    if block_size % 2 == 0:
        block_size += 1
    
    # Seleccionar método
    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    
    # Aplicar umbralización adaptativa para obtener máscara de normalización
    # Esto identifica regiones con diferentes niveles de intensidad
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c
    )
    
    # Normalización por regiones usando la máscara
    # Calcular media local
    mean_local = cv2.blur(gray.astype(np.float32), (block_size, block_size))
    
    # Calcular desviación estándar local
    mean_sq_local = cv2.blur((gray.astype(np.float32) ** 2), (block_size, block_size))
    std_local = np.sqrt(np.maximum(mean_sq_local - mean_local ** 2, 0))
    
    # Evitar división por cero
    std_local = np.maximum(std_local, 1.0)
    
    # Normalizar: (img - media_local) / std_local
    normalized = (gray.astype(np.float32) - mean_local) / std_local
    
    # Reescalar a rango [0, 255]
    normalized = ((normalized - normalized.min()) / 
                  (normalized.max() - normalized.min()) * 255)
    
    return normalized.astype(np.uint8)


def normalize_adaptive_clahe_combined(
    img: np.ndarray,
    clahe_clip_limit: float = 2.0,
    norm_block_size: int = 51
) -> np.ndarray:
    """
    Combina normalización adaptativa con CLAHE para mejores resultados.
    Pipeline recomendado para imágenes médicas con variaciones de intensidad.
    
    Args:
        img: Imagen de entrada
        clahe_clip_limit: Límite de contraste para CLAHE
        norm_block_size: Tamaño de bloque para normalización
    
    Returns:
        Imagen procesada con ambas técnicas
    """
    # Paso 1: Normalización adaptativa
    normalized = normalize_adaptive(img, block_size=norm_block_size)
    
    # Paso 2: CLAHE para mejorar contraste local
    enhanced = enhance_bone_contrast(
        normalized, 
        clip_limit=clahe_clip_limit,
        convert_to_rgb=False
    )
    
    return enhanced


# ============================================================================
# FUNCIÓN PIPELINE COMPLETO - FASE 1
# ============================================================================

def preprocess_medical_image(
    img: np.ndarray,
    use_clahe: bool = True,
    use_normalization: bool = False,
    detect_edges: bool = False,
    return_all_steps: bool = False
) -> dict:
    """
    Pipeline completo de preprocesamiento - Fase 1.
    
    Aplica todas las mejoras de la Fase 1 según configuración.
    
    Args:
        img: Imagen de entrada (RGB o escala de grises)
        use_clahe: Aplicar CLAHE para contraste
        use_normalization: Aplicar normalización adaptativa
        detect_edges: Detectar bordes
        return_all_steps: Retornar resultados intermedios
    
    Returns:
        Diccionario con resultados:
        - 'enhanced': Imagen mejorada final
        - 'edges': Bordes detectados (si detect_edges=True)
        - 'steps': Pasos intermedios (si return_all_steps=True)
        
    Example:
        >>> img = cv2.imread('medical.png')
        >>> result = preprocess_medical_image(img, use_clahe=True, detect_edges=True)
        >>> enhanced_img = result['enhanced']
        >>> edges = result['edges']
    """
    results = {
        'enhanced': None,
        'edges': None,
        'steps': {} if return_all_steps else None
    }
    
    current_img = img.copy()
    
    # Paso 1: Normalización adaptativa (opcional)
    if use_normalization:
        current_img = normalize_adaptive(current_img)
        if return_all_steps:
            results['steps']['normalized'] = current_img.copy()
    
    # Paso 2: CLAHE para contraste
    if use_clahe:
        current_img = enhance_bone_contrast(current_img, convert_to_rgb=True)
        if return_all_steps:
            results['steps']['clahe'] = current_img.copy()
    
    results['enhanced'] = current_img
    
    # Paso 3: Detección de bordes (opcional)
    if detect_edges:
        edges = detect_bone_edges(current_img)
        results['edges'] = edges
        if return_all_steps:
            results['steps']['edges'] = edges
    
    return results
