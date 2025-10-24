"""
Detector automático del húmero en imágenes MRI
Implementa Fase 2: Detección por circularidad, intensidad y scoring
Mejorado con técnicas de geometric_project_unal
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks


# ============================================================================
# TAREA 2.1: Detección por Circularidad (Hough Circles)
# ============================================================================

def detect_humerus_by_circularity(
    img_enhanced: np.ndarray,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.15,
    num_candidates: int = 15
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Detecta el húmero usando Transformada de Hough Circular MEJORADA.
    Usa skimage en lugar de OpenCV para mejor robustez.
    
    MEJORAS CLAVE (de geometric_project_unal):
    1. Invierte la imagen (húmero es OSCURO)
    2. Usa skimage.hough_circle (más robusto)
    3. Filtra por área como % de imagen
    4. Detecta MUCHOS candidatos para scoring posterior
    
    Args:
        img_enhanced: Imagen mejorada (de Fase 1)
        min_area_ratio: Área mínima como % de imagen (default: 1%)
        max_area_ratio: Área máxima como % de imagen (default: 15%)
        num_candidates: Número de candidatos a detectar
    
    Returns:
        Lista de círculos detectados como (x, y, radius) o None
        
    Example:
        >>> circles = detect_humerus_by_circularity(img_enhanced)
        >>> if circles:
        >>>     x, y, r = circles[0]  # Mejor candidato
    """
    # Convertir a escala de grises si es necesario
    if len(img_enhanced.shape) == 3:
        gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_enhanced.copy()
    
    # Normalizar a [0, 1]
    gray_norm = gray.astype(np.float32) / 255.0
    
    # CLAVE: INVERTIR porque el húmero es OSCURO
    inverted = 1.0 - gray_norm
    
    # Suavizar con Gaussiano
    smoothed = gaussian_filter(inverted, sigma=2.0)
    
    # Detección de bordes con Canny
    edges = feature.canny(smoothed, sigma=2.0, low_threshold=0.1, high_threshold=0.2)
    
    # Calcular rangos de radio basados en área
    image_area = gray.shape[0] * gray.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio
    
    # Convertir área a radio (área = π * r²)
    min_radius = int(np.sqrt(min_area / np.pi))
    max_radius = int(np.sqrt(max_area / np.pi))
    
    # Asegurar límites razonables
    min_radius = max(min_radius, min(gray.shape) // 20)
    max_radius = min(max_radius, min(gray.shape) // 4)
    
    # Hough transform con skimage (más robusto)
    hough_radii = np.arange(min_radius, max_radius, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    # Extraer MUCHOS candidatos
    accums, cy, cx, radii = hough_circle_peaks(
        hough_res, hough_radii,
        total_num_peaks=num_candidates,
        threshold=0.3 * np.max(hough_res)  # Umbral bajo para más candidatos
    )
    
    if len(accums) == 0:
        return None
    
    # Convertir a formato (x, y, r) con acumulador
    circles_list = [(int(cx[i]), int(cy[i]), int(radii[i]), float(accums[i])) 
                    for i in range(len(accums))]
    
    return circles_list


def detect_humerus_by_ellipse(
    img_enhanced: np.ndarray,
    edges: Optional[np.ndarray] = None
) -> List[Tuple]:
    """
    Detecta el húmero usando detección de elipses.
    El húmero puede tener forma elíptica/ovalada (forma de ojo) en algunos cortes.
    
    Args:
        img_enhanced: Imagen mejorada
        edges: Bordes detectados (opcional)
    
    Returns:
        Lista de elipses detectadas como ((x, y), (width, height), angle)
    """
    if len(img_enhanced.shape) == 3:
        gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_enhanced.copy()
    
    # Detectar bordes si no se proporcionan
    if edges is None:
        edges = cv2.Canny(gray, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipses = []
    
    for contour in contours:
        # Necesitamos al menos 5 puntos para ajustar una elipse
        if len(contour) >= 5:
            area = cv2.contourArea(contour)
            
            # Filtrar por área razonable
            if 500 < area < 15000:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (width, height), angle = ellipse
                    
                    # Filtrar elipses muy alargadas o muy pequeñas
                    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                    if aspect_ratio < 2.5 and min(width, height) > 40:  # No muy alargada
                        ellipses.append(ellipse)
                except:
                    pass
    
    return ellipses


# ============================================================================
# TAREA 2.2: Detección por Análisis de Intensidad
# ============================================================================

def detect_humerus_by_intensity(
    img_enhanced: np.ndarray,
    use_otsu: bool = True,
    morph_kernel_size: int = 5,
    min_area: int = 500,
    max_area: int = 10000
) -> List[np.ndarray]:
    """
    Detecta el húmero por análisis de intensidad.
    El húmero tiene un patrón característico: anillo oscuro con centro más claro.
    
    Args:
        img_enhanced: Imagen mejorada (de Fase 1)
        use_otsu: Usar umbralización de Otsu automática
        morph_kernel_size: Tamaño del kernel para operaciones morfológicas
        min_area: Área mínima de contornos válidos
        max_area: Área máxima de contornos válidos
    
    Returns:
        Lista de contornos candidatos
        
    Example:
        >>> contours = detect_humerus_by_intensity(img_enhanced)
        >>> for contour in contours:
        >>>     score = score_humerus_candidate(contour, img.shape)
    """
    # Convertir a escala de grises
    if len(img_enhanced.shape) == 3:
        gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_enhanced.copy()
    
    # Umbralización
    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Umbral adaptativo como alternativa
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    # Operaciones morfológicas para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por área
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
    
    return valid_contours


def analyze_ring_pattern(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int
) -> float:
    """
    Analiza si hay un patrón de "anillo oscuro" característico del húmero.
    El húmero tiene una franja negra distintiva que lo rodea.
    
    Args:
        img: Imagen en escala de grises
        center: Centro del círculo (x, y)
        radius: Radio del círculo
    
    Returns:
        Score de 0 a 1 indicando qué tan bien coincide con el patrón
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    x, y = center
    
    # Verificar que el círculo esté dentro de la imagen
    if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
        return 0.0
    
    # Crear múltiples anillos concéntricos para análisis más preciso
    mask_outer = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_outer, (x, y), radius, 255, -1)
    
    mask_middle = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_middle, (x, y), int(radius * 0.75), 255, -1)
    
    mask_inner = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_inner, (x, y), int(radius * 0.5), 255, -1)
    
    # Máscara del anillo exterior (donde está la franja negra)
    mask_dark_ring = cv2.subtract(mask_outer, mask_middle)
    
    # Máscara del anillo medio
    mask_middle_ring = cv2.subtract(mask_middle, mask_inner)
    
    # Calcular intensidades
    intensity_dark_ring = cv2.mean(gray, mask=mask_dark_ring)[0]
    intensity_middle = cv2.mean(gray, mask=mask_middle_ring)[0]
    intensity_center = cv2.mean(gray, mask=mask_inner)[0]
    
    # Intensidad fuera del círculo
    mask_outside = np.ones_like(gray, dtype=np.uint8) * 255
    mask_outside = cv2.subtract(mask_outside, mask_outer)
    intensity_outside = cv2.mean(gray, mask=mask_outside)[0]
    
    score = 0.0
    
    # PATRÓN CARACTERÍSTICO DEL HÚMERO:
    # 1. Anillo oscuro (franja negra) - debe ser significativamente más oscuro
    if intensity_dark_ring < intensity_center * 0.85:  # Anillo 15%+ más oscuro que centro
        score += 0.35
    
    if intensity_dark_ring < intensity_middle * 0.90:  # Anillo más oscuro que zona media
        score += 0.25
    
    # 2. El anillo debe ser más oscuro que el exterior
    if intensity_dark_ring < intensity_outside * 0.95:
        score += 0.20
    
    # 3. Contraste fuerte - debe haber diferencia clara
    contrast = abs(intensity_center - intensity_dark_ring)
    if contrast > 15:  # Diferencia mínima de 15 en intensidad
        score += 0.20 * min(contrast / 50, 1.0)  # Normalizar
    return min(score, 1.0)


# ============================================================================
# TAREA 2.4b: Tracking Temporal y Corrección de Outliers
# ============================================================================

def filter_candidates_by_proximity(
    candidates: List[Dict],
    previous_center: Tuple[int, int],
    max_distance: float,
    img_shape: Tuple[int, int]
) -> List[Dict]:
    """
    Filtra candidatos que estén dentro de un radio de búsqueda desde la detección previa.
    Esto asegura continuidad temporal en secuencias de imágenes.
    
    Args:
        candidates: Lista de candidatos detectados
        previous_center: Centro (x, y) de la detección previa
        max_distance: Distancia máxima permitida (en píxeles)
        img_shape: Forma de la imagen
    
    Returns:
        Lista filtrada de candidatos cercanos
    """
    if not previous_center:
        return candidates
    
    prev_x, prev_y = previous_center
    filtered = []
    
    for candidate in candidates:
        # Extraer centro del candidato
        if candidate['type'] == 'circle':
            params = candidate['params']
            if len(params) == 4:
                x, y, r, _ = params
            else:
                x, y, r = params
        elif candidate['type'] == 'ellipse':
            (x, y), _, _ = candidate['params']
        elif candidate['type'] == 'contour':
            contour = candidate['params']
            M = cv2.moments(contour)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
            else:
                continue
        else:
            continue
        
        # Calcular distancia al centro previo
        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        
        if distance <= max_distance:
            filtered.append(candidate)
    
    return filtered if filtered else candidates  # Si no hay ninguno cerca, devolver todos


def correct_first_frame(
    img: np.ndarray,
    img_enhanced: np.ndarray,
    first_detection: Dict,
    second_detection: Dict,
    max_distance: float = 60.0,
    max_size_diff_ratio: float = 0.5
) -> Dict:
    """
    Corrección "backward tracking": si la primera detección está muy lejos
    de la segunda O tiene tamaño muy diferente, redetectar usando la segunda como referencia.
    
    Args:
        img: Imagen original del primer frame
        img_enhanced: Imagen mejorada del primer frame
        first_detection: Detección del primer frame
        second_detection: Detección del segundo frame
        max_distance: Distancia máxima permitida antes de redetectar
        max_size_diff_ratio: Diferencia máxima de tamaño (0.5 = 50%)
    
    Returns:
        Detección corregida del primer frame
    """
    if not first_detection['success'] or not second_detection['success']:
        return first_detection
    
    first_center = first_detection.get('center')
    second_center = second_detection.get('center')
    
    if not first_center or not second_center:
        return first_detection
    
    # Extraer radios
    def get_radius(det):
        if det['detection']['type'] == 'circle':
            params = det['detection']['params']
            return params[2]  # radio es el tercer parámetro
        return None
    
    first_radius = get_radius(first_detection)
    second_radius = get_radius(second_detection)
    
    # Calcular distancia
    distance = np.sqrt((first_center[0] - second_center[0])**2 + 
                      (first_center[1] - second_center[1])**2)
    
    # Calcular diferencia de tamaño
    size_diff = 0
    if first_radius and second_radius:
        size_diff = abs(first_radius - second_radius) / max(first_radius, second_radius)
    
    needs_correction = False
    reason = []
    
    # Verificar distancia
    if distance > max_distance:
        needs_correction = True
        reason.append(f"distancia {distance:.1f}px > {max_distance}px")
    
    # Verificar tamaño
    if size_diff > max_size_diff_ratio:
        needs_correction = True
        reason.append(f"diferencia tamaño {size_diff*100:.1f}% > {max_size_diff_ratio*100:.1f}%")
    
    # Si necesita corrección, redetectar
    if needs_correction:
        print(f"⚠️  Frame 1 inconsistente con Frame 2: {', '.join(reason)}")
        print(f"   Frame 1: centro={first_center}, radio={first_radius}px")
        print(f"   Frame 2: centro={second_center}, radio={second_radius}px")
        print(f"   Redetectando Frame 1 con referencia de Frame 2...")
        
        corrected = detect_humerus_automatic(
            img, img_enhanced,
            method='combined',
            previous_center=second_center,
            search_radius=80.0
        )
        
        if corrected['success']:
            new_center = corrected.get('center')
            new_radius = get_radius(corrected)
            new_distance = np.sqrt((new_center[0] - second_center[0])**2 + 
                                  (new_center[1] - second_center[1])**2)
            new_size_diff = abs(new_radius - second_radius) / max(new_radius, second_radius) if new_radius else 0
            
            print(f"   ✅ Corregido:")
            print(f"      Centro: {first_center} → {new_center} (dist: {new_distance:.1f}px)")
            print(f"      Radio: {first_radius}px → {new_radius}px (diff: {new_size_diff*100:.1f}%)")
            return corrected
    
    return first_detection


def correct_outlier_detection(
    detections: List[Dict],
    index: int,
    max_jump: float = 100.0
) -> Optional[Dict]:
    """
    Corrección "sandwich": si detección N está muy lejos de N-1 y N+1,
    reposicionarla entre ellas.
    
    Args:
        detections: Lista de detecciones (puede tener None)
        index: Índice de la detección a verificar
        max_jump: Distancia máxima permitida para considerar outlier
    
    Returns:
        Detección corregida o None
    """
    if index <= 0 or index >= len(detections) - 1:
        return detections[index]  # No se puede corregir primero o último
    
    prev_det = detections[index - 1]
    curr_det = detections[index]
    next_det = detections[index + 1]
    
    # Si alguno es None, no podemos corregir
    if not prev_det or not curr_det or not next_det:
        return curr_det
    
    # Extraer centros
    def get_center(det):
        if det['detection']['type'] == 'circle':
            params = det['detection']['params']
            if len(params) == 4:
                return params[0], params[1]
            return params[0], params[1]
        return None
    
    prev_center = get_center(prev_det)
    curr_center = get_center(curr_det)
    next_center = get_center(next_det)
    
    if not all([prev_center, curr_center, next_center]):
        return curr_det
    
    # Calcular distancias
    dist_prev_curr = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
    dist_curr_next = np.sqrt((next_center[0] - curr_center[0])**2 + 
                             (next_center[1] - curr_center[1])**2)
    dist_prev_next = np.sqrt((next_center[0] - prev_center[0])**2 + 
                             (next_center[1] - prev_center[1])**2)
    
    # Si curr está muy lejos de ambos vecinos, pero vecinos están cerca entre sí
    if dist_prev_curr > max_jump and dist_curr_next > max_jump and dist_prev_next < max_jump:
        # Outlier detectado! Reposicionar en el punto medio entre prev y next
        new_x = int((prev_center[0] + next_center[0]) / 2)
        new_y = int((prev_center[1] + next_center[1]) / 2)
        
        # Mantener el radio original
        if curr_det['detection']['type'] == 'circle':
            params = curr_det['detection']['params']
            if len(params) == 4:
                r, acc = params[2], params[3]
                curr_det['detection']['params'] = (new_x, new_y, r, acc)
            else:
                r = params[2]
                curr_det['detection']['params'] = (new_x, new_y, r)
        
        # Regenerar bounding box
        curr_det['box'] = generate_bounding_box(curr_det['detection'], 
                                                 curr_det['box'].shape if hasattr(curr_det['box'], 'shape') else (512, 512))
        
        print(f"⚠️  Outlier corregido en frame {index}: movido de ({curr_center[0]}, {curr_center[1]}) → ({new_x}, {new_y})")
    
    return curr_det


# ============================================================================
# TAREA 2.5: Pipeline Automático Completo
# ============================================================================

def score_humerus_candidate(
    candidate: Dict,
    img_shape: Tuple[int, int],
    img: Optional[np.ndarray] = None
) -> float:
    """
    Calcula score de probabilidad de que un candidato sea el húmero.
    Combina múltiples características anatómicas y visuales.
    
    Args:
        candidate: Diccionario con 'type' ('circle', 'ellipse', o 'contour') y 'params'
        img_shape: Forma de la imagen (height, width)
        img: Imagen original (opcional, para análisis de intensidad)
    
    Returns:
        Score de 0 a 1 (mayor = más probable que sea húmero)
        
    Example:
        >>> candidate = {'type': 'circle', 'params': (x, y, r)}
        >>> score = score_humerus_candidate(candidate, img.shape, img)
    """
    score = 0.0
    h, w = img_shape[:2]
    
    if candidate['type'] == 'circle':
        # Nuevo formato incluye acumulador: (x, y, r, accumulator)
        if len(candidate['params']) == 4:
            x, y, r, accumulator = candidate['params']
        else:
            x, y, r = candidate['params']
            accumulator = 0
        area = np.pi * r * r
        circularity = 1.0  # Los círculos son perfectamente circulares
        
    elif candidate['type'] == 'ellipse':
        (x, y), (width, height), angle = candidate['params']
        x, y = int(x), int(y)
        area = np.pi * (width / 2) * (height / 2)
        # Radio promedio para análisis de anillo
        r = int((width + height) / 4)
        # Calcular circularidad para elipse
        circularity = min(width, height) / max(width, height)
        
    elif candidate['type'] == 'contour':
        contour = candidate['params']
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calcular circularidad: 4π*area / perimeter²
        # Círculo perfecto = 1.0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0.0
        
        # Calcular centroide
        M = cv2.moments(contour)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        else:
            return 0.0
        
        # Calcular radio aproximado
        r = int(np.sqrt(area / np.pi))
    else:
        return 0.0
    
    # ========== Factor 0: Acumulador Hough (si disponible) ==========
    if candidate['type'] == 'circle' and 'accumulator' in locals() and accumulator > 0:
        # Normalizar acumulador (típicamente 0-200)
        acc_normalized = min(accumulator / 200.0, 1.0)
        score += 0.10 * acc_normalized
    
    # ========== Factor 1: Intensidad Promedio (peso: 30%) - NUEVO Y CRÍTICO ==========
    # El húmero es OSCURO - verificar intensidad promedio dentro del círculo
    if img is not None:
        # Crear máscara circular
        mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        circle_mask = (xx - x)**2 + (yy - y)**2 <= r**2
        mask[circle_mask] = True
        
        # Calcular intensidad promedio (normalizada 0-1)
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img
        
        mean_intensity = np.mean(gray_img[mask]) / 255.0
        
        # El húmero es OSCURO - menor intensidad = mejor score
        darkness_score = 1.0 - mean_intensity
        score += 0.30 * darkness_score
        
        # Bonus si es MUY oscuro
        if mean_intensity < 0.3:
            score += 0.10
    
    # ========== Factor 2: Patrón de anillo oscuro (peso: 30%) ==========
    # El húmero tiene una franja negra muy distintiva
    if img is not None:
        ring_score = analyze_ring_pattern(img, (x, y), r)
        score += 0.30 * ring_score
        
        # Si NO tiene el patrón de anillo oscuro, penalizar
        if ring_score < 0.4:
            score -= 0.20
    else:
        # Sin imagen para verificar, penalizar
        score -= 0.20
    
    # ========== Factor 2: Circularidad (peso: 20%) ==========
    # El húmero debe ser aproximadamente circular
    if 0.7 < circularity <= 1.0:
        score += 0.20 * circularity
    
    # ========== Factor 3: Tamaño apropiado (peso: 25%) - CRÍTICO ==========
    # Área esperada del húmero en píxeles
    # Radio ~13px = área 530px² (mínimo)
    # Radio ~50px = área 7850px² (máximo)
    expected_area_min = 1500  # Aumentado: radio mínimo ~22px
    expected_area_max = 8000
    
    if expected_area_min < area < expected_area_max:
        # Score proporcional a qué tan cerca está del rango ideal
        ideal_area = 3000  # Área ideal: radio ~31px
        area_score = 1.0 - abs(area - ideal_area) / ideal_area
        score += 0.25 * max(0, area_score)
        
        # Bonus adicional para tamaños medianos-grandes
        if 2000 < area < 5000:
            score += 0.10  # Bonus por estar en rango óptimo
    else:
        # Penalizar MUY fuertemente si está fuera del rango
        if area < expected_area_min:
            # Muy pequeño - probablemente no es el húmero
            penalty = 0.50 * (1.0 - area / expected_area_min)
            score -= penalty
        else:
            # Muy grande
            score -= 0.20
    
    # ========== Factor 4: Centralidad (peso: 15%) - MEJORADO ==========
    # El húmero típicamente está CENTRADO en la imagen
    image_center = np.array([w / 2, h / 2])
    candidate_center = np.array([x, y])
    
    # Calcular distancia al centro
    distance = np.linalg.norm(candidate_center - image_center)
    max_distance = np.sqrt(w**2 + h**2) / 2
    
    # Centralidad: 1.0 = perfecto centro, 0.0 = esquina
    centrality = 1.0 - (distance / max_distance)
    score += 0.15 * centrality
    
    # Bonus adicional si está MUY centrado
    if centrality > 0.7:
        score += 0.05
    
    # Penalizar fuertemente si está en los bordes
    edge_margin = min(w, h) * 0.10
    if x < edge_margin or x > w - edge_margin or y < edge_margin or y > h - edge_margin:
        score -= 0.25
    
    return min(score, 1.0)


def merge_nearby_candidates(
    candidates: List[Dict],
    max_distance: float = 50.0,
    min_score: float = 0.7
) -> Optional[Dict]:
    """
    Fusiona candidatos cercanos con scores altos en un candidato promedio.
    Útil cuando Hough detecta múltiples fragmentos del mismo húmero.
    
    Args:
        candidates: Lista de candidatos con scores
        max_distance: Distancia máxima para considerar candidatos como cluster
        min_score: Score mínimo para incluir en el cluster
    
    Returns:
        Candidato fusionado (promedio del cluster) o None
    """
    # Filtrar candidatos con score alto
    high_score_candidates = [c for c in candidates if c.get('score', 0) >= min_score]
    
    if len(high_score_candidates) < 2:
        return None  # No hay suficientes para fusionar
    
    # Extraer centros y radios
    centers = []
    radii = []
    scores = []
    
    for cand in high_score_candidates:
        if cand['type'] == 'circle':
            params = cand['params']
            if len(params) == 4:
                x, y, r, _ = params
            else:
                x, y, r = params
            centers.append((x, y))
            radii.append(r)
            scores.append(cand['score'])
    
    if len(centers) < 2:
        return None
    
    # Verificar si están cercanos (clustering simple)
    centers_array = np.array(centers)
    center_mean = centers_array.mean(axis=0)
    
    # Calcular distancias al centro promedio
    distances = [np.linalg.norm(c - center_mean) for c in centers_array]
    
    # Contar cuántos están cerca del centro promedio
    nearby_count = sum(1 for d in distances if d <= max_distance)
    
    # Si al menos 3 candidatos están cerca, fusionar
    if nearby_count >= 3:
        # Calcular promedio ponderado por score
        total_score = sum(scores)
        weighted_x = sum(c[0] * s for c, s in zip(centers, scores)) / total_score
        weighted_y = sum(c[1] * s for c, s in zip(centers, scores)) / total_score
        weighted_r = sum(r * s for r, s in zip(radii, scores)) / total_score
        
        # Crear candidato fusionado
        merged = {
            'type': 'circle',
            'params': (int(weighted_x), int(weighted_y), int(weighted_r)),
            'method': 'merged',
            'score': max(scores),  # Usar el score más alto
            'merged_count': nearby_count
        }
        
        print(f"🔗 Fusionados {nearby_count} candidatos cercanos:")
        print(f"   Centro promedio: ({int(weighted_x)}, {int(weighted_y)}), Radio: {int(weighted_r)}px")
        
        return merged
    
    return None


def select_best_candidate(
    candidates: List[Dict],
    img_shape: Tuple[int, int],
    img: Optional[np.ndarray] = None
) -> Optional[Dict]:
    """
    Selecciona el mejor candidato de una lista basado en scoring.
    Si hay múltiples candidatos cercanos con scores altos, los fusiona.
    
    Args:
        candidates: Lista de candidatos
        img_shape: Forma de la imagen
        img: Imagen original (para scoring)
    
    Returns:
        Mejor candidato o None
        
    Example:
        >>> best = select_best_candidate(candidates, img.shape, img)
    """
    if not candidates:
        return None
    
    # Calcular scores para todos los candidatos
    scored_candidates = []
    for candidate in candidates:
        score = score_humerus_candidate(candidate, img_shape, img)
        candidate['score'] = score
        scored_candidates.append(candidate)
    
    # Ordenar por score descendente
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Intentar fusionar candidatos cercanos con scores altos
    merged = merge_nearby_candidates(scored_candidates, max_distance=50.0, min_score=0.7)
    
    if merged:
        return merged
    
    # Si no se pudo fusionar, retornar el mejor
    return scored_candidates[0]


# ============================================================================
# TAREA 2.4: Generación de Bounding Box Automática
# ============================================================================

def refine_circle_center(
    img: np.ndarray,
    initial_center: Tuple[int, int],
    initial_radius: int,
    search_margin: float = 0.3
) -> Tuple[int, int, int]:
    """
    Refina el centro del círculo detectado analizando el ROI local.
    Busca el anillo oscuro característico del húmero para recentrar.
    
    Args:
        img: Imagen original
        initial_center: Centro inicial (x, y)
        initial_radius: Radio inicial
        search_margin: Margen de búsqueda como % del radio
    
    Returns:
        Centro refinado (x, y) y radio refinado
    """
    x_init, y_init = initial_center
    h, w = img.shape[:2] if len(img.shape) == 2 else img.shape[:2]
    
    # Convertir a escala de grises
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Definir ROI de búsqueda
    search_radius = int(initial_radius * (1 + search_margin))
    x_min = max(0, x_init - search_radius)
    y_min = max(0, y_init - search_radius)
    x_max = min(w, x_init + search_radius)
    y_max = min(h, y_init + search_radius)
    
    roi = gray[y_min:y_max, x_min:x_max]
    
    # Detectar bordes en el ROI
    edges = cv2.Canny(roi, 30, 100)
    
    # Buscar círculos en el ROI
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=max(10, int(initial_radius * 0.7)),
        maxRadius=int(initial_radius * 1.3)
    )
    
    if circles is not None and len(circles[0]) > 0:
        # Tomar el círculo más cercano al centro inicial
        circles = np.uint16(np.around(circles[0]))
        
        best_circle = None
        min_dist = float('inf')
        
        for (cx_roi, cy_roi, r_roi) in circles:
            # Convertir a coordenadas globales
            cx_global = cx_roi + x_min
            cy_global = cy_roi + y_min
            
            # Distancia al centro inicial
            dist = np.sqrt((cx_global - x_init)**2 + (cy_global - y_init)**2)
            
            if dist < min_dist:
                min_dist = dist
                best_circle = (cx_global, cy_global, r_roi)
        
        if best_circle and min_dist < initial_radius * 0.5:
            print(f"   🎯 Centro refinado: ({x_init}, {y_init}) → ({best_circle[0]}, {best_circle[1]}), "
                  f"Radio: {initial_radius} → {best_circle[2]}")
            return best_circle
    
    # Si no se encontró mejor círculo, mantener el original
    return (x_init, y_init, initial_radius)


def generate_bounding_box(
    detection_result: Dict,
    img_shape: Tuple[int, int],
    margin: float = 0.40,
    img: Optional[np.ndarray] = None,
    refine_center: bool = True
) -> np.ndarray:
    """
    Genera bounding box alrededor del húmero detectado.
    Opcionalmente refina el centro para mejor centrado.
    
    Args:
        detection_result: Resultado de detección con 'type' y 'params'
        img_shape: Forma de la imagen (height, width)
        margin: Margen adicional alrededor del húmero (porcentaje)
                AUMENTADO a 40% para dar más contexto a SAM
        img: Imagen original (para refinamiento)
        refine_center: Si True, refina el centro del círculo
    
    Returns:
        Array [x_min, y_min, x_max, y_max]
        
    Example:
        >>> box = generate_bounding_box(best_candidate, img.shape, img=img)
        >>> masks = predictor.predict(box=box)
    """
    h, w = img_shape[:2]
    
    if detection_result['type'] == 'circle':
        # Manejar formato con o sin acumulador
        params = detection_result['params']
        if len(params) == 4:
            x, y, r, _ = params  # Ignorar acumulador
        else:
            x, y, r = params
        
        # REFINAMIENTO: Ajustar centro si se proporciona imagen
        if refine_center and img is not None:
            x, y, r = refine_circle_center(img, (x, y), r)
        
        # Expandir con margen
        r_expanded = int(r * (1 + margin))
        x_min = max(0, x - r_expanded)
        y_min = max(0, y - r_expanded)
        x_max = min(w, x + r_expanded)
        y_max = min(h, y + r_expanded)
    
    elif detection_result['type'] == 'ellipse':
        (x, y), (width, height), angle = detection_result['params']
        x, y = int(x), int(y)
        # Usar el eje mayor para calcular el margen
        max_axis = max(width, height) / 2
        margin_px = int(max_axis * (1 + margin))
        x_min = max(0, x - margin_px)
        y_min = max(0, y - margin_px)
        x_max = min(w, x + margin_px)
        y_max = min(h, y + margin_px)
        
    elif detection_result['type'] == 'contour':
        contour = detection_result['params']
        x, y, cw, ch = cv2.boundingRect(contour)
        # Expandir con margen
        margin_px = int(max(cw, ch) * margin)
        x_min = max(0, x - margin_px)
        y_min = max(0, y - margin_px)
        x_max = min(w, x + cw + margin_px)
        y_max = min(h, y + ch + margin_px)
    else:
        # Fallback: región central
        margin_px = min(w, h) // 4
        x_min = margin_px
        y_min = margin_px
        x_max = w - margin_px
        y_max = h - margin_px
    
    return np.array([x_min, y_min, x_max, y_max])


# ============================================================================
# PIPELINE COMPLETO - FASE 2
# ============================================================================

def detect_humerus_automatic(
    img: np.ndarray,
    img_enhanced: np.ndarray,
    method: str = 'combined',
    return_all_candidates: bool = False,
    previous_center: Optional[Tuple[int, int]] = None,
    search_radius: float = 80.0
) -> Dict:
    """
    Pipeline completo de detección automática del húmero (Fase 2).
    CON TRACKING TEMPORAL para secuencias de imágenes.
    
    Args:
        img: Imagen original
        img_enhanced: Imagen mejorada (de Fase 1)
        method: 'circular', 'intensity', o 'combined'
        return_all_candidates: Si True, retorna todos los candidatos
        previous_center: Centro (x, y) de detección previa para tracking temporal
        search_radius: Radio de búsqueda desde previous_center (píxeles)
    
    Returns:
        Diccionario con:
        - 'success': bool
        - 'box': bounding box [x_min, y_min, x_max, y_max]
        - 'detection': mejor candidato
        - 'confidence': score de confianza
        - 'method': método usado
        - 'all_candidates': lista de todos (si return_all_candidates=True)
        - 'center': centro (x, y) de la detección
        
    Example:
        >>> result = detect_humerus_automatic(img, img_enhanced)
        >>> if result['success']:
        >>>     box = result['box']
        >>>     confidence = result['confidence']
        >>>     # Para siguiente frame:
        >>>     next_result = detect_humerus_automatic(img2, img2_enh, 
        >>>                                            previous_center=result['center'])
    """
    result = {
        'success': False,
        'box': None,
        'detection': None,
        'confidence': 0.0,
        'method': method,
        'all_candidates': [] if return_all_candidates else None
    }
    
    candidates = []
    
    # Método 1: Detección por circularidad (círculos)
    if method in ['circular', 'combined']:
        circles = detect_humerus_by_circularity(img_enhanced)
        if circles:
            for circle in circles:
                candidates.append({
                    'type': 'circle',
                    'params': circle,
                    'method': 'circular'
                })
    
    # Método 1b: Detección de elipses (forma de ojo)
    if method in ['ellipse', 'combined']:
        ellipses = detect_humerus_by_ellipse(img_enhanced)
        if ellipses:
            for ellipse in ellipses:
                candidates.append({
                    'type': 'ellipse',
                    'params': ellipse,
                    'method': 'ellipse'
                })
    
    # Método 2: Detección por intensidad
    if method in ['intensity', 'combined']:
        contours = detect_humerus_by_intensity(img_enhanced)
        for contour in contours:
            candidates.append({
                'type': 'contour',
                'params': contour,
                'method': 'intensity'
            })
    
    # TRACKING TEMPORAL: Filtrar candidatos por proximidad si hay detección previa
    if previous_center is not None and len(candidates) > 0:
        candidates_filtered = filter_candidates_by_proximity(
            candidates, previous_center, search_radius, img.shape
        )
        if len(candidates_filtered) < len(candidates):
            print(f"🎯 Tracking: {len(candidates)} candidatos → {len(candidates_filtered)} cercanos a {previous_center}")
        candidates = candidates_filtered
    
    if return_all_candidates:
        result['all_candidates'] = candidates
    
    # Seleccionar mejor candidato
    best_candidate = select_best_candidate(candidates, img.shape, img_enhanced)
    
    if best_candidate is None:
        return result
    
    # Generar bounding box con refinamiento de centro
    box = generate_bounding_box(best_candidate, img.shape, img=img, refine_center=True)
    
    # Extraer centro para tracking en siguiente frame
    if best_candidate['type'] == 'circle':
        params = best_candidate['params']
        if len(params) == 4:
            center = (params[0], params[1])
        else:
            center = (params[0], params[1])
    elif best_candidate['type'] == 'ellipse':
        (x, y), _, _ = best_candidate['params']
        center = (int(x), int(y))
    else:
        center = None
    
    result['success'] = True
    result['box'] = box
    result['detection'] = best_candidate
    result['confidence'] = best_candidate.get('score', 0.0)
    result['center'] = center
    
    return result
