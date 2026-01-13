import numpy as np




def calculate_negative_point(mask, center, distance_factor=0.30):
    """
    Calcula un punto negativo fuera de la máscara a una distancia del 30%.
    El punto se coloca en la dirección opuesta al centro de masa,
    alejándose del borde de la máscara.
    
    Args:
        mask: Máscara binaria de la segmentación anterior
        center: Centro de la máscara [x, y]
        distance_factor: Factor de distancia (0.30 = 30%)
    
    Returns:
        Punto negativo [x, y] o None si no se puede calcular
    """
    if mask is None or np.sum(mask) == 0 or center is None:
        return None
    
    # Encontrar el bounding box de la máscara
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    # Calcular el radio aproximado de la máscara
    width = max_x - min_x
    height = max_y - min_y
    radius = max(width, height) / 2
    
    # Distancia para el punto negativo (30% más allá del borde)
    offset = radius * (1 + distance_factor)
    
    # Probar diferentes direcciones para encontrar un punto válido fuera de la máscara
    h, w = mask.shape
    directions = [
        (0, -1),   # Arriba
        (0, 1),    # Abajo
        (-1, 0),   # Izquierda
        (1, 0),    # Derecha
        (-1, -1),  # Arriba-izquierda
        (1, -1),   # Arriba-derecha
        (-1, 1),   # Abajo-izquierda
        (1, 1),    # Abajo-derecha
    ]
    
    center_x, center_y = center
    
    for dx, dy in directions:
        # Calcular punto candidato
        neg_x = center_x + dx * offset
        neg_y = center_y + dy * offset
        
        # Verificar que esté dentro de los límites de la imagen
        if 0 <= neg_x < w and 0 <= neg_y < h:
            # Verificar que esté FUERA de la máscara
            if not mask[int(neg_y), int(neg_x)]:
                return [neg_x, neg_y]
    
    # Si no se encontró en las direcciones principales, buscar en el borde más cercano
    # y alejarse un 30% adicional
    for dx, dy in directions:
        neg_x = center_x + dx * offset * 1.5  # Intentar más lejos
        neg_y = center_y + dy * offset * 1.5
        
        if 0 <= neg_x < w and 0 <= neg_y < h:
            if not mask[int(neg_y), int(neg_x)]:
                return [neg_x, neg_y]
    
    return None
