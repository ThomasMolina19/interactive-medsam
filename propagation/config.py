"""Configuración y constantes para el módulo de propagación."""

# Umbrales de similitud
SIMILARITY_THRESHOLD = 0.20  # 20% - Advertencias leves
WARNING_THRESHOLD = 0.30     # 30% - Advertencias severas

# Extensiones de imagen soportadas
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']

# Offsets para reintentos de segmentación cuando falla el punto original
RETRY_OFFSETS = [
    (0, 0), (-10, 0), (10, 0), (0, -10), (0, 10),
    (-20, 0), (20, 0), (0, -20), (0, 20)
]

# Parámetros de mejora de imagen
IMAGE_ENHANCE_ALPHA = 1.2
IMAGE_ENHANCE_BETA = 10

# Parámetros de refinamiento de máscara
MIN_MASK_SIZE = 500
DISK_RADIUS = 2

# Factor de distancia para puntos negativos
NEGATIVE_POINT_DISTANCE_FACTOR = 0.30
