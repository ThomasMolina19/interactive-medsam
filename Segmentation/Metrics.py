import numpy as np

def calculate_dice_coefficient(mask1, mask2):
    # 1. Verificar que ambas máscaras tengan el mismo tamaño
    if mask1.shape != mask2.shape:
        return 0.0  # Si son diferentes, no se pueden comparar
    
    # 2. Calcular la INTERSECCIÓN (píxeles que son 1 en AMBAS máscaras)
    #    mask1 * mask2 → solo da 1 donde AMBOS son 1
    intersection = np.sum(mask1 * mask2)
    
    # 3. Calcular la SUMA de píxeles de ambas máscaras
    sum_masks = np.sum(mask1) + np.sum(mask2)
    
    # 4. Caso especial: si ambas están vacías
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0
    
    # 5. Aplicar la fórmula de Dice
    dice = (2.0 * intersection) / sum_masks
    return dice


def calculate_iou(mask1, mask2):


    # 1. Verificar dimensiones iguales IoU=∣A∪B∣/∣A∩B∣

    if mask1.shape != mask2.shape:
        return 0.0
    
    # 2. Calcular INTERSECCIÓN (píxeles en AMBAS máscaras)
    intersection = np.sum(mask1 * mask2)
    
    # 3. Calcular UNIÓN = Total de píxeles únicos en cualquiera de las dos
    #    Fórmula: Área1 + Área2 - Intersección (para no contar 2 veces)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    # 4. Caso especial: ambas máscaras vacías
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    # 5. Calcular IoU
    iou = intersection / union
    return iou
