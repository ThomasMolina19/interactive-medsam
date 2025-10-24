"""
Script de debug para verificar detección y bounding box
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from src.preprocessing import enhance_bone_contrast
from src.detection import detect_humerus_automatic

# Cargar imagen
img = np.array(Image.open("dicom_pngs/I01.png").convert("RGB"))
img_enhanced = enhance_bone_contrast(img)

# Detectar
detection = detect_humerus_automatic(img, img_enhanced, method='combined')

print("\n" + "="*60)
print("DEBUG - Detección del Húmero")
print("="*60)

if detection['success']:
    print(f"\n✅ Detección exitosa")
    print(f"Confianza: {detection['confidence']:.3f}")
    
    # Parámetros del círculo
    if detection['detection']['type'] == 'circle':
        params = detection['detection']['params']
        if len(params) == 4:
            x, y, r, acc = params
            print(f"\n🔵 Círculo detectado:")
            print(f"   Centro: ({x}, {y})")
            print(f"   Radio: {r} px")
            print(f"   Acumulador: {acc:.2f}")
            print(f"   Área: {int(np.pi * r * r)} px²")
        else:
            x, y, r = params
            print(f"\n🔵 Círculo detectado:")
            print(f"   Centro: ({x}, {y})")
            print(f"   Radio: {r} px")
            print(f"   Área: {int(np.pi * r * r)} px²")
    
    # Bounding box generada
    box = detection['box']
    x_min, y_min, x_max, y_max = box
    print(f"\n📦 Bounding Box:")
    print(f"   x_min: {x_min}")
    print(f"   y_min: {y_min}")
    print(f"   x_max: {x_max}")
    print(f"   y_max: {y_max}")
    print(f"   Ancho: {x_max - x_min} px")
    print(f"   Alto: {y_max - y_min} px")
    
    # Verificar si la caja contiene el círculo
    if detection['detection']['type'] == 'circle':
        params = detection['detection']['params']
        if len(params) == 4:
            x, y, r, _ = params
        else:
            x, y, r = params
        contains_circle = (x_min <= x - r and x + r <= x_max and 
                          y_min <= y - r and y + r <= y_max)
        print(f"\n✓ ¿Caja contiene el círculo?: {contains_circle}")
        
        if not contains_circle:
            print("   ⚠️ ERROR: La caja NO contiene completamente el círculo!")
            print(f"   Círculo va de x:[{x-r}, {x+r}], y:[{y-r}, {y+r}]")
            print(f"   Caja va de x:[{x_min}, {x_max}], y:[{y_min}, {y_max}]")
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original con círculo
    img_circle = img.copy()
    if detection['detection']['type'] == 'circle':
        params = detection['detection']['params']
        if len(params) == 4:
            x, y, r, _ = params
        else:
            x, y, r = params
        cv2.circle(img_circle, (x, y), r, (255, 0, 0), 2)
        cv2.circle(img_circle, (x, y), 3, (255, 0, 0), -1)
    
    axes[0].imshow(img_circle)
    axes[0].set_title(f"Círculo Detectado\nCentro: ({x}, {y}), Radio: {r}")
    axes[0].axis('off')
    
    # Con bounding box
    img_box = img.copy()
    cv2.rectangle(img_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    axes[1].imshow(img_box)
    axes[1].set_title(f"Bounding Box\n[{x_min}, {y_min}, {x_max}, {y_max}]")
    axes[1].axis('off')
    
    # Ambos
    img_both = img.copy()
    if detection['detection']['type'] == 'circle':
        params = detection['detection']['params']
        if len(params) == 4:
            x, y, r, _ = params
        else:
            x, y, r = params
        cv2.circle(img_both, (x, y), r, (255, 0, 0), 2)
        cv2.circle(img_both, (x, y), 3, (255, 0, 0), -1)
    cv2.rectangle(img_both, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
    axes[2].imshow(img_both)
    axes[2].set_title("Círculo (rojo) + Caja (verde)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_results/debug_detection.png", dpi=150)
    print(f"\n💾 Imagen guardada: test_results/debug_detection.png")
    
else:
    print("❌ Detección fallida")

print("="*60 + "\n")
