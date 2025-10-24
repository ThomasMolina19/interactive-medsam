"""
Test de Tracking Temporal y Corrección de Outliers
Procesa secuencia de imágenes con continuidad temporal
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from src.preprocessing import enhance_bone_contrast
from src.detection import detect_humerus_automatic, correct_outlier_detection, correct_first_frame

# Cargar secuencia de imágenes
image_paths = [
    "dicom_pngs/I01.png",
    "dicom_pngs/I02.png",
    "dicom_pngs/I03.png",
    "dicom_pngs/I04.png",
    "dicom_pngs/I05.png"
]

print("\n" + "="*70)
print("🎯 TEST: Tracking Temporal con Corrección de Outliers")
print("="*70)

# Fase 1: Detección con tracking temporal
print("\n📍 FASE 1: Detección con Tracking Temporal")
print("-" * 70)

detections = []
previous_center = None

for i, img_path in enumerate(image_paths):
    img = np.array(Image.open(img_path).convert("RGB"))
    img_enhanced = enhance_bone_contrast(img)
    
    # Detectar con tracking temporal
    detection = detect_humerus_automatic(
        img, img_enhanced, 
        method='combined',
        previous_center=previous_center,
        search_radius=80.0
    )
    
    detections.append(detection)
    
    if detection['success']:
        center = detection['center']
        conf = detection['confidence']
        
        # Calcular distancia desde anterior
        if previous_center and center:
            dist = np.sqrt((center[0] - previous_center[0])**2 + 
                          (center[1] - previous_center[1])**2)
            print(f"[{i+1}/5] {Path(img_path).name}: ✅ Conf={conf:.3f}, "
                  f"Centro={center}, Dist={dist:.1f}px")
        else:
            print(f"[{i+1}/5] {Path(img_path).name}: ✅ Conf={conf:.3f}, Centro={center}")
        
        previous_center = center
    else:
        print(f"[{i+1}/5] {Path(img_path).name}: ❌ Detección fallida")
        previous_center = None

# Fase 2: Corrección backward del primer frame
print("\n⏪ FASE 2: Corrección Backward del Primer Frame")
print("-" * 70)

if len(detections) >= 2:
    # Cargar primera imagen nuevamente
    img1 = np.array(Image.open(image_paths[0]).convert("RGB"))
    img1_enhanced = enhance_bone_contrast(img1)
    
    # Corregir primer frame usando segundo como referencia
    # Verificar distancia Y tamaño
    detections[0] = correct_first_frame(
        img1, img1_enhanced,
        detections[0], detections[1],
        max_distance=35.0,  # Umbral más estricto
        max_size_diff_ratio=0.3  # 30% diferencia máxima en tamaño
    )

# Fase 3: Corrección de outliers (sandwich)
print("\n🥪 FASE 3: Corrección de Outliers (Sandwich)")
print("-" * 70)

for i in range(len(detections)):
    corrected = correct_outlier_detection(detections, i, max_jump=100.0)
    if corrected != detections[i]:
        detections[i] = corrected

# Visualización
print("\n📊 Generando visualización...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, (img_path, detection) in enumerate(zip(image_paths, detections)):
    img = np.array(Image.open(img_path).convert("RGB"))
    
    # Fila 1: Original con detección
    img_vis = img.copy()
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        
        # Dibujar bounding box
        cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Dibujar círculo
        if detection['detection']['type'] == 'circle':
            params = detection['detection']['params']
            if len(params) == 4:
                x, y, r, _ = params
            else:
                x, y, r = params
            cv2.circle(img_vis, (x, y), r, (255, 0, 0), 2)
            cv2.circle(img_vis, (x, y), 3, (255, 0, 0), -1)
        
        title = f"I{i+1:02d}\nConf: {detection['confidence']:.3f}"
        color = 'green'
    else:
        title = f"I{i+1:02d}\nFALLIDO"
        color = 'red'
    
    axes[0, i].imshow(img_vis)
    axes[0, i].set_title(title, color=color, fontweight='bold')
    axes[0, i].axis('off')
    
    # Fila 2: ROI ampliada
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        roi = img[y_min:y_max, x_min:x_max]
        axes[1, i].imshow(roi)
        axes[1, i].set_title(f"ROI {x_max-x_min}x{y_max-y_min}px")
    else:
        axes[1, i].text(0.5, 0.5, "No ROI", ha='center', va='center')
    axes[1, i].axis('off')

plt.suptitle("Tracking Temporal con Corrección de Outliers", fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = "test_results/temporal_tracking.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"💾 Guardado: {output_path}")

# Resumen
print("\n" + "="*70)
print("📈 RESUMEN")
print("="*70)

successful = sum(1 for d in detections if d['success'])
avg_conf = np.mean([d['confidence'] for d in detections if d['success']])

print(f"✅ Detecciones exitosas: {successful}/{len(detections)}")
print(f"🎯 Confianza promedio: {avg_conf:.3f}")

# Analizar consistencia de posiciones
centers = [d['center'] for d in detections if d['success'] and d['center']]
if len(centers) > 1:
    distances = []
    for i in range(1, len(centers)):
        dist = np.sqrt((centers[i][0] - centers[i-1][0])**2 + 
                      (centers[i][1] - centers[i-1][1])**2)
        distances.append(dist)
    
    print(f"\n📏 Distancias entre frames consecutivos:")
    for i, dist in enumerate(distances):
        print(f"   I{i+1:02d} → I{i+2:02d}: {dist:.1f} px")
    print(f"   Promedio: {np.mean(distances):.1f} px")
    print(f"   Máxima: {np.max(distances):.1f} px")

print("="*70 + "\n")
