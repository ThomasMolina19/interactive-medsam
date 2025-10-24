"""
Pipeline Automático Completo: Detección + SAM
Fase 1 (CLAHE) → Fase 2 (Detección) → Fase 3 (SAM)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import torch

from src.preprocessing import enhance_bone_contrast
from src.detection import detect_humerus_automatic, correct_first_frame
from segment_anything import sam_model_registry, SamPredictor

print("\n" + "="*70)
print("🤖 PIPELINE AUTOMÁTICO COMPLETO: Detección + SAM")
print("="*70)

# Configuración
image_paths = [
    "dicom_pngs/I01.png",
    "dicom_pngs/I02.png",
    "dicom_pngs/I03.png",
    "dicom_pngs/I04.png",
    "dicom_pngs/I05.png"
]

# Cargar MedSAM
print("\n📦 Cargando modelo MedSAM...")
ckpt = "checkpoints/medsam_vit_b.pth"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Verificar si existe el checkpoint
if not os.path.exists(ckpt):
    print(f"\n❌ ERROR: Checkpoint no encontrado en: {ckpt}")
    print("\n📥 Por favor descarga el modelo MedSAM:")
    print("   1. Visita: https://github.com/bowang-lab/MedSAM")
    print("   2. Descarga medsam_vit_b.pth (~2.4 GB)")
    print("   3. Colócalo en: checkpoints/medsam_vit_b.pth")
    print("\n   O usa gdown:")
    print("   mkdir -p checkpoints")
    print("   pip install gdown")
    print("   gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam_vit_b.pth")
    sys.exit(1)

sam = sam_model_registry["vit_b"]()
checkpoint = torch.load(ckpt, map_location=device)
sam.load_state_dict(checkpoint)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)

print(f"   ✅ MedSAM cargado en: {device}")

# Pipeline completo
print("\n" + "="*70)
print("🔄 PROCESANDO SECUENCIA COMPLETA")
print("="*70)

detections = []
masks_results = []
previous_center = None

for i, img_path in enumerate(image_paths):
    print(f"\n[{i+1}/5] {Path(img_path).name}")
    print("-" * 70)
    
    # Cargar imagen
    img = np.array(Image.open(img_path).convert("RGB"))
    
    # FASE 1: Mejora de contraste (CLAHE)
    print("   📊 Fase 1: CLAHE...")
    img_enhanced = enhance_bone_contrast(img)
    
    # FASE 2: Detección automática con tracking
    print("   🎯 Fase 2: Detección automática...")
    detection = detect_humerus_automatic(
        img, img_enhanced,
        method='combined',
        previous_center=previous_center,
        search_radius=80.0
    )
    
    detections.append(detection)
    
    if not detection['success']:
        print("   ❌ Detección fallida")
        masks_results.append(None)
        previous_center = None
        continue
    
    # Actualizar centro para siguiente frame
    previous_center = detection['center']
    
    box = detection['box']
    conf = detection['confidence']
    center = detection['center']
    
    print(f"   ✅ Detectado: centro={center}, conf={conf:.3f}")
    print(f"   📦 BBox: {box}")
    
    # FASE 3: Segmentación con SAM
    print("   🔮 Fase 3: SAM...")
    predictor.set_image(img)
    
    # Preparar point prompts (centro del círculo detectado)
    center_x, center_y = center
    point_coords = np.array([[center_x, center_y]])
    point_labels = np.array([1])  # 1 = foreground point
    
    # Predecir con bounding box + point prompt
    masks, scores, logits = predictor.predict(
        box=box,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # SELECCIÓN MEJORADA: Combinar score + área + circularidad
    print(f"   📊 Evaluando {len(masks)} máscaras...")
    
    # Calcular área esperada del círculo detectado
    if detection['detection']['type'] == 'circle':
        params = detection['detection']['params']
        if len(params) == 4:
            _, _, r, _ = params
        else:
            _, _, r = params
        expected_area = np.pi * r * r
    else:
        expected_area = None
    
    best_mask_idx = 0
    best_combined_score = -1
    
    for idx, (mask, sam_score) in enumerate(zip(masks, scores)):
        # Factor 1: Score de SAM (50%)
        score_component = sam_score * 0.5
        
        # Factor 2: Similitud de área (30%)
        mask_area = np.sum(mask)
        if expected_area:
            area_ratio = min(mask_area, expected_area) / max(mask_area, expected_area)
            area_component = area_ratio * 0.3
        else:
            area_component = 0.15
        
        # Factor 3: Circularidad (20%)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * mask_area / (perimeter * perimeter)
                circularity = min(circularity, 1.0)
            else:
                circularity = 0
            circularity_component = circularity * 0.2
        else:
            circularity_component = 0
        
        combined_score = score_component + area_component + circularity_component
        
        print(f"      Máscara {idx+1}: SAM={sam_score:.3f}, Área={mask_area:.0f}px², "
              f"Circ={circularity if contours else 0:.3f}, Combined={combined_score:.3f}")
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_mask_idx = idx
    
    best_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]
    
    print(f"   ✅ Mejor máscara: #{best_mask_idx+1} (SAM score={best_score:.3f}, Combined={best_combined_score:.3f})")
    
    masks_results.append({
        'masks': masks,
        'scores': scores,
        'best_mask': best_mask,
        'best_score': best_score
    })

# Corrección backward del primer frame
print("\n" + "="*70)
print("⏪ CORRECCIÓN BACKWARD DEL PRIMER FRAME")
print("="*70)

if len(detections) >= 2 and detections[0]['success'] and detections[1]['success']:
    img1 = np.array(Image.open(image_paths[0]).convert("RGB"))
    img1_enhanced = enhance_bone_contrast(img1)
    
    detections[0] = correct_first_frame(
        img1, img1_enhanced,
        detections[0], detections[1],
        max_distance=35.0,
        max_size_diff_ratio=0.3
    )
    
    # Si se corrigió, regenerar máscara con point prompts
    if detections[0]['success']:
        box = detections[0]['box']
        center = detections[0]['center']
        center_x, center_y = center
        
        point_coords = np.array([[center_x, center_y]])
        point_labels = np.array([1])
        
        predictor.set_image(img1)
        masks, scores, logits = predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Usar selección mejorada también
        best_mask_idx = np.argmax(scores)  # Simplificado para backward
        
        masks_results[0] = {
            'masks': masks,
            'scores': scores,
            'best_mask': masks[best_mask_idx],
            'best_score': scores[best_mask_idx]
        }
        print(f"   ✅ I01 recorregida y re-segmentada con point prompts")

# Visualización
print("\n" + "="*70)
print("📊 GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for i, (img_path, detection, mask_result) in enumerate(zip(image_paths, detections, masks_results)):
    img = np.array(Image.open(img_path).convert("RGB"))
    
    # Fila 1: Imagen original con bbox
    img_bbox = img.copy()
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(img_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        title = f"I{i+1:02d}\nConf: {detection['confidence']:.3f}"
        color = 'green'
    else:
        title = f"I{i+1:02d}\nFALLIDO"
        color = 'red'
    
    axes[0, i].imshow(img_bbox)
    axes[0, i].set_title(title, color=color, fontweight='bold')
    axes[0, i].axis('off')
    
    # Fila 2: ROI
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        roi = img[y_min:y_max, x_min:x_max]
        axes[1, i].imshow(roi)
        axes[1, i].set_title(f"ROI {x_max-x_min}x{y_max-y_min}px")
    else:
        axes[1, i].text(0.5, 0.5, "No ROI", ha='center', va='center')
    axes[1, i].axis('off')
    
    # Fila 3: Máscara de SAM superpuesta
    if mask_result:
        # Crear overlay
        overlay = img.copy()
        mask = mask_result['best_mask']
        
        # Colorear máscara (azul semi-transparente)
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        # Dibujar contorno
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f"SAM Score: {mask_result['best_score']:.3f}")
    else:
        axes[2, i].text(0.5, 0.5, "No Mask", ha='center', va='center')
    axes[2, i].axis('off')

plt.suptitle("Pipeline Automático: Detección + SAM", fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = "test_results/automatic_sam_pipeline.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"💾 Guardado: {output_path}")

# Resumen
print("\n" + "="*70)
print("📈 RESUMEN FINAL")
print("="*70)

successful_detections = sum(1 for d in detections if d['success'])
successful_masks = sum(1 for m in masks_results if m is not None)

print(f"\n✅ Detecciones exitosas: {successful_detections}/{len(detections)}")
print(f"✅ Máscaras generadas: {successful_masks}/{len(detections)}")

if successful_detections > 0:
    avg_det_conf = np.mean([d['confidence'] for d in detections if d['success']])
    print(f"🎯 Confianza detección promedio: {avg_det_conf:.3f}")

if successful_masks > 0:
    avg_sam_score = np.mean([m['best_score'] for m in masks_results if m])
    print(f"🔮 Score SAM promedio: {avg_sam_score:.3f}")

# Analizar consistencia
centers = [d['center'] for d in detections if d['success'] and d['center']]
if len(centers) > 1:
    distances = []
    for j in range(1, len(centers)):
        dist = np.sqrt((centers[j][0] - centers[j-1][0])**2 + 
                      (centers[j][1] - centers[j-1][1])**2)
        distances.append(dist)
    
    print(f"\n📏 Consistencia temporal:")
    print(f"   Distancia promedio: {np.mean(distances):.1f} px")
    print(f"   Distancia máxima: {np.max(distances):.1f} px")

print("\n" + "="*70)
print("✅ PIPELINE COMPLETO FINALIZADO")
print("="*70 + "\n")
