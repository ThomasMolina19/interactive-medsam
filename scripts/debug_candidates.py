"""
Debug de todos los candidatos en I01
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

# Cargar I01
img1 = np.array(Image.open("dicom_pngs/I01.png").convert("RGB"))
img1_enh = enhance_bone_contrast(img1)

print("\n" + "="*70)
print("🔍 DEBUG: Todos los Candidatos en I01")
print("="*70)

# Detectar con todos los candidatos
det = detect_humerus_automatic(img1, img1_enh, method='combined', 
                                return_all_candidates=True)

if det['all_candidates']:
    print(f"\nTotal candidatos: {len(det['all_candidates'])}")
    print("\nTop 10 candidatos por score:\n")
    
    # Calcular scores para todos
    from src.detection import score_humerus_candidate
    
    scored_candidates = []
    for cand in det['all_candidates']:
        score = score_humerus_candidate(cand, img1.shape, img1)
        cand['score'] = score
        scored_candidates.append(cand)
    
    # Ordenar por score
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    for i, cand in enumerate(scored_candidates[:10]):
        if cand['type'] == 'circle':
            params = cand['params']
            if len(params) == 4:
                x, y, r, acc = params
            else:
                x, y, r = params
                acc = 0
            area = np.pi * r * r
            print(f"{i+1:2d}. Centro: ({x:3d}, {y:3d}), Radio: {r:2d}px, "
                  f"Área: {int(area):4d}px², Score: {cand['score']:.3f}, Acc: {acc:.3f}")

# Visualizar top 5
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(min(6, len(scored_candidates))):
    img_vis = img1.copy()
    cand = scored_candidates[i]
    
    if cand['type'] == 'circle':
        params = cand['params']
        if len(params) == 4:
            x, y, r, _ = params
        else:
            x, y, r = params
        
        # Dibujar círculo
        cv2.circle(img_vis, (x, y), r, (255, 0, 0), 2)
        cv2.circle(img_vis, (x, y), 3, (255, 0, 0), -1)
        
        # Dibujar bounding box
        r_exp = int(r * 1.4)
        x_min = max(0, x - r_exp)
        y_min = max(0, y - r_exp)
        x_max = min(img1.shape[1], x + r_exp)
        y_max = min(img1.shape[0], y + r_exp)
        cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        area = int(np.pi * r * r)
        axes[i].imshow(img_vis)
        axes[i].set_title(f"#{i+1}: r={r}px, área={area}px²\nScore: {cand['score']:.3f}")
        axes[i].axis('off')

plt.tight_layout()
plt.savefig("test_results/debug_candidates.png", dpi=150)
print(f"\n💾 Guardado: test_results/debug_candidates.png")
print("="*70 + "\n")
