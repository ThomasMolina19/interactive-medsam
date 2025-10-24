"""
Debug específico del primer frame
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

# Cargar I01 y I02
img1 = np.array(Image.open("dicom_pngs/I01.png").convert("RGB"))
img2 = np.array(Image.open("dicom_pngs/I02.png").convert("RGB"))

img1_enh = enhance_bone_contrast(img1)
img2_enh = enhance_bone_contrast(img2)

print("\n" + "="*70)
print("🔍 DEBUG: Primer Frame")
print("="*70)

# Detectar I01 SIN referencia
print("\n1️⃣  I01 SIN referencia:")
det1_no_ref = detect_humerus_automatic(img1, img1_enh, method='combined', 
                                        return_all_candidates=True)
if det1_no_ref['success']:
    params = det1_no_ref['detection']['params']
    if len(params) == 4:
        x, y, r, acc = params
    else:
        x, y, r = params
        acc = 0
    print(f"   Centro: ({x}, {y}), Radio: {r}px, Acc: {acc:.3f}")
    print(f"   Confianza: {det1_no_ref['confidence']:.3f}")
    print(f"   Total candidatos: {len(det1_no_ref['all_candidates'])}")

# Detectar I02
print("\n2️⃣  I02:")
det2 = detect_humerus_automatic(img2, img2_enh, method='combined')
if det2['success']:
    center2 = det2['center']
    params = det2['detection']['params']
    if len(params) == 4:
        x, y, r, acc = params
    else:
        x, y, r = params
        acc = 0
    print(f"   Centro: ({x}, {y}), Radio: {r}px, Acc: {acc:.3f}")
    print(f"   Confianza: {det2['confidence']:.3f}")

# Detectar I01 CON referencia de I02
print("\n3️⃣  I01 CON referencia de I02:")
det1_with_ref = detect_humerus_automatic(img1, img1_enh, method='combined',
                                         previous_center=center2, search_radius=80.0,
                                         return_all_candidates=True)
if det1_with_ref['success']:
    params = det1_with_ref['detection']['params']
    if len(params) == 4:
        x, y, r, acc = params
    else:
        x, y, r = params
        acc = 0
    print(f"   Centro: ({x}, {y}), Radio: {r}px, Acc: {acc:.3f}")
    print(f"   Confianza: {det1_with_ref['confidence']:.3f}")
    print(f"   Candidatos filtrados: {len(det1_with_ref['all_candidates'])}")

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# I01 sin referencia
img1_vis1 = img1.copy()
if det1_no_ref['success']:
    box = det1_no_ref['box']
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(img1_vis1, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    params = det1_no_ref['detection']['params']
    if len(params) == 4:
        x, y, r, _ = params
    else:
        x, y, r = params
    cv2.circle(img1_vis1, (x, y), r, (255, 0, 0), 2)

axes[0].imshow(img1_vis1)
axes[0].set_title(f"I01 SIN referencia\nRadio: {r}px")
axes[0].axis('off')

# I02
img2_vis = img2.copy()
if det2['success']:
    box = det2['box']
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(img2_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    params = det2['detection']['params']
    if len(params) == 4:
        x, y, r, _ = params
    else:
        x, y, r = params
    cv2.circle(img2_vis, (x, y), r, (255, 0, 0), 2)

axes[1].imshow(img2_vis)
axes[1].set_title(f"I02 (referencia)\nRadio: {r}px")
axes[1].axis('off')

# I01 con referencia
img1_vis2 = img1.copy()
if det1_with_ref['success']:
    box = det1_with_ref['box']
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(img1_vis2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    params = det1_with_ref['detection']['params']
    if len(params) == 4:
        x, y, r, _ = params
    else:
        x, y, r = params
    cv2.circle(img1_vis2, (x, y), r, (255, 0, 0), 2)

axes[2].imshow(img1_vis2)
axes[2].set_title(f"I01 CON referencia\nRadio: {r}px")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("test_results/debug_first_frame.png", dpi=150)
print("\n💾 Guardado: test_results/debug_first_frame.png")
print("="*70 + "\n")
