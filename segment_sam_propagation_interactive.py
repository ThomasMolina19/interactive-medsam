#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentación completa de carpeta con SAM usando propagación de centros.
VERSIÓN INTERACTIVA: El usuario ingresa los paths por consola.

Proceso:
1. Usuario ingresa paths por consola
2. Usuario clickea la imagen del medio
3. Propaga hacia arriba y abajo usando centros calculados
4. Procesa TODAS las imágenes (no se detiene por umbral)
5. Registra advertencias cuando hay cambios grandes
"""

import sys
sys.path.append('path/to/segment-anything')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage
from skimage import morphology
import cv2
import os
from pathlib import Path
import glob
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox


def get_user_paths():
    """Solicita al usuario los 3 paths usando el Finder de macOS"""
    
    # Crear ventana root oculta
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    root.attributes('-topmost', True)  # Mantener diálogos al frente
    
    print("\n" + "="*70)
    print("🔧 CONFIGURACIÓN DE PATHS (usando Finder)")
    print("="*70)
    
    # 1. Seleccionar checkpoint SAM
    print("\n📦 Selecciona el archivo del CHECKPOINT SAM (.pth)...")
    ckpt = filedialog.askopenfilename(
        title="Seleccionar checkpoint SAM (.pth)",
        filetypes=[("PyTorch checkpoint", "*.pth"), ("Todos los archivos", "*.*")]
    )
    
    if not ckpt:
        print("❌ No se seleccionó checkpoint. Cancelado.")
        root.destroy()
        sys.exit(1)
    print(f"   ✅ Checkpoint: {ckpt}")
    
    # 2. Seleccionar directorio de imágenes
    print("\n📁 Selecciona la CARPETA con las imágenes (JPG/PNG)...")
    data_dir = filedialog.askdirectory(
        title="Seleccionar carpeta de imágenes"
    )
    
    if not data_dir:
        print("❌ No se seleccionó carpeta de imágenes. Cancelado.")
        root.destroy()
        sys.exit(1)
    print(f"   ✅ Imágenes: {data_dir}")
    
    # 3. Seleccionar directorio de salida
    print("\n💾 Selecciona la CARPETA de SALIDA para los resultados...")
    output_dir = filedialog.askdirectory(
        title="Seleccionar carpeta de salida"
    )
    
    if not output_dir:
        print("❌ No se seleccionó carpeta de salida. Cancelado.")
        root.destroy()
        sys.exit(1)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"   ✅ Salida: {output_dir}")
    
    # Mostrar resumen y confirmar
    print("\n" + "="*70)
    print("📋 RESUMEN DE CONFIGURACIÓN:")
    print(f"   • Checkpoint: {ckpt}")
    print(f"   • Imágenes:   {data_dir}")
    print(f"   • Salida:     {output_dir}")
    print("="*70)
    
    # Diálogo de confirmación
    confirm = messagebox.askyesno(
        "Confirmar configuración",
        f"¿Confirmar la siguiente configuración?\n\n"
        f"• Checkpoint:\n  {ckpt}\n\n"
        f"• Imágenes:\n  {data_dir}\n\n"
        f"• Salida:\n  {output_dir}"
    )
    
    root.destroy()
    
    if not confirm:
        print("❌ Cancelado por el usuario.")
        sys.exit(1)
    
    print("\n✅ Configuración confirmada!")
    return ckpt, data_dir, output_dir


# Obtener paths del usuario
ckpt, data_dir, output_dir = get_user_paths()

# Parámetros
SIMILARITY_THRESHOLD = 0.20  # 20% - Solo para advertencias, NO detiene la propagación
WARNING_THRESHOLD = 0.30     # 30% - Advertencia severa pero continúa

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n🖥️  Using device: {device}")

# Load SAM model
print("🔄 Loading SAM model...")
sam = sam_model_registry["vit_l"](checkpoint=ckpt)
sam = sam.to(device)
predictor = SamPredictor(sam)
print("✅ SAM model loaded!")


def read_image_file(filepath):
    """Read JPG or PNG file and return RGB array"""
    try:
        img = np.array(Image.open(filepath).convert("RGB"))
        return img
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return None


def refine_medical_mask(mask):
    """Clean up the segmentation mask"""
    if np.sum(mask) == 0:
        return mask
    
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth


def calculate_mask_center(mask):
    """Calculate the centroid of a binary mask"""
    if np.sum(mask) == 0:
        return None
    
    # Find center of mass
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    return [center_x, center_y]


def calculate_dice_coefficient(mask1, mask2):
    """Calculate Dice similarity coefficient between two masks"""
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    
    if sum_masks == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / sum_masks
    return dice


def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks"""
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


def calculate_negative_point(mask, center, distance_factor=0.30):
    """
    Calcula un punto negativo fuera de la máscara a una distancia del 30%.
    """
    if mask is None or np.sum(mask) == 0 or center is None:
        return None
    
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    radius = max(width, height) / 2
    
    offset = radius * (1 + distance_factor)
    
    h, w = mask.shape
    directions = [
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
    ]
    
    center_x, center_y = center
    
    for dx, dy in directions:
        neg_x = center_x + dx * offset
        neg_y = center_y + dy * offset
        
        if 0 <= neg_x < w and 0 <= neg_y < h:
            if not mask[int(neg_y), int(neg_x)]:
                return [neg_x, neg_y]
    
    for dx, dy in directions:
        neg_x = center_x + dx * offset * 1.5
        neg_y = center_y + dy * offset * 1.5
        
        if 0 <= neg_x < w and 0 <= neg_y < h:
            if not mask[int(neg_y), int(neg_x)]:
                return [neg_x, neg_y]
    
    return None


def segment_with_point(img, point, label=1, verbose=False):
    """Segment an image using a single point"""
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    predictor.set_image(img_enhanced)
    
    input_point = np.array([point])
    input_label = np.array([label])
    
    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        raw_area = np.sum(best_mask)
        if verbose:
            print(f"    🔍 SAM raw mask area: {raw_area} px, score: {best_score:.3f}")
        
        if raw_area == 0:
            if verbose:
                print(f"    ⚠️ SAM no generó máscara (punto fuera del objeto?)")
            return None, 0.0
        
        refined_mask = refine_medical_mask(best_mask)
        refined_area = np.sum(refined_mask)
        
        if refined_area == 0 and raw_area > 0:
            if verbose:
                print(f"    ⚠️ Refinamiento eliminó máscara ({raw_area}px < 500px), usando raw")
            refined_mask = ndimage.binary_fill_holes(best_mask)
        
        return refined_mask, best_score
        
    except Exception as e:
        print(f"    ⚠️ Error in segmentation: {e}")
        return None, 0.0


def segment_with_points(img, positive_points, negative_points):
    """Segment an image using multiple positive and negative points"""
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    predictor.set_image(img_enhanced)
    
    input_points = []
    input_labels = []
    
    for point in positive_points:
        input_points.append(point)
        input_labels.append(1)
    
    for point in negative_points:
        input_points.append(point)
        input_labels.append(0)
    
    if len(input_points) == 0:
        return None, 0.0
    
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)
    
    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        refined_mask = refine_medical_mask(best_mask)
        
        return refined_mask, best_score
        
    except Exception as e:
        print(f"⚠️  Error in segmentation: {e}")
        return None, 0.0


def run_segment_sam_points(image_path):
    """
    Ejecuta segment_sam_points.py con la imagen especificada.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "segment_sam_points.py")
    output_dir_local = os.path.join(script_dir, "segmentation_results")
    output_mask = os.path.join(output_dir_local, "segmentation_result_points.png")
    
    with open(script_path, 'r') as f:
        original_content = f.read()
    
    import re
    modified_content = re.sub(
        r'img = np\.array\(Image\.open\(".*?"\)\.convert\("RGB"\)\)',
        f'img = np.array(Image.open("{image_path}").convert("RGB"))',
        original_content
    )
    
    temp_script = os.path.join(script_dir, "_temp_segment_sam_points.py")
    
    try:
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        print(f"    🔄 Ejecutando segment_sam_points.py...")
        result = subprocess.run(
            [sys.executable, temp_script],
            cwd=script_dir,
            capture_output=False
        )
        
        if result.returncode != 0:
            print(f"    ❌ Error ejecutando segment_sam_points.py")
            return None
        
        if os.path.exists(output_mask):
            mask = np.array(Image.open(output_mask).convert("L"))
            mask = mask > 127
            return mask
        else:
            print(f"    ❌ No se encontró la máscara en {output_mask}")
            return None
            
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def save_segmentation_result(img, mask, filename, out_dir, center=None, seg_point=None, neg_point=None, info=""):
    """Save segmentation visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"{filename}\nOriginal")
    axes[0].axis('off')
    
    axes[1].imshow(img)
    axes[1].imshow(mask, alpha=0.5, cmap='Blues')
    
    if seg_point is not None:
        axes[1].plot(seg_point[0], seg_point[1], 'r*', markersize=18, markeredgewidth=2, label='Pto positivo')
    
    if neg_point is not None:
        axes[1].plot(neg_point[0], neg_point[1], 'bX', markersize=16, markeredgewidth=3, label='Pto negativo')
    
    if center is not None:
        axes[1].plot(center[0], center[1], 'g*', markersize=14, markeredgewidth=2, label='Centro máscara')
    
    if seg_point is not None or center is not None or neg_point is not None:
        axes[1].legend(loc='upper right', fontsize=8)
    
    axes[1].set_title(f"Overlay\n{info}")
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Mask\nArea: {np.sum(mask)} px")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(out_dir, f"{filename}_seg.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main function with iterative propagation"""
    
    # Get all image files (JPG or PNG)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    files = sorted(files, key=lambda x: os.path.basename(x))
    
    if len(files) == 0:
        print(f"❌ No image files found in {data_dir}")
        return
    
    print(f"📁 Found {len(files)} image files in {data_dir}")
    print(f"   Files: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
    
    # Find middle image
    middle_idx = len(files) // 2
    middle_file = files[middle_idx]
    middle_name = os.path.basename(middle_file).split('.')[0]
    
    print(f"\n🎯 Middle image: {os.path.basename(middle_file)} (index {middle_idx+1}/{len(files)})")
    
    middle_img = read_image_file(middle_file)
    
    if middle_img is None:
        print("❌ Error reading middle image!")
        return
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    
    # STEP 1: Usuario segmenta la imagen del medio
    print("\n" + "="*70)
    print("🖱️  PASO 1: SEGMENTACIÓN DE IMAGEN DEL MEDIO")
    print("="*70)
    print("Se abrirá segment_sam_points.py para segmentar la imagen del medio.")
    print("Instrucciones:")
    print("  • Click DERECHO: Punto positivo (dentro del objeto)")
    print("  • Click IZQUIERDO: Punto negativo (fuera del objeto)")
    print("  • Tecla 'z': Deshacer último punto")
    print("  • Tecla 'c': Limpiar todos los puntos")
    print("  • Cierra la ventana para guardar y continuar")
    print("="*70)
    
    middle_mask = run_segment_sam_points(middle_file)
    
    if middle_mask is None or np.sum(middle_mask) == 0:
        print("❌ No se pudo segmentar la imagen del medio!")
        return
    
    middle_score = 1.0
    
    print(f"\n📌 Segmentación completada usando segment_sam_points.py")
    
    middle_center = calculate_mask_center(middle_mask)
    
    if middle_center is None:
        print("❌ Failed to calculate center of middle segmentation!")
        return
    
    print(f"✅ Centro calculado: ({middle_center[0]:.0f}, {middle_center[1]:.0f})")
    print(f"   Score: {middle_score:.3f}, Area: {np.sum(middle_mask)} px")
    
    save_segmentation_result(middle_img, middle_mask, middle_name, output_dir, 
                            center=middle_center, seg_point=middle_center, 
                            info=f"Score: {middle_score:.3f}")
    
    # STEP 2: Segmentar TODAS las imágenes
    print("\n" + "="*70)
    print("🔄 PASO 2: SEGMENTACIÓN DE TODAS LAS IMÁGENES")
    print("="*70)
    print("⚡ Procesará TODAS las imágenes sin detenerse")
    print(f"⚠️  Advertencias cuando diferencia > {SIMILARITY_THRESHOLD*100:.0f}%")
    
    segmentations = {middle_idx: {
        'mask': middle_mask,
        'center': middle_center,
        'seg_point': middle_center,
        'score': middle_score,
        'area': np.sum(middle_mask)
    }}
    
    # STEP 3: Propagar a todas las imágenes
    print("\n" + "="*70)
    print("🔄 PASO 3: PROPAGACIÓN COMPLETA")
    print("="*70)
    print(f"📊 Umbrales de advertencia: {SIMILARITY_THRESHOLD*100:.0f}% (leve) / {WARNING_THRESHOLD*100:.0f}% (severa)")
    print("✅ Se procesarán TODAS las imágenes")
    print("="*70)
    
    # Propagate backwards (hacia arriba)
    print("\n📤 PROPAGACIÓN HACIA ARRIBA (imágenes anteriores)...")
    current_idx = middle_idx
    reference_mask = middle_mask
    reference_center = middle_center
    
    while current_idx > 0:
        prev_idx = current_idx - 1
        prev_file = files[prev_idx]
        prev_name = os.path.basename(prev_file).split('.')[0]
        
        print(f"\n  [{prev_idx+1}/{len(files)}] Procesando {os.path.basename(prev_file)}...")
        
        prev_img = read_image_file(prev_file)
        if prev_img is None:
            print(f"    ❌ Error leyendo imagen. Saltando...")
            current_idx = prev_idx
            continue
        
        prev_mask, prev_score = segment_with_point(prev_img, reference_center, verbose=True)
        
        if prev_mask is None or np.sum(prev_mask) == 0:
            print(f"    🔄 Intentando con centro ajustado...")
            found_valid = False
            offsets = [(0, 0), (-10, 0), (10, 0), (0, -10), (0, 10), (-20, 0), (20, 0), (0, -20), (0, 20)]
            
            for dx, dy in offsets[1:]:
                adjusted_point = [reference_center[0] + dx, reference_center[1] + dy]
                h, w = prev_img.shape[:2]
                if 0 <= adjusted_point[0] < w and 0 <= adjusted_point[1] < h:
                    prev_mask, prev_score = segment_with_point(prev_img, adjusted_point, verbose=False)
                    if prev_mask is not None and np.sum(prev_mask) > 0:
                        print(f"    ✅ Encontrada segmentación con offset ({dx}, {dy})")
                        found_valid = True
                        break
            
            if not found_valid:
                print(f"    ❌ Segmentación falló en todos los intentos. Saltando imagen...")
                current_idx = prev_idx
                continue
        
        dice = calculate_dice_coefficient(reference_mask, prev_mask)
        iou = calculate_iou(reference_mask, prev_mask)
        difference = 1.0 - dice
        
        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        
        used_neg_point = None
        
        if difference > WARNING_THRESHOLD:
            print(f"    🚨 ADVERTENCIA SEVERA: Diferencia ({difference*100:.1f}%) > {WARNING_THRESHOLD*100:.0f}%")
            print(f"    🔄 Reintentando con punto negativo...")
            
            negative_point = calculate_negative_point(reference_mask, reference_center, distance_factor=0.30)
            
            if negative_point is not None:
                print(f"    📍 Punto negativo: ({negative_point[0]:.0f}, {negative_point[1]:.0f})")
                
                prev_mask_retry, prev_score_retry = segment_with_points(
                    prev_img, 
                    positive_points=[reference_center], 
                    negative_points=[negative_point]
                )
                
                if prev_mask_retry is not None and np.sum(prev_mask_retry) > 0:
                    dice_retry = calculate_dice_coefficient(reference_mask, prev_mask_retry)
                    iou_retry = calculate_iou(reference_mask, prev_mask_retry)
                    difference_retry = 1.0 - dice_retry
                    
                    print(f"    📊 Nuevo Dice: {dice_retry:.3f}, IoU: {iou_retry:.3f}, Diferencia: {difference_retry*100:.1f}%")
                    
                    if dice_retry > dice:
                        print(f"    ✅ Segmentación mejoró de Dice {dice:.3f} a {dice_retry:.3f}")
                        prev_mask = prev_mask_retry
                        prev_score = prev_score_retry
                        dice = dice_retry
                        iou = iou_retry
                        difference = difference_retry
                        used_neg_point = list(negative_point)
                    else:
                        print(f"    ⚠️ La nueva segmentación no mejoró, manteniendo original")
                else:
                    print(f"    ⚠️ Reintento falló, manteniendo segmentación original")
            else:
                print(f"    ⚠️ No se pudo calcular punto negativo válido")
        elif difference > SIMILARITY_THRESHOLD:
            print(f"    ⚠️  Advertencia: Diferencia ({difference*100:.1f}%) > umbral ({SIMILARITY_THRESHOLD*100:.0f}%)")
        
        prev_center = calculate_mask_center(prev_mask)
        
        if prev_center is None:
            print(f"    ⚠️  No se pudo calcular centro. Usando centro anterior.")
            prev_center = reference_center
        
        print(f"    ✅ Válida. Centro: ({prev_center[0]:.0f}, {prev_center[1]:.0f})")
        
        used_seg_point = list(reference_center)
        
        segmentations[prev_idx] = {
            'mask': prev_mask,
            'center': prev_center,
            'seg_point': used_seg_point,
            'score': prev_score,
            'area': np.sum(prev_mask),
            'dice': dice,
            'iou': iou
        }
        
        save_segmentation_result(prev_img, prev_mask, prev_name, output_dir, 
                                center=prev_center, seg_point=used_seg_point, neg_point=used_neg_point,
                                info=f"Dice: {dice:.3f} | Score: {prev_score:.3f}")
        
        reference_mask = prev_mask
        reference_center = prev_center
        current_idx = prev_idx
    
    # Propagate forwards (hacia abajo)
    print("\n📥 PROPAGACIÓN HACIA ABAJO (imágenes posteriores)...")
    current_idx = middle_idx
    reference_mask = middle_mask
    reference_center = middle_center
    
    while current_idx < len(files) - 1:
        next_idx = current_idx + 1
        next_file = files[next_idx]
        next_name = os.path.basename(next_file).split('.')[0]
        
        print(f"\n  [{next_idx+1}/{len(files)}] Procesando {os.path.basename(next_file)}...")
        
        next_img = read_image_file(next_file)
        if next_img is None:
            print(f"    ❌ Error leyendo imagen. Saltando...")
            current_idx = next_idx
            continue
        
        next_mask, next_score = segment_with_point(next_img, reference_center, verbose=True)
        
        if next_mask is None or np.sum(next_mask) == 0:
            print(f"    🔄 Intentando con centro ajustado...")
            found_valid = False
            offsets = [(0, 0), (-10, 0), (10, 0), (0, -10), (0, 10), (-20, 0), (20, 0), (0, -20), (0, 20)]
            
            for dx, dy in offsets[1:]:
                adjusted_point = [reference_center[0] + dx, reference_center[1] + dy]
                h, w = next_img.shape[:2]
                if 0 <= adjusted_point[0] < w and 0 <= adjusted_point[1] < h:
                    next_mask, next_score = segment_with_point(next_img, adjusted_point, verbose=False)
                    if next_mask is not None and np.sum(next_mask) > 0:
                        print(f"    ✅ Encontrada segmentación con offset ({dx}, {dy})")
                        found_valid = True
                        break
            
            if not found_valid:
                print(f"    ❌ Segmentación falló en todos los intentos. Saltando imagen...")
                current_idx = next_idx
                continue
        
        dice = calculate_dice_coefficient(reference_mask, next_mask)
        iou = calculate_iou(reference_mask, next_mask)
        difference = 1.0 - dice
        
        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        
        used_neg_point = None
        
        if difference > WARNING_THRESHOLD:
            print(f"    🚨 ADVERTENCIA SEVERA: Diferencia ({difference*100:.1f}%) > {WARNING_THRESHOLD*100:.0f}%")
            print(f"    🔄 Reintentando con punto negativo...")
            
            negative_point = calculate_negative_point(reference_mask, reference_center, distance_factor=0.30)
            
            if negative_point is not None:
                print(f"    📍 Punto negativo: ({negative_point[0]:.0f}, {negative_point[1]:.0f})")
                
                next_mask_retry, next_score_retry = segment_with_points(
                    next_img, 
                    positive_points=[reference_center], 
                    negative_points=[negative_point]
                )
                
                if next_mask_retry is not None and np.sum(next_mask_retry) > 0:
                    dice_retry = calculate_dice_coefficient(reference_mask, next_mask_retry)
                    iou_retry = calculate_iou(reference_mask, next_mask_retry)
                    difference_retry = 1.0 - dice_retry
                    
                    print(f"    📊 Nuevo Dice: {dice_retry:.3f}, IoU: {iou_retry:.3f}, Diferencia: {difference_retry*100:.1f}%")
                    
                    if dice_retry > dice:
                        print(f"    ✅ Segmentación mejoró de Dice {dice:.3f} a {dice_retry:.3f}")
                        next_mask = next_mask_retry
                        next_score = next_score_retry
                        dice = dice_retry
                        iou = iou_retry
                        difference = difference_retry
                        used_neg_point = list(negative_point)
                    else:
                        print(f"    ⚠️ La nueva segmentación no mejoró, manteniendo original")
                else:
                    print(f"    ⚠️ Reintento falló, manteniendo segmentación original")
            else:
                print(f"    ⚠️ No se pudo calcular punto negativo válido")
        elif difference > SIMILARITY_THRESHOLD:
            print(f"    ⚠️  Advertencia: Diferencia ({difference*100:.1f}%) > umbral ({SIMILARITY_THRESHOLD*100:.0f}%)")
        
        next_center = calculate_mask_center(next_mask)
        
        if next_center is None:
            print(f"    ⚠️  No se pudo calcular centro. Usando centro anterior.")
            next_center = reference_center
        
        print(f"    ✅ Válida. Centro: ({next_center[0]:.0f}, {next_center[1]:.0f})")
        
        used_seg_point = list(reference_center)
        
        segmentations[next_idx] = {
            'mask': next_mask,
            'center': next_center,
            'seg_point': used_seg_point,
            'score': next_score,
            'area': np.sum(next_mask),
            'dice': dice,
            'iou': iou
        }
        
        save_segmentation_result(next_img, next_mask, next_name, output_dir, 
                                center=next_center, seg_point=used_seg_point, neg_point=used_neg_point,
                                info=f"Dice: {dice:.3f} | Score: {next_score:.3f}")
        
        reference_mask = next_mask
        reference_center = next_center
        current_idx = next_idx
    
    # Summary
    print("\n" + "="*70)
    print("🎉 PROCESAMIENTO COMPLETADO - CARPETA COMPLETA")
    print("="*70)
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"✅ Segmentaciones exitosas: {len(segmentations)}/{len(files)} imágenes")
    
    warnings_count = sum(1 for s in segmentations.values() if 'dice' in s and (1.0 - s['dice']) > SIMILARITY_THRESHOLD)
    severe_warnings = sum(1 for s in segmentations.values() if 'dice' in s and (1.0 - s['dice']) > WARNING_THRESHOLD)
    
    if warnings_count > 0:
        print(f"⚠️  Imágenes con advertencias: {warnings_count}")
    if severe_warnings > 0:
        print(f"🚨 Imágenes con advertencias severas: {severe_warnings}")
    
    if len(segmentations) > 1:
        dice_scores = [s['dice'] for s in segmentations.values() if 'dice' in s]
        if dice_scores:
            print(f"📊 Estadísticas de similitud:")
            print(f"   - Dice promedio: {np.mean(dice_scores):.3f}")
            print(f"   - Dice mínimo: {np.min(dice_scores):.3f}")
            print(f"   - Dice máximo: {np.max(dice_scores):.3f}")
    
    print("="*70)
    
    # Save summary
    summary_path = os.path.join(output_dir, "propagation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SAM Complete Folder Segmentation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Checkpoint: {ckpt}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Middle image: {os.path.basename(files[middle_idx])} (index {middle_idx+1})\n")
        f.write(f"Initial segmentation: segment_sam_points.py\n")
        f.write(f"Warning threshold: {SIMILARITY_THRESHOLD*100:.0f}%\n")
        f.write(f"Severe warning threshold: {WARNING_THRESHOLD*100:.0f}%\n")
        f.write(f"Total images: {len(files)}\n")
        f.write(f"Successfully segmented: {len(segmentations)}\n")
        f.write(f"Images with warnings: {warnings_count}\n")
        f.write(f"Images with severe warnings: {severe_warnings}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Results per image:\n")
        f.write("-"*70 + "\n")
        
        for idx in sorted(segmentations.keys()):
            s = segmentations[idx]
            filename = os.path.basename(files[idx])
            dice_str = f"{s['dice']:.3f}" if 'dice' in s else "REF"
            
            warning_marker = ""
            if 'dice' in s:
                diff = 1.0 - s['dice']
                if diff > WARNING_THRESHOLD:
                    warning_marker = " [SEVERA]"
                elif diff > SIMILARITY_THRESHOLD:
                    warning_marker = " [ADVERTENCIA]"
            
            f.write(f"{filename}: Dice={dice_str}, Area={s['area']}px{warning_marker}\n")
    
    print(f"\n📝 Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
