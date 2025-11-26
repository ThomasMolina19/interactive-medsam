#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Propagación iterativa de segmentación con SAM usando validación de similitud.
Proceso:
1. Usuario clickea la imagen del medio
2. Segmenta anterior y posterior
3. Valida similitud (< 10% diferencia)
4. Propaga usando centros calculados hacia arriba/abajo
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


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Paths - MODIFICAR SEGÚN TUS NECESIDADES
ckpt = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/Checkpoints/sam_vit_l_0b3195.pth"
data_dir = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/D1/pngs"  # Carpeta con JPG o PNG
output_dir = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/D1_propagation_results"

# Parámetros
SIMILARITY_THRESHOLD = 0.20  # 20% diferencia máxima permitida (era 10%)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

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


def segment_with_point(img, point, label=1):
    """Segment an image using a single point"""
    # Enhance contrast
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    # Set image for SAM
    predictor.set_image(img_enhanced)
    
    # Prepare point
    input_point = np.array([point])
    input_label = np.array([label])
    
    # Generate masks
    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        # Refine mask
        refined_mask = refine_medical_mask(best_mask)
        
        return refined_mask, best_score
        
    except Exception as e:
        print(f"⚠️  Error in segmentation: {e}")
        return None, 0.0


def interactive_point_selector(img, img_name):
    """Interactive point selection - single click derecho"""
    
    class PointSelector:
        def __init__(self, ax):
            self.point = None
            self.ax = ax
            self.marker = None
            
        def onclick(self, event):
            if event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            
            # Solo click derecho = punto positivo
            if event.button == 3:
                x, y = event.xdata, event.ydata
                
                # Remove previous marker
                if self.marker is not None:
                    self.marker.remove()
                
                self.point = [x, y]
                self.marker = self.ax.plot(x, y, 'g*', markersize=25, markeredgewidth=3)[0]
                print(f"✅ Punto seleccionado: ({x:.0f}, {y:.0f})")
                
                self.ax.set_title(f"{img_name} | ✅ Punto seleccionado: ({x:.0f}, {y:.0f}) | Cierra ventana")
                plt.draw()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.set_title(f"{img_name} | Click derecho en el centro del húmero | Cierra ventana para continuar")
    ax.axis('off')
    
    selector_obj = PointSelector(ax)
    fig.canvas.mpl_connect('button_press_event', selector_obj.onclick)
    
    plt.tight_layout()
    plt.show()
    
    return selector_obj.point


def save_segmentation_result(img, mask, filename, output_dir, center=None, info=""):
    """Save segmentation visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title(f"{filename}\nOriginal")
    axes[0].axis('off')
    
    # Overlay
    axes[1].imshow(img)
    axes[1].imshow(mask, alpha=0.5, cmap='Blues')
    if center is not None:
        axes[1].plot(center[0], center[1], 'r*', markersize=20, markeredgewidth=3)
        axes[1].plot(center[0], center[1], 'yo', markersize=10, fillstyle='none', markeredgewidth=2)
    axes[1].set_title(f"Overlay\n{info}")
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Mask\nArea: {np.sum(mask)} px")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{filename}_seg.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main function with iterative propagation"""
    
    # Get all image files (JPG or PNG)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    # Sort files by name
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
    
    # Read middle image
    middle_img = read_image_file(middle_file)
    
    if middle_img is None:
        print("❌ Error reading middle image!")
        return
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    
    # STEP 1: Usuario clickea en la imagen del medio
    print("\n" + "="*70)
    print("🖱️  PASO 1: SELECCIÓN DE PUNTO EN IMAGEN DEL MEDIO")
    print("="*70)
    print("Instrucciones:")
    print("  • Click DERECHO en el centro del húmero")
    print("  • Cierra la ventana para continuar")
    print("="*70)
    
    initial_point = interactive_point_selector(middle_img, os.path.basename(middle_file))
    
    if initial_point is None:
        print("❌ No point selected! Exiting...")
        return
    
    print(f"\n📌 Punto inicial seleccionado: ({initial_point[0]:.0f}, {initial_point[1]:.0f})")
    
    # Segment middle image
    print(f"🔄 Segmentando imagen del medio...")
    middle_mask, middle_score = segment_with_point(middle_img, initial_point)
    
    if middle_mask is None or np.sum(middle_mask) == 0:
        print("❌ Failed to segment middle image!")
        return
    
    # Calculate center of middle segmentation
    middle_center = calculate_mask_center(middle_mask)
    
    if middle_center is None:
        print("❌ Failed to calculate center of middle segmentation!")
        return
    
    print(f"✅ Centro calculado: ({middle_center[0]:.0f}, {middle_center[1]:.0f})")
    print(f"   Score: {middle_score:.3f}, Area: {np.sum(middle_mask)} px")
    
    # Save middle segmentation
    save_segmentation_result(middle_img, middle_mask, middle_name, output_dir, 
                            middle_center, f"Score: {middle_score:.3f}")
    
    # STEP 2: Segmentar imagen anterior y posterior
    print("\n" + "="*70)
    print("🔄 PASO 2: SEGMENTACIÓN DE IMÁGENES ANTERIOR Y POSTERIOR")
    print("="*70)
    
    # Initialize results storage
    segmentations = {middle_idx: {
        'mask': middle_mask,
        'center': middle_center,
        'score': middle_score,
        'area': np.sum(middle_mask)
    }}
    
    # STEP 3: Validar y propagar
    print("\n" + "="*70)
    print("🔄 PASO 3: PROPAGACIÓN ITERATIVA CON VALIDACIÓN")
    print("="*70)
    print(f"Umbral de similitud: {(1-SIMILARITY_THRESHOLD)*100:.0f}% (diferencia máxima: {SIMILARITY_THRESHOLD*100:.0f}%)")
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
        
        # Read image
        prev_img = read_image_file(prev_file)
        if prev_img is None:
            print(f"    ❌ Error leyendo imagen. Deteniendo propagación hacia arriba.")
            break
        
        # Segment using reference center
        prev_mask, prev_score = segment_with_point(prev_img, reference_center)
        
        if prev_mask is None or np.sum(prev_mask) == 0:
            print(f"    ❌ Segmentación falló. Deteniendo propagación hacia arriba.")
            break
        
        # Calculate similarity with reference
        dice = calculate_dice_coefficient(reference_mask, prev_mask)
        iou = calculate_iou(reference_mask, prev_mask)
        difference = 1.0 - dice
        
        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        
        # VALIDATION: Check if difference is acceptable
        if difference > SIMILARITY_THRESHOLD:
            print(f"    🚩 BANDERA: Diferencia ({difference*100:.1f}%) > umbral ({SIMILARITY_THRESHOLD*100:.0f}%)")
            print(f"    ⚠️  Propagación detenida. Revisar imagen {os.path.basename(prev_file)}")
            break
        
        # Calculate center for next iteration
        prev_center = calculate_mask_center(prev_mask)
        
        if prev_center is None:
            print(f"    ❌ No se pudo calcular centro. Deteniendo propagación hacia arriba.")
            break
        
        print(f"    ✅ Válida. Centro: ({prev_center[0]:.0f}, {prev_center[1]:.0f})")
        
        # Save result
        segmentations[prev_idx] = {
            'mask': prev_mask,
            'center': prev_center,
            'score': prev_score,
            'area': np.sum(prev_mask),
            'dice': dice,
            'iou': iou
        }
        
        save_segmentation_result(prev_img, prev_mask, prev_name, output_dir, 
                                prev_center, f"Dice: {dice:.3f} | Score: {prev_score:.3f}")
        
        # Update reference for next iteration
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
        
        # Read image
        next_img = read_image_file(next_file)
        if next_img is None:
            print(f"    ❌ Error leyendo imagen. Deteniendo propagación hacia abajo.")
            break
        
        # Segment using reference center
        next_mask, next_score = segment_with_point(next_img, reference_center)
        
        if next_mask is None or np.sum(next_mask) == 0:
            print(f"    ❌ Segmentación falló. Deteniendo propagación hacia abajo.")
            break
        
        # Calculate similarity with reference
        dice = calculate_dice_coefficient(reference_mask, next_mask)
        iou = calculate_iou(reference_mask, next_mask)
        difference = 1.0 - dice
        
        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        
        # VALIDATION: Check if difference is acceptable
        if difference > SIMILARITY_THRESHOLD:
            print(f"    🚩 BANDERA: Diferencia ({difference*100:.1f}%) > umbral ({SIMILARITY_THRESHOLD*100:.0f}%)")
            print(f"    ⚠️  Propagación detenida. Revisar imagen {os.path.basename(next_file)}")
            break
        
        # Calculate center for next iteration
        next_center = calculate_mask_center(next_mask)
        
        if next_center is None:
            print(f"    ❌ No se pudo calcular centro. Deteniendo propagación hacia abajo.")
            break
        
        print(f"    ✅ Válida. Centro: ({next_center[0]:.0f}, {next_center[1]:.0f})")
        
        # Save result
        segmentations[next_idx] = {
            'mask': next_mask,
            'center': next_center,
            'score': next_score,
            'area': np.sum(next_mask),
            'dice': dice,
            'iou': iou
        }
        
        save_segmentation_result(next_img, next_mask, next_name, output_dir, 
                                next_center, f"Dice: {dice:.3f} | Score: {next_score:.3f}")
        
        # Update reference for next iteration
        reference_mask = next_mask
        reference_center = next_center
        current_idx = next_idx
    
    # Summary
    print("\n" + "="*70)
    print("🎉 PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"✅ Segmentaciones exitosas: {len(segmentations)}/{len(files)} imágenes")
    print(f"   - Imagen central: {middle_idx+1}")
    print(f"   - Rango procesado: {min(segmentations.keys())+1} - {max(segmentations.keys())+1}")
    
    # Calculate statistics
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
        f.write("SAM Iterative Propagation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Middle image: {os.path.basename(middle_file)} (index {middle_idx+1})\n")
        f.write(f"Initial point: ({initial_point[0]:.0f}, {initial_point[1]:.0f})\n")
        f.write(f"Similarity threshold: {SIMILARITY_THRESHOLD*100:.0f}%\n")
        f.write(f"Total images: {len(files)}\n")
        f.write(f"Successfully segmented: {len(segmentations)}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Results per image:\n")
        f.write("-"*70 + "\n")
        
        for idx in sorted(segmentations.keys()):
            s = segmentations[idx]
            filename = os.path.basename(files[idx])
            dice_str = f"{s['dice']:.3f}" if 'dice' in s else "REF"
            f.write(f"{idx+1:3d}. {filename:<15} | Area: {s['area']:>7.0f} px | Dice: {dice_str} | Score: {s['score']:.3f}\n")
    
    print(f"💾 Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
