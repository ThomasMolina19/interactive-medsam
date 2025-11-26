#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch segmentation with SAM using a reference point from middle image.
Processes entire DICOM dataset with a single manual click.
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
import pydicom
import os
from pathlib import Path


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Paths
ckpt = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/Checkpoints/sam_vit_l_0b3195.pth"
data_dir = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/D1"
output_dir = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/D1_segmentations"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load SAM model
print("🔄 Loading SAM model...")
sam = sam_model_registry["vit_l"](checkpoint=ckpt)
sam = sam.to(device)
predictor = SamPredictor(sam)
print("✅ SAM model loaded!")


def read_dicom_file(filepath):
    """Read DICOM file and return image array"""
    try:
        dcm = pydicom.dcmread(filepath)
        img_array = dcm.pixel_array
        
        # Normalize to 0-255
        img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        
        return img_rgb
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return None


def refine_medical_mask(mask):
    """Clean up the segmentation mask for medical images"""
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth


def interactive_point_selector(img, img_name):
    """
    Interactive point selection for the middle image.
    Click derecho = punto positivo
    Click izquierdo = punto negativo
    Tecla 'z' = deshacer último punto
    Tecla 'c' = limpiar todos los puntos
    Presiona ENTER para continuar
    """
    
    class PointSelector:
        def __init__(self, ax):
            self.positive_points = []
            self.negative_points = []
            self.ax = ax
            self.point_markers = []
            
        def onclick(self, event):
            if event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            
            # Botón izquierdo (1) = Punto NEGATIVO
            if event.button == 1:
                self.negative_points.append([x, y])
                marker = self.ax.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_markers.append(('neg', marker))
                print(f"❌ Punto NEGATIVO: ({x:.0f}, {y:.0f})")
                
            # Botón derecho (3) = Punto POSITIVO
            elif event.button == 3:
                self.positive_points.append([x, y])
                marker = self.ax.plot(x, y, 'g*', markersize=20, markeredgewidth=2)[0]
                self.point_markers.append(('pos', marker))
                print(f"✅ Punto POSITIVO: ({x:.0f}, {y:.0f})")
            
            self.ax.set_title(f"{img_name} | ✅ {len(self.positive_points)} positivos | ❌ {len(self.negative_points)} negativos | 'z': deshacer | 'c': limpiar")
            plt.draw()
        
        def onkey(self, event):
            """Handle keyboard events"""
            # Z = Undo last point
            if event.key == 'z':
                if len(self.point_markers) > 0:
                    point_type, marker = self.point_markers.pop()
                    marker.remove()
                    
                    if point_type == 'pos' and len(self.positive_points) > 0:
                        removed = self.positive_points.pop()
                        print(f"↩️  Deshecho punto POSITIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    elif point_type == 'neg' and len(self.negative_points) > 0:
                        removed = self.negative_points.pop()
                        print(f"↩️  Deshecho punto NEGATIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    
                    self.ax.set_title(f"{img_name} | ✅ {len(self.positive_points)} positivos | ❌ {len(self.negative_points)} negativos | 'z': deshacer | 'c': limpiar")
                    plt.draw()
            
            # C = Clear all points
            elif event.key == 'c':
                for _, marker in self.point_markers:
                    marker.remove()
                self.point_markers.clear()
                self.positive_points.clear()
                self.negative_points.clear()
                print("🧹 Todos los puntos limpiados")
                self.ax.set_title(f"{img_name} | ✅ 0 positivos | ❌ 0 negativos | 'z': deshacer | 'c': limpiar")
                plt.draw()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.set_title(f"{img_name} | Click derecho=POSITIVO | Click izquierdo=NEGATIVO | 'z': deshacer | 'c': limpiar | ENTER: continuar")
    ax.axis('off')
    
    selector_obj = PointSelector(ax)
    fig.canvas.mpl_connect('button_press_event', selector_obj.onclick)
    fig.canvas.mpl_connect('key_press_event', selector_obj.onkey)
    
    plt.tight_layout()
    plt.show()
    
    return selector_obj.positive_points, selector_obj.negative_points


def segment_image_with_points(img, positive_points, negative_points):
    """Segment an image using the reference points"""
    # Enhance contrast
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    # Set image for SAM
    predictor.set_image(img_enhanced)
    
    # Prepare points
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
    
    # Generate masks
    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
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


def show_dataset_preview(files, data_dir):
    """
    Show 6 sample images from the dataset as preview.
    User closes the window to continue.
    """
    print("\n" + "="*60)
    print("📷 VISTA PREVIA DEL DATASET")
    print("="*60)
    print("Mostrando 6 imágenes del dataset...")
    print("Cierra la ventana para continuar con la segmentación")
    print("="*60)
    
    # Select 6 images evenly distributed
    n_images = min(6, len(files))
    indices = np.linspace(0, len(files)-1, n_images, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        filename = files[idx]
        filepath = os.path.join(data_dir, filename)
        img = read_dicom_file(filepath)
        
        if img is not None:
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"{filename} (#{idx+1}/{len(files)})", fontsize=14, weight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[i].set_title(f"{filename} - Error")
            axes[i].axis('off')
    
    plt.suptitle(f"Dataset Preview: {len(files)} imágenes en total\nCierra esta ventana para continuar", 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ Vista previa cerrada. Continuando con la segmentación...\n")


def main():
    """Main function"""
    
    # Get all DICOM files (excluding .DS_Store and pngs folder)
    files = sorted([f for f in os.listdir(data_dir) 
                   if not f.startswith('.') and os.path.isfile(os.path.join(data_dir, f)) and f.startswith('I')])
    
    print(f"📁 Found {len(files)} DICOM files in {data_dir}")
    print(f"   Files: {files[0]} ... {files[-1]}")
    
    if len(files) == 0:
        print("❌ No files found!")
        return
    
    # STEP 1: Show dataset preview (6 images)
    show_dataset_preview(files, data_dir)
    
    # Find middle image
    middle_idx = len(files) // 2
    middle_file = files[middle_idx]
    
    print(f"\n🎯 Middle image: {middle_file} (index {middle_idx+1}/{len(files)})")
    
    # Read middle image
    middle_path = os.path.join(data_dir, middle_file)
    middle_img = read_dicom_file(middle_path)
    
    if middle_img is None:
        print("❌ Error reading middle image!")
        return
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    
    # STEP 2: Interactive point selection on middle image
    print("\n" + "="*60)
    print("🖱️  SELECCIÓN MANUAL DE PUNTO DE REFERENCIA")
    print("="*60)
    print("Instrucciones:")
    print("  • Click DERECHO: Punto positivo (dentro del objeto)")
    print("  • Click IZQUIERDO: Punto negativo (fuera del objeto)")
    print("  • Tecla 'z': Deshacer último punto")
    print("  • Tecla 'c': Limpiar todos los puntos")
    print("  • ENTER o cierra ventana: Continuar")
    print("="*60)
    
    positive_points, negative_points = interactive_point_selector(middle_img, middle_file)
    
    if len(positive_points) == 0 and len(negative_points) == 0:
        print("❌ No points selected! Exiting...")
        return
    
    print(f"\n📌 Reference points selected:")
    print(f"   ✅ Positive points: {len(positive_points)}")
    print(f"   ❌ Negative points: {len(negative_points)}")
    
    # Process all images with the same points
    print(f"\n🔄 Processing all {len(files)} images...")
    print("="*60)
    
    results = []
    
    for i, filename in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing {filename}...", end=" ")
        
        filepath = os.path.join(data_dir, filename)
        img = read_dicom_file(filepath)
        
        if img is None:
            print("❌ Error reading file")
            continue
        
        # Segment with reference points
        mask, score = segment_image_with_points(img, positive_points, negative_points)
        
        if mask is None:
            print("❌ Segmentation failed")
            continue
        
        area = np.sum(mask)
        print(f"✅ Score: {score:.3f} | Area: {area} px")
        
        # Save visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(img)
        axes[0].set_title(f"{filename}\nOriginal")
        axes[0].axis('off')
        
        # Overlay
        axes[1].imshow(img)
        axes[1].imshow(mask, alpha=0.5, cmap='Blues')
        for point in positive_points:
            axes[1].plot(point[0], point[1], 'g*', markersize=12, markeredgewidth=2)
        for point in negative_points:
            axes[1].plot(point[0], point[1], 'rx', markersize=10, markeredgewidth=2)
        axes[1].set_title(f"Overlay\nScore: {score:.3f}")
        axes[1].axis('off')
        
        # Mask
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f"Mask\nArea: {area} px")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename}_segmentation.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Skip saving individual masks
        # mask_path = os.path.join(output_dir, f"{filename}_mask.png")
        # mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        # mask_pil.save(mask_path)
        
        results.append({
            'filename': filename,
            'score': score,
            'area': area
        })
    
    # Summary
    print("\n" + "="*60)
    print("🎉 PROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"📁 Results saved to: {output_dir}")
    print(f"✅ Successfully processed: {len(results)}/{len(files)} images")
    
    if results:
        avg_score = np.mean([r['score'] for r in results])
        avg_area = np.mean([r['area'] for r in results])
        print(f"📊 Average score: {avg_score:.3f}")
        print(f"📏 Average area: {avg_area:.0f} pixels")
    
    print("="*60)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SAM Batch Segmentation Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Reference image: {middle_file}\n")
        f.write(f"Positive points: {len(positive_points)}\n")
        f.write(f"Negative points: {len(negative_points)}\n")
        f.write(f"Total images: {len(files)}\n")
        f.write(f"Successfully processed: {len(results)}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Results per image:\n")
        f.write("-"*60 + "\n")
        for r in results:
            f.write(f"{r['filename']:<10} | Score: {r['score']:.3f} | Area: {r['area']:>8.0f} px\n")
    
    print(f"💾 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
