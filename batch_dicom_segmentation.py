import sys
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage
from skimage import morphology
import cv2
import pydicom
from datetime import datetime
import json


def load_dicom_as_image(dicom_path):
    """
    Load DICOM file and convert to RGB image array
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Extract pixel array
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 0-255 range
        if pixel_array.dtype != np.uint8:
            # Handle different bit depths
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                # Use DICOM windowing if available
                center = dicom_data.WindowCenter
                width = dicom_data.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]
                
                min_val = center - width // 2
                max_val = center + width // 2
                pixel_array = np.clip(pixel_array, min_val, max_val)
                pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                # Simple normalization
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB (duplicate grayscale to 3 channels)
        if len(pixel_array.shape) == 2:
            rgb_image = np.stack([pixel_array] * 3, axis=-1)
        else:
            rgb_image = pixel_array
            
        return rgb_image, dicom_data
        
    except Exception as e:
        print(f"⚠️ Error loading DICOM {dicom_path}: {e}")
        return None, None


def interactive_point_selector_batch(img, predictor, filename):
    """
    Interactive point selection for SAM segmentation with REAL-TIME preview.
    Modified for batch processing with filename display.
    """
    
    class PointSelector:
        def __init__(self, ax_img, ax_mask):
            self.positive_points = []
            self.negative_points = []
            self.ax_img = ax_img
            self.ax_mask = ax_mask
            self.point_markers = []
            self.mask_display = None
            self.current_mask = None
            
        def update_segmentation(self):
            """Update segmentation in real-time"""
            # Clear previous mask
            if self.mask_display is not None:
                self.mask_display.remove()
                self.mask_display = None
            
            # If no points, return
            if len(self.positive_points) == 0 and len(self.negative_points) == 0:
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.ax_mask.set_title("Máscara (agrega puntos para ver)")
                self.ax_mask.axis('off')
                fig.canvas.draw()
                return
            
            # Prepare points and labels
            input_points = []
            input_labels = []
            
            for point in self.positive_points:
                input_points.append(point)
                input_labels.append(1)
            
            for point in self.negative_points:
                input_points.append(point)
                input_labels.append(0)
            
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            
            # Generate mask
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]
                self.current_mask = best_mask
                
                # Display mask on right subplot
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.mask_display = self.ax_mask.imshow(best_mask, alpha=0.6, cmap='Blues')
                
                # Show points on mask view too
                for point in self.positive_points:
                    self.ax_mask.plot(point[0], point[1], 'g*', markersize=15, markeredgewidth=2)
                for point in self.negative_points:
                    self.ax_mask.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
                
                score = scores[np.argmax(scores)]
                area = np.sum(best_mask)
                self.ax_mask.set_title(f"Segmentación | Score: {score:.3f} | Área: {area} px")
                self.ax_mask.axis('off')
                
            except Exception as e:
                print(f"⚠️ Error en segmentación: {e}")
            
            fig.canvas.draw()
            
        def onclick(self, event):
            if event.inaxes != self.ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            
            # Botón izquierdo (1) = Punto NEGATIVO (rojo)
            if event.button == 1:
                self.negative_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_markers.append(('neg', marker))
                print(f"❌ Punto NEGATIVO agregado: ({x:.0f}, {y:.0f})")
                
            # Botón derecho (3) = Punto POSITIVO (verde)
            elif event.button == 3:
                self.positive_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'g*', markersize=20, markeredgewidth=2)[0]
                self.point_markers.append(('pos', marker))
                print(f"✅ Punto POSITIVO agregado: ({x:.0f}, {y:.0f})")
            
            # Update title with counts
            self.ax_img.set_title(f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar | 's': skip")
            
            # Update segmentation in real-time
            self.update_segmentation()
            
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
                    
                    self.ax_img.set_title(f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar | 's': skip")
                    self.update_segmentation()
            
            # C = Clear all points
            elif event.key == 'c':
                for _, marker in self.point_markers:
                    marker.remove()
                self.point_markers.clear()
                self.positive_points.clear()
                self.negative_points.clear()
                print("🧹 Todos los puntos limpiados")
                self.ax_img.set_title(f"✅ Positivos: 0 | ❌ Negativos: 0 | 'z': deshacer | 'c': limpiar | 's': skip")
                self.update_segmentation()
            
            # S = Skip this image
            elif event.key == 's':
                print(f"⏭️ Saltando imagen: {filename}")
                plt.close(fig)
    
    # Create the selector object with 2 subplots
    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Image with points
    ax_img.imshow(img)
    ax_img.set_title(f"🎯 {filename} | Click derecho = POSITIVO | Click izquierdo = NEGATIVO")
    ax_img.axis('off')
    
    # Right: Real-time mask
    ax_mask.imshow(img)
    ax_mask.set_title("Segmentación (agrega puntos para ver)")
    ax_mask.axis('off')
    
    selector_obj = PointSelector(ax_img, ax_mask)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', selector_obj.onclick)
    fig.canvas.mpl_connect('key_press_event', selector_obj.onkey)
    
    # Instructions
    plt.figtext(0.5, 0.02, 
                "🟢 Click DERECHO: Punto positivo | 🔴 Click IZQUIERDO: Punto negativo | ⌨️ 'z': Deshacer | 'c': Limpiar | 's': Saltar | ENTER/ESC: Siguiente", 
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="blue", linewidth=2))
    
    plt.tight_layout()
    plt.show()
    
    return selector_obj.positive_points, selector_obj.negative_points, selector_obj.current_mask


def refine_medical_mask(mask):
    """Clean up the segmentation mask for medical images"""
    if mask is None:
        return None
        
    # Remove small objects
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    
    # Fill holes
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Smooth with morphological operations
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth


def process_dicom_folder(dicom_folder_path, checkpoint_path, output_folder="batch_segmentation_results"):
    """
    Process all DICOM files in a folder with SAM segmentation
    """
    
    # Setup device and model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    # Load SAM model
    print("⏳ Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    print("✅ SAM model loaded!")
    
    # Find all DICOM files
    dicom_extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']
    dicom_files = []
    
    for ext in dicom_extensions:
        dicom_files.extend(glob.glob(os.path.join(dicom_folder_path, ext)))
    
    if not dicom_files:
        print(f"⚠️ No DICOM files found in {dicom_folder_path}")
        return
    
    print(f"📁 Found {len(dicom_files)} DICOM files")
    dicom_files.sort()  # Sort for consistent processing order
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize results tracking
    results_summary = {
        "processing_date": datetime.now().isoformat(),
        "total_files": len(dicom_files),
        "processed_files": [],
        "skipped_files": [],
        "failed_files": []
    }
    
    # Process each DICOM file
    for i, dicom_path in enumerate(dicom_files):
        filename = os.path.basename(dicom_path)
        print(f"\n{'='*60}")
        print(f"📄 Processing {i+1}/{len(dicom_files)}: {filename}")
        print(f"{'='*60}")
        
        # Load DICOM
        img, dicom_data = load_dicom_as_image(dicom_path)
        
        if img is None:
            print(f"❌ Failed to load {filename}")
            results_summary["failed_files"].append({
                "filename": filename,
                "error": "Failed to load DICOM"
            })
            continue
        
        # Enhance contrast for medical images
        img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        predictor.set_image(img_enhanced)
        
        print(f"📏 Image dimensions: {img.shape}")
        
        # Interactive point selection
        print("🎯 Starting point selection...")
        print("   - Click DERECHO: Puntos POSITIVOS (objeto de interés)")
        print("   - Click IZQUIERDO: Puntos NEGATIVOS (para omitir)")
        print("   - Tecla 's': SALTAR esta imagen")
        print("   - Tecla 'z': Deshacer último punto")
        print("   - Tecla 'c': Limpiar todos los puntos")
        
        positive_points, negative_points, current_mask = interactive_point_selector_batch(img, predictor, filename)
        
        # Check if user skipped or didn't add points
        if len(positive_points) == 0 and len(negative_points) == 0:
            print(f"⏭️ Skipping {filename} - no points selected")
            results_summary["skipped_files"].append(filename)
            continue
        
        # Generate final segmentation if needed
        if current_mask is None:
            # Prepare points and labels for SAM
            input_points = []
            input_labels = []
            
            for point in positive_points:
                input_points.append(point)
                input_labels.append(1)
            
            for point in negative_points:
                input_points.append(point)
                input_labels.append(0)
            
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            
            # Generate masks
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)]
            best_score = scores[np.argmax(scores)]
        else:
            best_mask = current_mask
            best_score = 0.0  # Score not available from interactive session
        
        # Refine mask
        refined_mask = refine_medical_mask(best_mask)
        
        if refined_mask is None:
            print(f"❌ Failed to process mask for {filename}")
            results_summary["failed_files"].append({
                "filename": filename,
                "error": "Failed to process mask"
            })
            continue
        
        # Create output filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        # Save results
        # 1. Save mask as PNG
        mask_binary = (refined_mask * 255).astype(np.uint8)
        mask_path = os.path.join(output_folder, f"{base_filename}_mask.png")
        Image.fromarray(mask_binary).save(mask_path)
        
        # 2. Save overlay image
        overlay = img.copy()
        overlay[refined_mask > 0] = overlay[refined_mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4
        overlay_path = os.path.join(output_folder, f"{base_filename}_overlay.png")
        Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)
        
        # 3. Save original image as PNG
        original_path = os.path.join(output_folder, f"{base_filename}_original.png")
        Image.fromarray(img).save(original_path)
        
        # 4. Save segmentation info
        segmentation_info = {
            "filename": filename,
            "processing_date": datetime.now().isoformat(),
            "positive_points": positive_points,
            "negative_points": negative_points,
            "mask_area_pixels": int(np.sum(refined_mask)),
            "image_dimensions": img.shape[:2],
            "score": float(best_score),
            "files_generated": {
                "mask": f"{base_filename}_mask.png",
                "overlay": f"{base_filename}_overlay.png",
                "original": f"{base_filename}_original.png"
            }
        }
        
        info_path = os.path.join(output_folder, f"{base_filename}_info.json")
        with open(info_path, 'w') as f:
            json.dump(segmentation_info, f, indent=2)
        
        # Add to results summary
        results_summary["processed_files"].append(segmentation_info)
        
        print(f"✅ Processed {filename}")
        print(f"   📊 Mask area: {np.sum(refined_mask)} pixels")
        print(f"   💾 Files saved:")
        print(f"      • {mask_path}")
        print(f"      • {overlay_path}")
        print(f"      • {original_path}")
        print(f"      • {info_path}")
    
    # Save final summary
    summary_path = os.path.join(output_folder, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"📁 Total files found: {results_summary['total_files']}")
    print(f"✅ Successfully processed: {len(results_summary['processed_files'])}")
    print(f"⏭️ Skipped: {len(results_summary['skipped_files'])}")
    print(f"❌ Failed: {len(results_summary['failed_files'])}")
    print(f"📂 Results saved in: {output_folder}")
    print(f"📋 Summary saved as: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    DICOM_FOLDER = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/Data/HumeroData"
    CHECKPOINT_PATH = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/Checkpoints/sam_vit_h_4b8939.pth"
    OUTPUT_FOLDER = "batch_segmentation_results"
    
    print("🔬 BATCH DICOM SEGMENTATION WITH SAM")
    print("="*60)
    print(f"📁 DICOM folder: {DICOM_FOLDER}")
    print(f"🤖 Model checkpoint: {CHECKPOINT_PATH}")
    print(f"💾 Output folder: {OUTPUT_FOLDER}")
    print("="*60)
    
    # Check if paths exist
    if not os.path.exists(DICOM_FOLDER):
        print(f"❌ DICOM folder not found: {DICOM_FOLDER}")
        sys.exit(1)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    # Check if pydicom is installed
    try:
        import pydicom
    except ImportError:
        print("❌ pydicom not installed. Install with:")
        print("   pip install pydicom")
        sys.exit(1)
    
    # Start processing
    process_dicom_folder(DICOM_FOLDER, CHECKPOINT_PATH, OUTPUT_FOLDER)



