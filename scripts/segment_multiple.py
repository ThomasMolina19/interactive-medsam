import sys
import os
sys.path.append('path/to/segment-anything')
# Agregar path al módulo src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage
from skimage import morphology
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from pathlib import Path

# Importar mejoras de Fase 1
from src.preprocessing import enhance_bone_contrast, detect_bone_edges


device = "mps" if torch.backends.mps.is_available() else "cpu"

# ============ CONFIGURACIÓN ============
# Here goes the path to your MedSAM model checkpoint
ckpt = "checkpoints/medsam_vit_b.pth"

# Here goes the input folder with medical images
input_folder = "/dicom_pngs"

# Carpeta de salida para los resultados
output_folder = "/medsam_output"

# Coordenadas fijas de la caja [x_min, y_min, x_max, y_max]
# Ajusta estos valores según tu región de interés
FIXED_BOX = np.array([100, 100, 400, 400])

# =======================================

# Load model
sam = sam_model_registry["vit_b"]()
checkpoint = torch.load(ckpt, map_location=device)
sam.load_state_dict(checkpoint)
sam = sam.to(device)

predictor = SamPredictor(sam)

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

def interactive_box_selector(img):
    """Enhanced interactive bounding box selection"""
    
    class BoxSelector:
        def __init__(self):
            self.box = None
            self.selected = False
            
        def onselect(self, eclick, erelease):
            """Callback for rectangle selection"""
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            # Ensure coordinates are in correct order
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            self.box = np.array([x_min, y_min, x_max, y_max])
            self.selected = True
            
            # Update title with coordinates
            ax.set_title(f"Selected: [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}] - Close window when ready")
            fig.canvas.draw()
    
    # Create the selector object
    selector_obj = BoxSelector()
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title("🎯 Drag to select region of interest")
    ax.axis('off')
    
    # Create rectangle selector with optimal settings
    rectangle_selector = RectangleSelector(
        ax, 
        selector_obj.onselect,
        useblit=True,           # Faster drawing
        button=[1],             # Left mouse button only
        minspanx=10, minspany=10,  # Minimum size
        spancoords='pixels',
        interactive=True,       # Enable resizing
        drag_from_anywhere=True # Drag from inside rectangle
    )
    
    # Beautiful styling
    rectangle_selector.rectprops = dict(
        facecolor='cyan', 
        edgecolor='red',
        alpha=0.4, 
        fill=True,
        linewidth=3
    )
    
    # Instructions
    plt.figtext(0.5, 0.02, 
                "💡 Drag to create box • Drag edges to resize • Close window when done", 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()
    
    # Return selected box or intelligent default
    if selector_obj.box is not None:
        return selector_obj.box.astype(int)
    else:
        # Smart default: center region
        H, W = img.shape[:2]
        margin = min(W, H) // 8
        return np.array([margin, margin, W-margin, H-margin])

# Post-process mask
def refine_medical_mask(mask):
    """Clean up the segmentation mask for medical images"""
    # Remove small objects
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    
    # Fill holes
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Smooth with morphological operations
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth

def process_single_image(image_path, box, save_visualization=True):
    """Procesa una sola imagen con la caja fija"""
    
    # Cargar imagen
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # ============ FASE 1: Preprocesamiento Mejorado ============
    # Método antiguo: img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    # Método nuevo - CLAHE (Fase 1)
    img_enhanced = enhance_bone_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Configurar predictor
    predictor.set_image(img_enhanced)
    
    # Generar máscaras
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )
    
    # Seleccionar mejor máscara
    best_mask = masks[np.argmax(scores)]
    
    # Refinar máscara
    refined_mask = refine_medical_mask(best_mask)
    
    # Guardar máscara
    image_name = Path(image_path).stem
    mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
    refined_mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8))
    refined_mask_pil.save(mask_path)
    
    # Guardar visualización si se solicita
    if save_visualization:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Original results
        axes[0,0].imshow(img)
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        
        axes[0,1].imshow(img)
        axes[0,1].imshow(best_mask, alpha=0.5, cmap='Reds')
        rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                        linewidth=3, edgecolor='green', facecolor='none')
        axes[0,1].add_patch(rect)
        axes[0,1].set_title("Raw SAM Output with Fixed Box")
        axes[0,1].axis('off')
        
        axes[0,2].imshow(best_mask, cmap='gray')
        axes[0,2].set_title("Raw Mask")
        axes[0,2].axis('off')
        
        # Row 2: Enhanced results
        axes[1,0].imshow(img_enhanced)
        axes[1,0].set_title("Enhanced Image")
        axes[1,0].axis('off')
        
        axes[1,1].imshow(img)
        axes[1,1].imshow(refined_mask, alpha=0.5, cmap='Blues')
        rect2 = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                         linewidth=3, edgecolor='yellow', facecolor='none')
        axes[1,1].add_patch(rect2)
        axes[1,1].set_title("Refined Segmentation")
        axes[1,1].axis('off')
        
        axes[1,2].imshow(refined_mask, cmap='gray')
        axes[1,2].set_title("Refined Mask")
        axes[1,2].axis('off')
        
        plt.suptitle(f"Análisis: {image_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = os.path.join(output_folder, f"{image_name}_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Retornar estadísticas
    return {
        'image_name': image_name,
        'mask_area': np.sum(refined_mask),
        'best_score': scores[np.argmax(scores)],
        'total_masks': len(masks),
        'mask_path': mask_path
    }


def process_folder(input_folder, box, image_extensions=['.png', '.jpg', '.jpeg']):
    """Procesa todas las imágenes de una carpeta con la caja fija"""
    
    print(f"\n{'='*60}")
    print(f"🚀 PROCESAMIENTO EN LOTE - SAM Medical Image Segmentation")
    print(f"{'='*60}")
    print(f"📁 Carpeta de entrada: {input_folder}")
    print(f"💾 Carpeta de salida: {output_folder}")
    print(f"📦 Caja fija: {box}")
    print(f"🖥️  Dispositivo: {device}")
    print(f"{'='*60}\n")
    
    # Obtener lista de imágenes
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f"*{ext}"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print("❌ No se encontraron imágenes en la carpeta especificada")
        return
    
    print(f"📊 Total de imágenes encontradas: {len(image_files)}\n")
    
    # Procesar cada imagen
    results = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Procesando: {image_path.name}...", end=' ')
        
        try:
            result = process_single_image(str(image_path), box, save_visualization=True)
            results.append(result)
            print(f"✅ Completado (Área: {result['mask_area']} px, Score: {result['best_score']:.4f})")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"📈 RESUMEN DEL PROCESAMIENTO")
    print(f"{'='*60}")
    print(f"✅ Imágenes procesadas exitosamente: {len(results)}/{len(image_files)}")
    
    if results:
        avg_area = np.mean([r['mask_area'] for r in results])
        avg_score = np.mean([r['best_score'] for r in results])
        print(f"📏 Área promedio de máscaras: {avg_area:.0f} píxeles")
        print(f"⭐ Score promedio: {avg_score:.4f}")
        print(f"\n💾 Resultados guardados en: {output_folder}")
    
    print(f"{'='*60}\n")
    
    return results


# ============ EJECUCIÓN PRINCIPAL ============
if __name__ == "__main__":
    
    # Opción 1: Usar caja fija predefinida
    #print("🎯 Modo de procesamiento: Caja fija")
    #results = process_folder(input_folder, FIXED_BOX)
    
    # Opción 2: Si deseas seleccionar la caja interactivamente primero
    # (descomenta las siguientes líneas y comenta la línea anterior)
    print("🎯 Selecciona la región de interés en la primera imagen...")
    first_image = sorted(Path(input_folder).glob("*.png"))[0]
    img_sample = np.array(Image.open(first_image).convert("RGB"))
    selected_box = interactive_box_selector(img_sample)
    print(f"📦 Caja seleccionada: {selected_box}")
    results = process_folder(input_folder, selected_box)
