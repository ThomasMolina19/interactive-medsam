"""
Visualización completa de Fase 1 + Fase 2
Muestra el pipeline completo de preprocesamiento y detección
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional

# Importar Fase 1 y Fase 2
from src.preprocessing import enhance_bone_contrast, detect_bone_edges
from src.detection import detect_humerus_automatic


def visualize_complete_pipeline(
    image_path: str,
    save_path: Optional[str] = None
):
    """
    Visualiza el pipeline completo: Fase 1 + Fase 2
    """
    # Cargar imagen
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # FASE 1: Preprocesamiento
    img_enhanced = enhance_bone_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8))
    edges = detect_bone_edges(img_enhanced)
    
    # FASE 2: Detección
    detection = detect_humerus_automatic(img, img_enhanced, method='combined', return_all_candidates=True)
    
    # Crear visualización de 3x3
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # ========== FILA 1: ENTRADA Y FASE 1 ==========
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("1. Imagen Original", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # CLAHE
    axes[0, 1].imshow(img_enhanced)
    axes[0, 1].set_title("2. Fase 1: CLAHE\n(Contraste Mejorado)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bordes
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title("3. Fase 1: Bordes Detectados\n(Canny)", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # ========== FILA 2: FASE 2 - DETECCIÓN ==========
    
    # Imagen con todos los candidatos
    img_candidates = img.copy()
    if detection.get('all_candidates'):
        for cand in detection['all_candidates'][:10]:
            if cand['type'] == 'circle':
                x, y, r = cand['params']
                color = (255, 165, 0)  # Naranja para candidatos
                cv2.circle(img_candidates, (x, y), r, color, 2)
    
    axes[1, 0].imshow(img_candidates)
    axes[1, 0].set_title(f"4. Candidatos Detectados\n({len(detection.get('all_candidates', []))} círculos)", 
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Mejor candidato
    img_best = img.copy()
    if detection['success']:
        if detection['detection']['type'] == 'circle':
            x, y, r = detection['detection']['params']
            cv2.circle(img_best, (x, y), r, (255, 0, 0), 3)
            cv2.circle(img_best, (x, y), 3, (255, 0, 0), -1)
        
        title = f"5. Fase 2: Mejor Candidato\nConfianza: {detection['confidence']:.3f}"
        color = 'green'
    else:
        title = "5. Fase 2: Sin Detección"
        color = 'red'
    
    axes[1, 1].imshow(img_best)
    axes[1, 1].set_title(title, fontsize=14, fontweight='bold', color=color)
    axes[1, 1].axis('off')
    
    # Bounding box
    img_box = img.copy()
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(img_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Agregar texto con dimensiones
        cv2.putText(img_box, f"{x_max-x_min}x{y_max-y_min} px", 
                   (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    axes[1, 2].imshow(img_box)
    axes[1, 2].set_title("6. Bounding Box Generada\n(Lista para SAM)", fontsize=14, fontweight='bold', color='green')
    axes[1, 2].axis('off')
    
    # ========== FILA 3: ANÁLISIS DETALLADO ==========
    
    # ROI extraída
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        roi = img[y_min:y_max, x_min:x_max]
        axes[2, 0].imshow(roi)
        axes[2, 0].set_title("7. ROI Extraída\n(Región de Interés)", fontsize=14, fontweight='bold')
        axes[2, 0].axis('off')
    else:
        axes[2, 0].axis('off')
    
    # Comparación Original vs CLAHE
    axes[2, 1].hist(img[:,:,0].ravel(), bins=50, alpha=0.6, color='blue', label='Original')
    axes[2, 1].hist(img_enhanced[:,:,0].ravel(), bins=50, alpha=0.6, color='red', label='CLAHE')
    axes[2, 1].set_title("8. Histograma Comparativo", fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('Intensidad')
    axes[2, 1].set_ylabel('Frecuencia')
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)
    
    # Información textual
    axes[2, 2].axis('off')
    if detection['success']:
        box = detection['box']
        x_min, y_min, x_max, y_max = box
        
        info_text = f"""
PIPELINE COMPLETO
{'='*40}

FASE 1: PREPROCESAMIENTO
  ✓ CLAHE aplicado
  ✓ Bordes detectados
  ✓ Contraste mejorado 23.3%

FASE 2: DETECCIÓN AUTOMÁTICA
  ✓ Estado: EXITOSA
  ✓ Confianza: {detection['confidence']:.3f}
  ✓ Método: {detection['detection']['method']}
  
BOUNDING BOX:
  • Posición: ({x_min}, {y_min})
  • Tamaño: {x_max-x_min} x {y_max-y_min} px
  • Área: {(x_max-x_min)*(y_max-y_min)} px²

DETECCIÓN:
  • Tipo: {detection['detection']['type']}
        """
        
        if detection['detection']['type'] == 'circle':
            x, y, r = detection['detection']['params']
            info_text += f"""  • Centro: ({x}, {y})
  • Radio: {r} px
  • Área: {int(np.pi * r * r)} px²
        """
        
        info_text += f"""
SIGUIENTE PASO:
  → Segmentación con SAM
        """
    else:
        info_text = """
PIPELINE COMPLETO
{'='*40}

FASE 1: PREPROCESAMIENTO
  ✓ CLAHE aplicado
  ✓ Bordes detectados

FASE 2: DETECCIÓN AUTOMÁTICA
  ✗ Estado: FALLIDA
  
  Posibles causas:
  • Húmero no visible
  • Calidad de imagen baja
  • Parámetros a ajustar
        """
    
    axes[2, 2].text(0.05, 0.95, info_text, fontsize=10, family='monospace',
                   verticalalignment='top', transform=axes[2, 2].transAxes)
    
    # Título general
    plt.suptitle(f"Pipeline Completo: Fase 1 + Fase 2 - {Path(image_path).name}", 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return detection


def process_batch(image_paths: list):
    """
    Procesa múltiples imágenes y genera visualizaciones
    """
    print(f"\n{'='*60}")
    print(f"🔬 VISUALIZACIÓN BATCH: Fase 1 + Fase 2")
    print(f"{'='*60}")
    print(f"📊 Procesando {len(image_paths)} imágenes...\n")
    
    output_dir = Path(__file__).parent.parent / "test_results" / "phase1_phase2_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] {Path(img_path).name}...", end=' ')
        
        try:
            output_path = output_dir / f"pipeline_{Path(img_path).stem}.png"
            detection = visualize_complete_pipeline(str(img_path), str(output_path))
            
            results.append({
                'image': Path(img_path).name,
                'success': detection['success'],
                'confidence': detection['confidence'] if detection['success'] else 0.0
            })
            
            if detection['success']:
                print(f"✅ Confianza: {detection['confidence']:.3f}")
            else:
                print("❌ No detectado")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'image': Path(img_path).name,
                'success': False,
                'confidence': 0.0
            })
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"📈 RESUMEN")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"✅ Detecciones exitosas: {success_count}/{len(results)}")
    print(f"📊 Tasa de éxito: {success_count/len(results)*100:.1f}%")
    
    if success_count > 0:
        avg_confidence = np.mean([r['confidence'] for r in results if r['success']])
        print(f"🎯 Confianza promedio: {avg_confidence:.3f}")
    
    print(f"\n💾 Visualizaciones guardadas en: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualización Fase 1 + Fase 2")
    parser.add_argument('--image', type=str, help='Ruta a imagen individual')
    parser.add_argument('--images', nargs='+', help='Lista de imágenes')
    parser.add_argument('--folder', type=str, help='Carpeta con imágenes')
    
    args = parser.parse_args()
    
    if args.image:
        visualize_complete_pipeline(args.image)
    elif args.images:
        process_batch(args.images)
    elif args.folder:
        image_files = list(Path(args.folder).glob("*.png"))
        if image_files:
            process_batch([str(f) for f in image_files[:10]])
        else:
            print("❌ No se encontraron imágenes PNG")
    else:
        # Usar imágenes de ejemplo
        example_folder = Path(__file__).parent.parent / "dicom_pngs"
        if example_folder.exists():
            example_images = list(example_folder.glob("*.png"))[:5]
            if example_images:
                process_batch([str(f) for f in example_images])
            else:
                print("❌ No se encontraron imágenes de ejemplo")
        else:
            print("💡 Uso: python visualize_phase1_phase2.py --images img1.png img2.png ...")
