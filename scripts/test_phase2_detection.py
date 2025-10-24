"""
Script de prueba para Fase 2: Detección Automática del Húmero
Prueba detección por circularidad, intensidad y scoring
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
from src.preprocessing import enhance_bone_contrast
from src.detection import detect_humerus_automatic


def visualize_detection(
    img: np.ndarray,
    img_enhanced: np.ndarray,
    detection_result: dict,
    save_path: Optional[str] = None
):
    """
    Visualiza el resultado de la detección automática
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Fila 1
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Imagen Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_enhanced)
    axes[0, 1].set_title("Fase 1: CLAHE", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Mostrar detección
    img_detection = img.copy()
    if detection_result['success']:
        box = detection_result['box']
        x_min, y_min, x_max, y_max = box
        
        # Dibujar bounding box
        cv2.rectangle(img_detection, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Dibujar círculo si es detección circular
        if detection_result['detection']['type'] == 'circle':
            params = detection_result['detection']['params']
            if len(params) == 4:
                x, y, r, _ = params
            else:
                x, y, r = params
            cv2.circle(img_detection, (x, y), r, (255, 0, 0), 2)
            cv2.circle(img_detection, (x, y), 3, (255, 0, 0), -1)
        
        title = f"Fase 2: Detección Automática\nConfianza: {detection_result['confidence']:.2f}"
        color = 'green'
    else:
        title = "Fase 2: Detección Fallida"
        color = 'red'
    
    axes[0, 2].imshow(img_detection)
    axes[0, 2].set_title(title, fontsize=12, fontweight='bold', color=color)
    axes[0, 2].axis('off')
    
    # Fila 2: Detalles
    if detection_result['success']:
        box = detection_result['box']
        x_min, y_min, x_max, y_max = box
        
        # Zoom en región detectada
        roi = img[y_min:y_max, x_min:x_max]
        axes[1, 0].imshow(roi)
        axes[1, 0].set_title("ROI Detectada", fontsize=12)
        axes[1, 0].axis('off')
        
        # Mostrar todos los candidatos si están disponibles
        if detection_result.get('all_candidates'):
            img_candidates = img.copy()
            for i, cand in enumerate(detection_result['all_candidates'][:5]):
                if cand['type'] == 'circle':
                    x, y, r = cand['params']
                    color = (0, 255, 0) if cand == detection_result['detection'] else (255, 165, 0)
                    cv2.circle(img_candidates, (x, y), r, color, 2)
            
            axes[1, 1].imshow(img_candidates)
            axes[1, 1].set_title(f"Candidatos ({len(detection_result['all_candidates'])})", fontsize=12)
            axes[1, 1].axis('off')
        else:
            axes[1, 1].axis('off')
        
        # Información textual
        axes[1, 2].axis('off')
        info_text = f"""
        📊 INFORMACIÓN DE DETECCIÓN
        
        ✅ Estado: Exitosa
        🎯 Confianza: {detection_result['confidence']:.3f}
        📦 Método: {detection_result['detection']['method']}
        
        📐 Bounding Box:
           x: {x_min} → {x_max}
           y: {y_min} → {y_max}
           Ancho: {x_max - x_min} px
           Alto: {y_max - y_min} px
        
        🔵 Detección:
           Tipo: {detection_result['detection']['type']}
        """
        
        if detection_result['detection']['type'] == 'circle':
            params = detection_result['detection']['params']
            if len(params) == 4:
                x, y, r, _ = params
            else:
                x, y, r = params
            info_text += f"""   Centro: ({x}, {y})
           Radio: {r} px
           Área: {int(np.pi * r * r)} px²
        """
        
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                       verticalalignment='center')
    else:
        for ax in axes[1, :]:
            ax.axis('off')
        axes[1, 1].text(0.5, 0.5, "❌ No se detectó el húmero\n\nIntenta ajustar parámetros",
                       ha='center', va='center', fontsize=14, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def test_single_image(image_path: str, save_results: bool = True):
    """
    Prueba detección automática en una imagen
    """
    print(f"\n{'='*60}")
    print(f"🧪 Prueba Fase 2: Detección Automática")
    print(f"{'='*60}")
    print(f"📁 Imagen: {image_path}")
    
    # Cargar imagen
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"✅ Imagen cargada: {img.shape}")
    
    # Fase 1: Preprocesamiento
    print("\n🔄 Fase 1: Aplicando CLAHE...")
    img_enhanced = enhance_bone_contrast(img)
    print("✅ Preprocesamiento completado")
    
    # Fase 2: Detección automática
    print("\n🔍 Fase 2: Detectando húmero...")
    detection_result = detect_humerus_automatic(
        img, 
        img_enhanced, 
        method='combined',
        return_all_candidates=True
    )
    
    if detection_result['success']:
        print(f"✅ Húmero detectado!")
        print(f"   Confianza: {detection_result['confidence']:.3f}")
        print(f"   Método: {detection_result['detection']['method']}")
        print(f"   Box: {detection_result['box']}")
    else:
        print("❌ No se pudo detectar el húmero")
    
    # Visualizar
    if save_results:
        output_dir = Path(__file__).parent.parent / "test_results" / "phase2"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detection_{Path(image_path).stem}.png"
        visualize_detection(img, img_enhanced, detection_result, str(output_path))
        print(f"\n💾 Resultado guardado: {output_path}")
    else:
        visualize_detection(img, img_enhanced, detection_result)
    
    print(f"{'='*60}\n")
    
    return detection_result


def test_batch_images(image_paths: list):
    """
    Prueba detección en múltiples imágenes
    """
    print(f"\n{'='*60}")
    print(f"🧪 Prueba Batch - Fase 2: Detección Automática")
    print(f"{'='*60}")
    print(f"📊 Procesando {len(image_paths)} imágenes...\n")
    
    results = []
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] {Path(img_path).name}...", end=' ')
        
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            img_enhanced = enhance_bone_contrast(img)
            detection = detect_humerus_automatic(img, img_enhanced, method='combined')
            
            results.append({
                'image': Path(img_path).name,
                'success': detection['success'],
                'confidence': detection['confidence'] if detection['success'] else 0.0,
                'method': detection.get('detection', {}).get('method', 'none')
            })
            
            if detection['success']:
                print(f"✅ Confianza: {detection['confidence']:.3f}")
            else:
                print("❌ No detectado")
            
            # Guardar visualización
            output_dir = Path(__file__).parent.parent / "test_results" / "phase2"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"detection_{Path(img_path).stem}.png"
            visualize_detection(img, img_enhanced, detection, str(output_path))
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'image': Path(img_path).name,
                'success': False,
                'confidence': 0.0,
                'error': str(e)
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
        
        # Contar métodos
        methods = [r['method'] for r in results if r['success']]
        from collections import Counter
        method_counts = Counter(methods)
        print(f"\n📊 Métodos de detección:")
        for method, count in method_counts.items():
            print(f"   {method}: {count}")
    
    print(f"{'='*60}\n")
    
    # Crear gráfico resumen
    create_summary_plot(results)
    
    return results


def create_summary_plot(results: list):
    """
    Crea gráfico resumen de detecciones
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    images = [r['image'] for r in results]
    successes = [1 if r['success'] else 0 for r in results]
    confidences = [r['confidence'] if r['success'] else 0 for r in results]
    
    # Gráfico 1: Éxito/Fallo
    colors = ['green' if s else 'red' for s in successes]
    axes[0].bar(range(len(images)), successes, color=colors, alpha=0.7)
    axes[0].set_xticks(range(len(images)))
    axes[0].set_xticklabels([img[:6] for img in images], rotation=45)
    axes[0].set_ylabel('Detección Exitosa')
    axes[0].set_ylim([0, 1.2])
    axes[0].set_title('Estado de Detección', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Confianza
    axes[1].bar(range(len(images)), confidences, color='steelblue', alpha=0.7)
    axes[1].set_xticks(range(len(images)))
    axes[1].set_xticklabels([img[:6] for img in images], rotation=45)
    axes[1].set_ylabel('Confianza')
    axes[1].set_ylim([0, 1.0])
    axes[1].set_title('Confianza de Detección', fontweight='bold')
    axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Umbral')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "test_results" / "phase2"
    output_path = output_dir / "summary_detection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Resumen guardado: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    from typing import Optional
    
    parser = argparse.ArgumentParser(description="Prueba de Fase 2: Detección Automática")
    parser.add_argument('--image', type=str, help='Ruta a imagen individual')
    parser.add_argument('--folder', type=str, help='Carpeta con imágenes')
    parser.add_argument('--images', nargs='+', help='Lista de imágenes')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image)
    elif args.images:
        test_batch_images(args.images)
    elif args.folder:
        image_files = list(Path(args.folder).glob("*.png"))
        if image_files:
            test_batch_images([str(f) for f in image_files[:10]])
        else:
            print("❌ No se encontraron imágenes PNG")
    else:
        # Usar imágenes de ejemplo
        example_folder = Path(__file__).parent.parent / "dicom_pngs"
        if example_folder.exists():
            example_images = list(example_folder.glob("*.png"))[:5]
            if example_images:
                test_batch_images([str(f) for f in example_images])
            else:
                print("❌ No se encontraron imágenes de ejemplo")
        else:
            print("💡 Uso: python test_phase2_detection.py --images img1.png img2.png ...")
