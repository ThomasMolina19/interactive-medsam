"""
Script de prueba para Fase 1: Preprocesamiento Avanzado
Prueba las mejoras de CLAHE, detección de bordes y normalización adaptativa
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Importar módulo de preprocesamiento
from src.preprocessing import enhance_bone_contrast, detect_bone_edges, normalize_adaptive
from src.preprocessing.enhance import preprocess_medical_image


def test_single_image(image_path: str, save_results: bool = True):
    """
    Prueba el preprocesamiento en una sola imagen
    
    Args:
        image_path: Ruta a la imagen de prueba
        save_results: Si True, guarda visualización
    """
    print(f"\n{'='*60}")
    print(f"🧪 Prueba de Preprocesamiento - Fase 1")
    print(f"{'='*60}")
    print(f"📁 Imagen: {image_path}")
    
    # Cargar imagen
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"✅ Imagen cargada: {img.shape}")
    
    # Aplicar preprocesamiento completo
    print("\n🔄 Aplicando preprocesamiento...")
    results = preprocess_medical_image(
        img,
        use_clahe=True,
        use_normalization=True,
        detect_edges=True,
        return_all_steps=True
    )
    
    # Extraer resultados
    img_enhanced = results['enhanced']
    edges = results['edges']
    steps = results['steps']
    
    print("✅ Preprocesamiento completado")
    
    # Crear visualización comparativa
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Fila 1: Proceso paso a paso
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("1. Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    if 'normalized' in steps:
        axes[0, 1].imshow(steps['normalized'], cmap='gray')
        axes[0, 1].set_title("2. Normalización Adaptativa", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_enhanced)
    axes[0, 2].set_title("3. CLAHE Aplicado", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(edges, cmap='gray')
    axes[0, 3].set_title("4. Bordes Detectados", fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Fila 2: Comparaciones
    # Original vs CLAHE
    axes[1, 0].imshow(img)
    axes[1, 0].set_title("Original", fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_enhanced)
    axes[1, 1].set_title("Con CLAHE (Mejor Contraste)", fontsize=11)
    axes[1, 1].axis('off')
    
    # Overlay de bordes sobre imagen mejorada
    img_with_edges = img_enhanced.copy()
    if len(img_with_edges.shape) == 2:
        img_with_edges = cv2.cvtColor(img_with_edges, cv2.COLOR_GRAY2RGB)
    # Resaltar bordes en rojo
    img_with_edges[edges > 0] = [255, 0, 0]
    
    axes[1, 2].imshow(img_with_edges)
    axes[1, 2].set_title("Bordes Superpuestos", fontsize=11)
    axes[1, 2].axis('off')
    
    # Histograma comparativo
    axes[1, 3].hist(img[:,:,0].ravel(), bins=50, alpha=0.5, label='Original', color='blue')
    if len(img_enhanced.shape) == 3:
        axes[1, 3].hist(img_enhanced[:,:,0].ravel(), bins=50, alpha=0.5, label='CLAHE', color='red')
    else:
        axes[1, 3].hist(img_enhanced.ravel(), bins=50, alpha=0.5, label='CLAHE', color='red')
    axes[1, 3].set_title("Histograma de Intensidades", fontsize=11)
    axes[1, 3].legend()
    axes[1, 3].set_xlabel('Intensidad')
    axes[1, 3].set_ylabel('Frecuencia')
    
    plt.suptitle(f"Fase 1: Preprocesamiento Avanzado - {Path(image_path).name}", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Guardar resultados
    if save_results:
        output_dir = Path(__file__).parent.parent / "test_results" / "phase1"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"phase1_{Path(image_path).stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualización guardada: {output_path}")
        
        # Guardar imágenes individuales
        Image.fromarray(img_enhanced).save(output_dir / f"enhanced_{Path(image_path).stem}.png")
        Image.fromarray(edges).save(output_dir / f"edges_{Path(image_path).stem}.png")
        print(f"💾 Imágenes procesadas guardadas en: {output_dir}")
    
    plt.show()
    
    print(f"\n{'='*60}")
    print("✅ Prueba completada exitosamente")
    print(f"{'='*60}\n")
    
    return results


def test_batch_images(input_folder: str, max_images: int = 5):
    """
    Prueba el preprocesamiento en múltiples imágenes
    
    Args:
        input_folder: Carpeta con imágenes de prueba
        max_images: Número máximo de imágenes a procesar
    """
    print(f"\n{'='*60}")
    print(f"🧪 Prueba Batch - Fase 1 Preprocesamiento")
    print(f"{'='*60}")
    
    # Obtener lista de imágenes
    image_files = list(Path(input_folder).glob("*.png"))[:max_images]
    
    if not image_files:
        print("❌ No se encontraron imágenes PNG en la carpeta")
        return
    
    print(f"📊 Procesando {len(image_files)} imágenes...\n")
    
    results_summary = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Procesando: {img_path.name}...", end=' ')
        
        try:
            # Cargar y procesar
            img = np.array(Image.open(img_path).convert("RGB"))
            result = preprocess_medical_image(img, use_clahe=True, detect_edges=True)
            
            # Calcular métricas
            edge_density = np.sum(result['edges'] > 0) / result['edges'].size
            
            results_summary.append({
                'image': img_path.name,
                'success': True,
                'edge_density': edge_density
            })
            
            print(f"✅ (Densidad bordes: {edge_density:.4f})")
            
        except Exception as e:
            results_summary.append({
                'image': img_path.name,
                'success': False,
                'error': str(e)
            })
            print(f"❌ Error: {str(e)}")
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results_summary if r['success'])
    print(f"✅ Procesadas exitosamente: {success_count}/{len(image_files)}")
    
    if success_count > 0:
        avg_edge_density = np.mean([r['edge_density'] for r in results_summary if r['success']])
        print(f"📈 Densidad promedio de bordes: {avg_edge_density:.4f}")
    
    print(f"{'='*60}\n")
    
    return results_summary


def compare_methods(image_path: str):
    """
    Compara el método antiguo (convertScaleAbs) vs nuevo (CLAHE)
    
    Args:
        image_path: Ruta a imagen de prueba
    """
    print(f"\n{'='*60}")
    print(f"⚖️  Comparación: Método Antiguo vs Fase 1")
    print(f"{'='*60}")
    
    # Cargar imagen
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # Método antiguo (del script original)
    img_old = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    # Método nuevo (Fase 1)
    img_new = enhance_bone_contrast(img)
    edges_new = detect_bone_edges(img_new)
    
    # Visualización comparativa
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_old)
    axes[0, 1].set_title("Método Antiguo\n(convertScaleAbs)", fontsize=14, fontweight='bold', color='red')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_new)
    axes[0, 2].set_title("Método Nuevo\n(CLAHE - Fase 1)", fontsize=14, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Histogramas
    axes[1, 0].hist(img[:,:,0].ravel(), bins=50, alpha=0.7, color='gray')
    axes[1, 0].set_title("Histograma Original")
    axes[1, 0].set_xlabel('Intensidad')
    
    axes[1, 1].hist(img_old[:,:,0].ravel(), bins=50, alpha=0.7, color='red')
    axes[1, 1].set_title("Histograma Método Antiguo")
    axes[1, 1].set_xlabel('Intensidad')
    
    axes[1, 2].imshow(edges_new, cmap='gray')
    axes[1, 2].set_title("Bordes Detectados\n(Solo Fase 1)")
    axes[1, 2].axis('off')
    
    plt.suptitle("Comparación de Métodos de Preprocesamiento", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ Comparación completada")
    print(f"{'='*60}\n")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prueba de Fase 1: Preprocesamiento")
    parser.add_argument('--image', type=str, help='Ruta a imagen individual')
    parser.add_argument('--folder', type=str, help='Carpeta con imágenes para batch')
    parser.add_argument('--compare', action='store_true', help='Comparar con método antiguo')
    parser.add_argument('--max-images', type=int, default=5, help='Máximo de imágenes en batch')
    
    args = parser.parse_args()
    
    # Si no se pasan argumentos, usar imagen de ejemplo
    if not args.image and not args.folder:
        # Buscar imagen de ejemplo en dicom_pngs
        example_folder = Path(__file__).parent.parent / "dicom_pngs"
        if example_folder.exists():
            example_images = list(example_folder.glob("*.png"))
            if example_images:
                args.image = str(example_images[0])
                print(f"ℹ️  Usando imagen de ejemplo: {args.image}\n")
            else:
                print("❌ No se encontraron imágenes de ejemplo")
                print("💡 Uso: python test_phase1_preprocessing.py --image <ruta_imagen>")
                sys.exit(1)
        else:
            print("❌ No se encontró carpeta dicom_pngs")
            print("💡 Uso: python test_phase1_preprocessing.py --image <ruta_imagen>")
            sys.exit(1)
    
    # Ejecutar pruebas según argumentos
    if args.image:
        if args.compare:
            compare_methods(args.image)
        else:
            test_single_image(args.image)
    
    if args.folder:
        test_batch_images(args.folder, max_images=args.max_images)
