"""
Script para comparar método antiguo vs Fase 1 en múltiples imágenes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Importar mejoras de Fase 1
from src.preprocessing import enhance_bone_contrast, detect_bone_edges


def compare_single_image(image_path: str, save_result: bool = True):
    """
    Compara método antiguo vs Fase 1 en una imagen
    """
    print(f"\n📊 Comparando: {Path(image_path).name}")
    
    # Cargar imagen
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # Método antiguo
    img_old = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    # Método nuevo (Fase 1)
    img_new = enhance_bone_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8))
    edges_new = detect_bone_edges(img_new)
    
    # Crear visualización
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fila 1
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_old)
    axes[0, 1].set_title("Método Antiguo\n(convertScaleAbs)", fontsize=12, color='red')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_new)
    axes[0, 2].set_title("Fase 1 - CLAHE", fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Fila 2
    axes[1, 0].hist(img[:,:,0].ravel(), bins=50, alpha=0.7, color='gray')
    axes[1, 0].set_title("Histograma Original")
    axes[1, 0].set_xlabel('Intensidad')
    axes[1, 0].set_ylabel('Frecuencia')
    
    axes[1, 1].hist(img_old[:,:,0].ravel(), bins=50, alpha=0.7, color='red')
    axes[1, 1].set_title("Histograma Antiguo")
    axes[1, 1].set_xlabel('Intensidad')
    
    axes[1, 2].imshow(edges_new, cmap='gray')
    axes[1, 2].set_title("Bordes Detectados\n(Fase 1)", fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Comparación: {Path(image_path).name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_result:
        output_dir = Path(__file__).parent.parent / "test_results" / "comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"compare_{Path(image_path).stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Guardado: {output_path.name}")
        plt.close()
    else:
        plt.show()
    
    # Calcular métricas
    edge_density = np.sum(edges_new > 0) / edges_new.size
    contrast_old = np.std(img_old)
    contrast_new = np.std(img_new)
    
    return {
        'image': Path(image_path).name,
        'edge_density': edge_density,
        'contrast_old': contrast_old,
        'contrast_new': contrast_new,
        'contrast_improvement': (contrast_new - contrast_old) / contrast_old * 100
    }


def compare_batch(image_paths: list):
    """
    Compara múltiples imágenes y genera reporte
    """
    print(f"\n{'='*60}")
    print(f"🔬 COMPARACIÓN BATCH: Antiguo vs Fase 1")
    print(f"{'='*60}")
    print(f"📊 Procesando {len(image_paths)} imágenes...\n")
    
    results = []
    
    for img_path in image_paths:
        try:
            result = compare_single_image(str(img_path), save_result=True)
            results.append(result)
            print(f"  ✅ {result['image']}: Densidad bordes={result['edge_density']:.4f}, "
                  f"Mejora contraste={result['contrast_improvement']:.1f}%")
        except Exception as e:
            print(f"  ❌ Error en {Path(img_path).name}: {str(e)}")
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"📈 RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    
    if results:
        avg_edge_density = np.mean([r['edge_density'] for r in results])
        avg_contrast_improvement = np.mean([r['contrast_improvement'] for r in results])
        
        print(f"✅ Imágenes procesadas: {len(results)}/{len(image_paths)}")
        print(f"📊 Densidad promedio de bordes: {avg_edge_density:.4f}")
        print(f"📈 Mejora promedio de contraste: {avg_contrast_improvement:.1f}%")
        
        if avg_contrast_improvement > 0:
            print(f"\n🎉 Fase 1 mejora el contraste en promedio!")
        
        # Crear gráfico resumen
        create_summary_plot(results)
    
    print(f"{'='*60}\n")
    
    return results


def create_summary_plot(results: list):
    """
    Crea gráfico resumen de las comparaciones
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    images = [r['image'] for r in results]
    edge_densities = [r['edge_density'] for r in results]
    improvements = [r['contrast_improvement'] for r in results]
    
    # Gráfico 1: Densidad de bordes
    axes[0].bar(range(len(images)), edge_densities, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(images)))
    axes[0].set_xticklabels([img[:6] for img in images], rotation=45)
    axes[0].set_ylabel('Densidad de Bordes')
    axes[0].set_title('Detección de Bordes (Fase 1)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Mejora de contraste
    colors = ['green' if x > 0 else 'red' for x in improvements]
    axes[1].bar(range(len(images)), improvements, color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(images)))
    axes[1].set_xticklabels([img[:6] for img in images], rotation=45)
    axes[1].set_ylabel('Mejora de Contraste (%)')
    axes[1].set_title('Mejora de Contraste: Fase 1 vs Antiguo', fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "test_results" / "comparison"
    output_path = output_dir / "summary_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Resumen guardado: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparar preprocesamiento antiguo vs Fase 1")
    parser.add_argument('images', nargs='+', help='Rutas a las imágenes')
    
    args = parser.parse_args()
    
    # Convertir a Path objects
    image_paths = [Path(img) for img in args.images]
    
    # Ejecutar comparación
    compare_batch(image_paths)
