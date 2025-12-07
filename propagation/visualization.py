"""Visualización y guardado de resultados de segmentación."""

import os
import numpy as np
import matplotlib.pyplot as plt

from .config import SIMILARITY_THRESHOLD, WARNING_THRESHOLD


def save_segmentation_result(img, mask, filename, out_dir,
                              center=None, seg_point=None, neg_point=None, info=""):
    """
    Guarda visualización de segmentación con overlay y puntos.
    
    Args:
        img: Imagen RGB numpy array
        mask: Máscara binaria
        filename: Nombre base del archivo (sin extensión)
        out_dir: Directorio de salida
        center: Centro de la máscara [x, y]
        seg_point: Punto de segmentación usado [x, y]
        neg_point: Punto negativo usado [x, y]
        info: Información adicional para el título
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Imagen original
    axes[0].imshow(img)
    axes[0].set_title(f"{filename}\nOriginal")
    axes[0].axis('off')
    
    # Panel 2: Overlay con puntos
    axes[1].imshow(img)
    axes[1].imshow(mask, alpha=0.5, cmap='Blues')
    
    if seg_point is not None:
        axes[1].plot(seg_point[0], seg_point[1], 'r*', markersize=18,
                     markeredgewidth=2, label='Pto positivo')
    
    if neg_point is not None:
        axes[1].plot(neg_point[0], neg_point[1], 'bX', markersize=16,
                     markeredgewidth=3, label='Pto negativo')
    
    if center is not None:
        axes[1].plot(center[0], center[1], 'g*', markersize=14,
                     markeredgewidth=2, label='Centro máscara')
    
    if seg_point is not None or center is not None or neg_point is not None:
        axes[1].legend(loc='upper right', fontsize=8)
    
    axes[1].set_title(f"Overlay\n{info}")
    axes[1].axis('off')
    
    # Panel 3: Máscara sola
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Mask\nArea: {np.sum(mask)} px")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(out_dir, f"{filename}_seg.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_summary(output_dir, data_dir, ckpt, files, middle_idx, segmentations,
                 similarity_threshold=None, warning_threshold=None):
    """
    Guarda resumen de la propagación en archivo de texto.
    
    Args:
        output_dir: Directorio de salida
        data_dir: Directorio de datos original
        ckpt: Ruta al checkpoint
        files: Lista de archivos procesados
        middle_idx: Índice de la imagen del medio
        segmentations: Diccionario de segmentaciones
        similarity_threshold: Umbral para advertencias leves
        warning_threshold: Umbral para advertencias severas
        
    Returns:
        tuple: (summary_path, warnings_count, severe_warnings_count)
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD
    if warning_threshold is None:
        warning_threshold = WARNING_THRESHOLD
    
    # Contar advertencias
    warnings_count = sum(
        1 for s in segmentations.values()
        if 'dice' in s and (1.0 - s['dice']) > similarity_threshold
    )
    severe_warnings = sum(
        1 for s in segmentations.values()
        if 'dice' in s and (1.0 - s['dice']) > warning_threshold
    )
    
    # Calcular estadísticas de Dice
    dice_scores = [s['dice'] for s in segmentations.values() if 'dice' in s]
    
    summary_path = os.path.join(output_dir, "propagation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SAM Complete Folder Segmentation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Checkpoint: {ckpt}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Middle image: {os.path.basename(files[middle_idx])} (index {middle_idx+1})\n")
        f.write(f"Initial segmentation: segment_sam_points.py\n")
        f.write(f"Warning threshold: {similarity_threshold*100:.0f}%\n")
        f.write(f"Severe warning threshold: {warning_threshold*100:.0f}%\n")
        f.write(f"Total images: {len(files)}\n")
        f.write(f"Successfully segmented: {len(segmentations)}\n")
        f.write(f"Images with warnings: {warnings_count}\n")
        f.write(f"Images with severe warnings: {severe_warnings}\n")
        
        if dice_scores:
            f.write(f"\nDice Statistics:\n")
            f.write(f"  - Average: {np.mean(dice_scores):.3f}\n")
            f.write(f"  - Min: {np.min(dice_scores):.3f}\n")
            f.write(f"  - Max: {np.max(dice_scores):.3f}\n")
        
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
                if diff > warning_threshold:
                    warning_marker = " [SEVERA]"
                elif diff > similarity_threshold:
                    warning_marker = " [ADVERTENCIA]"
            
            f.write(f"{filename}: Dice={dice_str}, Area={s['area']}px{warning_marker}\n")
    
    return summary_path, warnings_count, severe_warnings


def print_final_summary(output_dir, segmentations, total_files, summary_path,
                        warnings_count, severe_warnings):
    """
    Imprime resumen final en consola.
    
    Args:
        output_dir: Directorio de salida
        segmentations: Diccionario de segmentaciones
        total_files: Número total de archivos
        summary_path: Ruta al archivo de resumen
        warnings_count: Número de advertencias leves
        severe_warnings: Número de advertencias severas
    """
    print("\n" + "="*70)
    print("🎉 PROCESAMIENTO COMPLETADO - CARPETA COMPLETA")
    print("="*70)
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"✅ Segmentaciones exitosas: {len(segmentations)}/{total_files} imágenes")
    
    if warnings_count > 0:
        print(f"⚠️  Imágenes con advertencias: {warnings_count}")
    if severe_warnings > 0:
        print(f"🚨 Imágenes con advertencias severas: {severe_warnings}")
    
    # Estadísticas de Dice
    dice_scores = [s['dice'] for s in segmentations.values() if 'dice' in s]
    if dice_scores:
        print(f"📊 Estadísticas de similitud:")
        print(f"   - Dice promedio: {np.mean(dice_scores):.3f}")
        print(f"   - Dice mínimo: {np.min(dice_scores):.3f}")
        print(f"   - Dice máximo: {np.max(dice_scores):.3f}")
    
    print("="*70)
    print(f"\n📝 Resumen guardado en: {summary_path}")
