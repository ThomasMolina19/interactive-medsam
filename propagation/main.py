#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentación completa de carpeta con SAM usando propagación de centros.
VERSIÓN INTERACTIVA MODULAR.

Proceso:
1. Usuario ingresa paths por consola (Finder)
2. Usuario clickea la imagen del medio (segment_sam_points.py)
3. Propaga hacia arriba y abajo usando centros calculados
4. Procesa TODAS las imágenes (no se detiene por umbral)
5. Registra advertencias cuando hay cambios grandes
"""

import sys
import os

# Agregar path de segment-anything si es necesario
# sys.path.append('path/to/segment-anything')

import glob
import numpy as np

from .config import SIMILARITY_THRESHOLD, WARNING_THRESHOLD, IMAGE_EXTENSIONS
from .ui import get_user_paths
from .model import SAMModel
from .mask_utils import calculate_mask_center
from .propagation import propagate_direction, read_image_file
from .visualization import save_segmentation_result, save_summary, print_final_summary
from .interactive import interactive_segment


def main():
    """Función principal con propagación iterativa."""
    
    # Obtener paths del usuario
    ckpt, data_dir, output_dir = get_user_paths()
    
    # Cargar modelo SAM
    model = SAMModel(ckpt)
    
    # Buscar archivos de imagen
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    files = sorted(files, key=lambda x: os.path.basename(x))
    
    if len(files) == 0:
        print(f"❌ No image files found in {data_dir}")
        return
    
    print(f"📁 Found {len(files)} image files in {data_dir}")
    print(f"   Files: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
    
    # Encontrar imagen del medio
    middle_idx = len(files) // 2
    middle_file = files[middle_idx]
    middle_name = os.path.basename(middle_file).split('.')[0]
    
    print(f"\n🎯 Middle image: {os.path.basename(middle_file)} (index {middle_idx+1}/{len(files)})")
    
    middle_img = read_image_file(middle_file)
    if middle_img is None:
        print("❌ Error reading middle image!")
        return
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    
    # PASO 1: Usuario segmenta la imagen del medio
    print("\n" + "="*70)
    print("🖱️  PASO 1: SEGMENTACIÓN DE IMAGEN DEL MEDIO")
    print("="*70)
    print("Instrucciones:")
    print("  • Click DERECHO: Punto positivo (dentro del objeto)")
    print("  • Click IZQUIERDO: Punto negativo (fuera del objeto)")
    print("  • Tecla 'z': Deshacer último punto")
    print("  • Tecla 'c': Limpiar todos los puntos")
    print("  • Cierra la ventana para guardar y continuar")
    print("="*70)
    
    middle_mask = interactive_segment(model, middle_img)
    
    if middle_mask is None or np.sum(middle_mask) == 0:
        print("❌ No se pudo segmentar la imagen del medio!")
        return
    
    middle_score = 1.0
    print(f"\n📌 Segmentación completada")
    
    # Calcular centro
    middle_center = calculate_mask_center(middle_mask)
    if middle_center is None:
        print("❌ Failed to calculate center of middle segmentation!")
        return
    
    print(f"✅ Centro calculado: ({middle_center[0]:.0f}, {middle_center[1]:.0f})")
    print(f"   Score: {middle_score:.3f}, Area: {np.sum(middle_mask)} px")
    
    # Guardar segmentación inicial
    save_segmentation_result(
        middle_img, middle_mask, middle_name, output_dir,
        center=middle_center, seg_point=middle_center,
        info=f"Score: {middle_score:.3f}"
    )
    
    # Inicializar diccionario de segmentaciones
    segmentations = {
        middle_idx: {
            'mask': middle_mask,
            'center': middle_center,
            'seg_point': middle_center,
            'score': middle_score,
            'area': np.sum(middle_mask)
        }
    }
    
    # PASO 2 y 3: Propagación
    print("\n" + "="*70)
    print("🔄 PASO 2-3: PROPAGACIÓN COMPLETA")
    print("="*70)
    print(f"📊 Umbrales de advertencia: {SIMILARITY_THRESHOLD*100:.0f}% (leve) / {WARNING_THRESHOLD*100:.0f}% (severa)")
    print("✅ Se procesarán TODAS las imágenes")
    print("="*70)
    
    # Propagar hacia arriba (backward)
    seg_backward = propagate_direction(
        model, files, middle_idx, middle_mask, middle_center,
        output_dir, direction='backward'
    )
    segmentations.update(seg_backward)
    
    # Propagar hacia abajo (forward)
    seg_forward = propagate_direction(
        model, files, middle_idx, middle_mask, middle_center,
        output_dir, direction='forward'
    )
    segmentations.update(seg_forward)
    
    # Guardar resumen
    summary_path, warnings_count, severe_warnings = save_summary(
        output_dir, data_dir, ckpt, files, middle_idx, segmentations,
        SIMILARITY_THRESHOLD, WARNING_THRESHOLD
    )
    
    # Imprimir resumen final
    print_final_summary(
        output_dir, segmentations, len(files), summary_path,
        warnings_count, severe_warnings
    )


if __name__ == "__main__":
    main()
