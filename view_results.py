#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualizador de resultados de segmentación.
Muestra las imágenes segmentadas en grillas de 3x3.
"""

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse


def view_segmentation_results(results_dir, images_per_page=9):
    """
    Visualiza los resultados de segmentación en grillas.
    
    Args:
        results_dir: Directorio con las imágenes de resultados
        images_per_page: Imágenes por página (default: 9 para grilla 3x3)
    """
    
    # Buscar todas las imágenes de segmentación
    patterns = ['*_seg.png', '*.png', '*.jpg']
    image_files = []
    
    for pattern in patterns:
        found = glob.glob(os.path.join(results_dir, pattern))
        for f in found:
            if f not in image_files and 'summary' not in f.lower():
                image_files.append(f)
    
    # Ordenar por nombre
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en: {results_dir}")
        return
    
    print(f"📁 Encontradas {len(image_files)} imágenes en {results_dir}")
    
    # Calcular número de páginas
    num_pages = (len(image_files) + images_per_page - 1) // images_per_page
    
    # Determinar grid size
    if images_per_page == 9:
        rows, cols = 3, 3
    elif images_per_page == 6:
        rows, cols = 2, 3
    elif images_per_page == 4:
        rows, cols = 2, 2
    else:
        cols = 3
        rows = (images_per_page + cols - 1) // cols
    
    # Mostrar cada página
    for page in range(num_pages):
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, len(image_files))
        page_images = image_files[start_idx:end_idx]
        
        # Crear figura
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        fig.suptitle(f"Resultados de Segmentación - Página {page + 1}/{num_pages}", 
                     fontsize=14, fontweight='bold')
        
        # Aplanar axes para fácil iteración
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten()
        
        # Mostrar imágenes
        for i, ax in enumerate(axes_flat):
            if i < len(page_images):
                img_path = page_images[i]
                img_name = os.path.basename(img_path)
                
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(img_name, fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                    ax.set_title(img_name, fontsize=9, color='red')
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Si hay más páginas, preguntar si continuar
        if page < num_pages - 1:
            print(f"\n📄 Página {page + 1}/{num_pages} mostrada.")
            print("   Cierra la ventana para ver la siguiente página...")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizador de resultados de segmentación"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/D1_propagation_results",
        help="Directorio con los resultados de segmentación"
    )
    parser.add_argument(
        "--grid", "-g",
        type=int,
        default=9,
        choices=[4, 6, 9],
        help="Imágenes por página: 4 (2x2), 6 (2x3), 9 (3x3)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🖼️  VISUALIZADOR DE RESULTADOS DE SEGMENTACIÓN")
    print("="*60)
    
    view_segmentation_results(args.input, args.grid)
    
    print("\n✅ Visualización completada!")


if __name__ == "__main__":
    main()
