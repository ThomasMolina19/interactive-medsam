#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentación completa de carpeta con SAM usando propagación de centros.
Proceso:
1. Usuario clickea la imagen del medio
2. Propaga hacia arriba y abajo usando centros calculados
3. Procesa TODAS las imágenes (no se detiene por umbral)
4. Registra advertencias cuando hay cambios grandes
"""

# Standard library
import os

# Third-party libraries
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# Local modules
from DCM.load_dicom_as_image import read_image_file, get_dataset_files
from Graphics.grafication import (
    extract_contour_points_3d,
    plot_3d_contours,
    plot_3d_contours_by_slice,
)
import Segmentation.Masks as Masks
from Segmentation.propagation import propagate_segmentation
from Segmentation.segment_image import segment_first_image


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Paths - MODIFICAR SEGÚN TUS NECESIDADES
ckpt = "/Users/thomasmolinamolina/Downloads/TopicosGeo/Checkpoints/sam_vit_b_01ec64.pth" # Ruta al checkpoint de SAM 
data_dir = "/Users/thomasmolinamolina/Downloads/TopicosGeo/DATA/D9/pngs"  # Carpeta con JPG o PNG
output_dir = "/Users/thomasmolinamolina/Downloads/TopicosGeo/DATA/D9_propagation_results" #carpeta de resultados



# Parámetros
SIMILARITY_THRESHOLD = 0.35  # 20% - Solo para advertencias, NO detiene la propagación
WARNING_THRESHOLD = 0.45     # 30% - Advertencia severa pero continúa

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load SAM model
print("🔄 Loading SAM model...")
# Cargar modelo sin checkpoint primero, luego cargar pesos con map_location
sam = sam_model_registry["vit_b"]()
checkpoint_data = torch.load(ckpt, map_location=device)
sam.load_state_dict(checkpoint_data)
sam = sam.to(device)
predictor = SamPredictor(sam)
print("✅ SAM model loaded!")


def main():
    
    # Obtener información del dataset (sin cargar imagen aún)
    dataset = get_dataset_files(data_dir)
    if dataset is None:
        return
    
    # Cargar imagen del medio solo cuando se necesita
    middle_img = read_image_file(dataset.middle_file)
    if middle_img is None:
        print("❌ Error reading middle image!")
        return
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    
    
    files = dataset.files
    middle_idx = dataset.middle_idx
    middle_name = dataset.middle_name
    middle_file = dataset.middle_file
    
    # STEP 1: Usuario segmenta la imagen del medio usando segment_sam_points.py
    print("\n" + "="*70)
    print("🖱️  PASO 1: SEGMENTACIÓN DE IMAGEN DEL MEDIO")
    print("="*70)
    print("Se abrirá segment_sam_points.py para segmentar la imagen del medio.")
    print("Instrucciones:")
    print("  • Click DERECHO: Punto positivo (dentro del objeto)")
    print("  • Click IZQUIERDO: Punto negativo (fuera del objeto)")
    print("  • Tecla 'z': Deshacer último punto")
    print("  • Tecla 'c': Limpiar todos los puntos")
    print("  • Cierra la ventana para guardar y continuar")
    print("="*70)
    
    # Ejecutar segment_sam_points.py con la imagen del medio
    middle_mask, middle_score = segment_first_image(predictor, middle_img , middle_name)
    
    if middle_mask is None or np.sum(middle_mask) == 0:
        print("❌ No se pudo segmentar la imagen del medio!")
        return
    
    
    
    print(f"\n📌 Segmentación completada usando segment_sam_points.py")
    
    # La máscara ya está refinada desde segment_sam_points.py
    # No necesitamos volver a segmentar
    
    # Calculate center of middle segmentation
    middle_center = Masks.calculate_mask_center(middle_mask)
    
    if middle_center is None:
        print("❌ Failed to calculate center of middle segmentation!")
        return
    
    print(f"✅ Centro calculado: ({middle_center[0]:.0f}, {middle_center[1]:.0f})")
    print(f"   Score: {middle_score:.3f}, Area: {np.sum(middle_mask)} px")
    
    # Save middle segmentation (para la imagen del medio, seg_point = center ya que fue interactiva)
    Masks.save_segmentation_result(middle_img, middle_mask, middle_name, output_dir, 
                            center=middle_center, seg_point=middle_center, 
                            info=f"Score: {middle_score:.3f}")
    
    
    # STEP 2: Segmentar TODAS las imágenes
    print("\n" + "="*70)
    print("🔄 PASO 2: SEGMENTACIÓN DE TODAS LAS IMÁGENES")
    print("="*70)
    print("⚡ Procesará TODAS las imágenes sin detenerse")
    print(f"⚠️  Advertencias cuando diferencia > {SIMILARITY_THRESHOLD*100:.0f}%")
    
    # Initialize results storage
    segmentations = {middle_idx: {
        'mask': middle_mask,
        'center': middle_center,
        'seg_point': middle_center,  # Para la imagen del medio, es el mismo
        'score': middle_score,
        'area': np.sum(middle_mask)
    }}
    
    # Lista de slices fallidas (diferencia > 30%)
    failed_slices = []
    
    # STEP 3: Propagar a todas las imágenes
    print("\n" + "="*70)
    print("🔄 PASO 3: PROPAGACIÓN COMPLETA")
    print("="*70)
    print(f"📊 Umbrales de advertencia: {SIMILARITY_THRESHOLD*100:.0f}% (leve) / {WARNING_THRESHOLD*100:.0f}% (severa)")
    print("✅ Se procesarán TODAS las imágenes")
    print("="*70)
    
    # Propagate backwards (hacia arriba)
    segmentations, failed_slices = propagate_segmentation(
        predictor, files, middle_idx, middle_mask, middle_center,
        segmentations, failed_slices, output_dir, direction="backward"
    )
    
    # Propagate forwards (hacia abajo)
    segmentations, failed_slices = propagate_segmentation(
        predictor, files, middle_idx, middle_mask, middle_center,
        segmentations, failed_slices, output_dir, direction="forward"
    )
    
    # STEP 4: Reconstrucción 3D
    print("\n" + "="*70)
    print("🎨 PASO 4: RECONSTRUCCIÓN 3D")
    print("="*70)
    
    # Extraer puntos 3D de los contornos
    print("📍 Extrayendo puntos de contornos...")
    z_spacing = 12  # Separación entre slices
    points_3d, slice_info = extract_contour_points_3d(segmentations, files, middle_idx, z_spacing=z_spacing)
    
    print(f"\n📊 Total: {len(points_3d)} puntos 3D de {len(segmentations)} slices")
    print(f"   Separación Z entre slices: {z_spacing} unidades")
    
    if len(points_3d) > 0:
        # Guardar puntos en archivo numpy (para usar en otros programas)
        points_path = os.path.join(output_dir, "contour_points_3d.npy")
        np.save(points_path, points_3d)
        print(f"💾 Puntos guardados en: {points_path}")
        
        # También guardar como CSV para compatibilidad
        csv_path = os.path.join(output_dir, "contour_points_3d.csv")
        np.savetxt(csv_path, points_3d, delimiter=',', header='x,y,z', comments='')
        print(f"💾 CSV guardado en: {csv_path}")
        
        # Visualización 3D - Nube de puntos
        print("\n🎨 Generando visualización 3D (nube de puntos)...")
        plot_3d_contours(points_3d, output_dir, title=f"Reconstrucción 3D - {len(segmentations)} slices")
        
        # Visualización 3D - Contornos por slice
        print("\n🎨 Generando visualización 3D (contornos por slice)...")
        plot_3d_contours_by_slice(segmentations, files, middle_idx, output_dir, z_spacing=z_spacing)
    else:
        print("⚠️ No hay puntos 3D para visualizar")
    
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("🎉 PROCESAMIENTO COMPLETADO - CARPETA COMPLETA")
    print("="*70)
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"✅ Segmentaciones exitosas: {len(segmentations)}/{len(files)} imágenes")
    
    # Contar advertencias
    warnings_count = sum(1 for s in segmentations.values() if 'dice' in s and (1.0 - s['dice']) > SIMILARITY_THRESHOLD)
    
    if warnings_count > 0:
        print(f"⚠️  Imágenes con advertencias leves: {warnings_count}")
    
    # Mostrar imágenes fallidas
    if len(failed_slices) > 0:
        print(f"\n🚨 IMÁGENES FALLIDAS ({len(failed_slices)} total):")
        for failed in failed_slices:
            dice_str = f" (Dice: {failed['dice']:.3f})" if 'dice' in failed else ""
            print(f"   - [{failed['idx']+1}] {failed['filename']}: {failed['reason']}{dice_str}")
    
    # Calculate statistics
    if len(segmentations) > 1:
        dice_scores = [s['dice'] for s in segmentations.values() if 'dice' in s]
        if dice_scores:
            print(f"📊 Estadísticas de similitud:")
            print(f"   - Dice promedio: {np.mean(dice_scores):.3f}")
            print(f"   - Dice mínimo: {np.min(dice_scores):.3f}")
            print(f"   - Dice máximo: {np.max(dice_scores):.3f}")
    
    print("="*70)
    
    # Save summary
    summary_path = os.path.join(output_dir, "propagation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SAM Complete Folder Segmentation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Middle image: {os.path.basename(middle_file)} (index {middle_idx+1})\n")
        f.write(f"Initial segmentation: segment_sam_points.py\n")
        f.write(f"Warning threshold: {SIMILARITY_THRESHOLD*100:.0f}%\n")
        f.write(f"Severe warning threshold: {WARNING_THRESHOLD*100:.0f}%\n")
        f.write(f"Total images: {len(files)}\n")
        f.write(f"Successfully segmented: {len(segmentations)}\n")
        f.write(f"Images with warnings: {warnings_count}\n")
        f.write(f"Failed images (skipped): {len(failed_slices)}\n")
        
        if len(failed_slices) > 0:
            f.write("\nFailed slices:\n")
            for failed in failed_slices:
                dice_str = f" (Dice: {failed['dice']:.3f})" if 'dice' in failed else ""
                f.write(f"  - [{failed['idx']+1}] {failed['filename']}: {failed['reason']}{dice_str}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Results per image:\n")
        f.write("-"*70 + "\n")
        
        for idx in sorted(segmentations.keys()):
            s = segmentations[idx]
            filename = os.path.basename(files[idx])
            dice_str = f"{s['dice']:.3f}" if 'dice' in s else "REF"
            
            # Marcar advertencias
            warning_marker = ""
            if 'dice' in s:
                diff = 1.0 - s['dice']
                if diff > WARNING_THRESHOLD:
                    warning_marker = " 🚨"
                elif diff > SIMILARITY_THRESHOLD:
                    warning_marker = " ⚠️"
            
            # Formatear puntos
            seg_pt_str = f"({s['seg_point'][0]:.0f},{s['seg_point'][1]:.0f})" if 'seg_point' in s else "N/A"
            center_str = f"({s['center'][0]:.0f},{s['center'][1]:.0f})"
            
            f.write(f"{idx+1:3d}. {filename:<15} | Area: {s['area']:>7.0f} px | Dice: {dice_str} | Score: {s['score']:.3f} | SegPt: {seg_pt_str} | Centro: {center_str}{warning_marker}\n")
    
    print(f"💾 Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
