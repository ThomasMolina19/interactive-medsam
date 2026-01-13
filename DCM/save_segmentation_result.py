from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
import Segmentation.Masks as Masks


def save_segmentation_result(img, mask, filename, output_dir, center=None, seg_point=None, neg_point=None, info=""):
    """Save segmentation visualization
    
    Args:
        img: Imagen original
        mask: Máscara de segmentación
        filename: Nombre del archivo
        output_dir: Directorio de salida
        center: Centro calculado de la máscara resultante (verde)
        seg_point: Punto usado para segmentar (rojo)
        neg_point: Punto negativo usado (azul con X)
        info: Información adicional para el título
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title(f"{filename}\nOriginal")
    axes[0].axis('off')
    
    # Overlay
    axes[1].imshow(img)
    axes[1].imshow(mask, alpha=0.5, cmap='Blues')
    
    # Punto usado para segmentar (ROJO)
    if seg_point is not None:
        axes[1].plot(seg_point[0], seg_point[1], 'r*', markersize=18, markeredgewidth=2, label='Pto positivo')
    
    # Punto negativo (AZUL con X)
    if neg_point is not None:
        axes[1].plot(neg_point[0], neg_point[1], 'bX', markersize=16, markeredgewidth=3, label='Pto negativo')
    
    # Centro calculado de la máscara (VERDE)
    if center is not None:
        axes[1].plot(center[0], center[1], 'g*', markersize=14, markeredgewidth=2, label='Centro máscara')
    
    # Leyenda
    if seg_point is not None or center is not None or neg_point is not None:
        axes[1].legend(loc='upper right', fontsize=8)
    
    axes[1].set_title(f"Overlay\n{info}")
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Mask\nArea: {np.sum(mask)} px")
    axes[2].axis('off')
    
    # Contornos/Bordes sobre la imagen original
    img_with_contours = Masks.draw_contours_on_image(img, mask, color=(0, 255, 0), thickness=2)
    axes[3].imshow(img_with_contours)
    
    # También mostrar puntos en el panel de contornos
    if seg_point is not None:
        axes[3].plot(seg_point[0], seg_point[1], 'r*', markersize=14, markeredgewidth=2)
    if center is not None:
        axes[3].plot(center[0], center[1], 'g*', markersize=10, markeredgewidth=2)
    
    # Contar contornos encontrados
    contours = Masks.find_mask_contours(mask)
    axes[3].set_title(f"Contornos\n{len(contours)} contorno(s)")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{filename}_seg.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
