"""Segmentación interactiva con selección de puntos."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from scipy import ndimage
from skimage import morphology

from .config import IMAGE_ENHANCE_ALPHA, IMAGE_ENHANCE_BETA, MIN_MASK_SIZE, DISK_RADIUS


def interactive_segment(model, img):
    """
    Segmentación interactiva con selección de puntos.
    
    Args:
        model: Instancia de SAMModel ya cargada
        img: Imagen RGB numpy array
        
    Returns:
        Máscara binaria refinada o None si se cancela
    """
    # Mejorar contraste
    img_enhanced = cv2.convertScaleAbs(img, alpha=IMAGE_ENHANCE_ALPHA, beta=IMAGE_ENHANCE_BETA)
    model.set_image(img_enhanced)
    
    # Limpiar memoria MPS si está disponible
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    class PointSelector:
        def __init__(self, ax_img, ax_mask, fig):
            self.positive_points = []
            self.negative_points = []
            self.ax_img = ax_img
            self.ax_mask = ax_mask
            self.fig = fig
            self.point_markers = []
            self.mask_display = None
            self.current_mask = None
            
        def update_segmentation(self):
            """Actualiza segmentación en tiempo real."""
            # Limpiar máscara anterior
            if self.mask_display is not None:
                self.mask_display.remove()
                self.mask_display = None
            
            # Si no hay puntos, mostrar imagen vacía
            if len(self.positive_points) == 0 and len(self.negative_points) == 0:
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.ax_mask.set_title("Mascara (agrega puntos para ver)")
                self.ax_mask.axis('off')
                self.fig.canvas.draw_idle()
                self.current_mask = None
                return
            
            # Preparar puntos y labels
            input_points = []
            input_labels = []
            
            for point in self.positive_points:
                input_points.append(point)
                input_labels.append(1)
            
            for point in self.negative_points:
                input_points.append(point)
                input_labels.append(0)
            
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            
            # Generar máscara
            try:
                # Limpiar cache MPS antes de predicción
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                masks, scores, _ = model.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]
                self.current_mask = best_mask
                
                # Mostrar máscara
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.mask_display = self.ax_mask.imshow(best_mask, alpha=0.6, cmap='Blues')
                
                # Mostrar puntos en vista de máscara
                for point in self.positive_points:
                    self.ax_mask.plot(point[0], point[1], 'g*', markersize=15, markeredgewidth=2)
                for point in self.negative_points:
                    self.ax_mask.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
                
                score = scores[np.argmax(scores)]
                area = np.sum(best_mask)
                self.ax_mask.set_title(f"Segmentacion | Score: {score:.3f} | Area: {area} px")
                self.ax_mask.axis('off')
                
            except Exception as e:
                print(f"[!] Error en segmentacion: {e}")
            
            self.fig.canvas.draw_idle()
            
        def onclick(self, event):
            if event.inaxes != self.ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            
            # Botón izquierdo (1) = Punto NEGATIVO
            if event.button == 1:
                self.negative_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_markers.append(('neg', marker))
                print(f"[-] Punto NEGATIVO: ({x:.0f}, {y:.0f})")
                
            # Botón derecho (3) = Punto POSITIVO
            elif event.button == 3:
                self.positive_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'g*', markersize=20, markeredgewidth=2)[0]
                self.point_markers.append(('pos', marker))
                print(f"[+] Punto POSITIVO: ({x:.0f}, {y:.0f})")
            
            # Actualizar título
            self.ax_img.set_title(
                f"[+] Positivos: {len(self.positive_points)} | "
                f"[-] Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar"
            )
            
            # Actualizar segmentación
            self.update_segmentation()
            
        def onkey(self, event):
            """Manejar eventos de teclado."""
            # Z = Deshacer
            if event.key == 'z':
                if len(self.point_markers) > 0:
                    point_type, marker = self.point_markers.pop()
                    marker.remove()
                    
                    if point_type == 'pos' and len(self.positive_points) > 0:
                        removed = self.positive_points.pop()
                        print(f"[<-] Deshecho POSITIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    elif point_type == 'neg' and len(self.negative_points) > 0:
                        removed = self.negative_points.pop()
                        print(f"[<-] Deshecho NEGATIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    
                    self.ax_img.set_title(
                        f"[+] Positivos: {len(self.positive_points)} | "
                        f"[-] Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar"
                    )
                    self.update_segmentation()
            
            # C = Limpiar todo
            elif event.key == 'c':
                for _, marker in self.point_markers:
                    marker.remove()
                self.point_markers.clear()
                self.positive_points.clear()
                self.negative_points.clear()
                print("[x] Puntos limpiados")
                self.ax_img.set_title("[+] Positivos: 0 | [-] Negativos: 0 | 'z': deshacer | 'c': limpiar")
                self.update_segmentation()
    
    # Crear figura con 2 subplots
    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Izquierda: Imagen con puntos
    ax_img.imshow(img)
    ax_img.set_title("Click derecho = POSITIVO (verde) | Click izquierdo = NEGATIVO (rojo)")
    ax_img.axis('off')
    
    # Derecha: Máscara en tiempo real
    ax_mask.imshow(img)
    ax_mask.set_title("Segmentacion (agrega puntos para ver)")
    ax_mask.axis('off')
    
    selector = PointSelector(ax_img, ax_mask, fig)
    
    # Conectar eventos
    fig.canvas.mpl_connect('button_press_event', selector.onclick)
    fig.canvas.mpl_connect('key_press_event', selector.onkey)
    
    # Instrucciones
    plt.figtext(0.5, 0.02, 
                "[DERECHO] Positivo | [IZQUIERDO] Negativo | "
                "'z': Deshacer | 'c': Limpiar | Cierra ventana para continuar", 
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", 
                         edgecolor="blue", linewidth=2))
    
    plt.tight_layout()
    plt.show()
    
    # Refinar máscara final
    if selector.current_mask is not None and np.sum(selector.current_mask) > 0:
        return refine_mask(selector.current_mask)
    
    return None


def refine_mask(mask, min_size=None, disk_radius=None):
    """Refina la máscara de segmentación."""
    if min_size is None:
        min_size = MIN_MASK_SIZE
    if disk_radius is None:
        disk_radius = DISK_RADIUS
        
    if np.sum(mask) == 0:
        return mask
    
    # Remover objetos pequeños
    mask_clean = morphology.remove_small_objects(mask, min_size=min_size)
    
    # Rellenar huecos
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Suavizar
    kernel = morphology.disk(disk_radius)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth
