import numpy as np
import matplotlib.pyplot as plt

def interactive_sam_point_selector(img, predictor, filename):
    """
    Interactive point selection for SAM segmentation with REAL-TIME preview.
    Modified for batch processing with filename display.
    """
    
    class PointSelector:
        def __init__(self, ax_img, ax_mask):
            self.positive_points = []
            self.negative_points = []
            self.ax_img = ax_img
            self.ax_mask = ax_mask
            self.point_markers = []
            self.mask_display = None
            self.current_mask = None
            
        def update_segmentation(self):
            """Update segmentation in real-time"""
            # Clear previous mask
            if self.mask_display is not None:
                self.mask_display.remove()
                self.mask_display = None
            
            # If no points, return
            if len(self.positive_points) == 0 and len(self.negative_points) == 0:
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.ax_mask.set_title("Máscara (agrega puntos para ver)")
                self.ax_mask.axis('off')
                fig.canvas.draw()
                return
            
            # Prepare points and labels
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
            
            # Generate mask
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]
                self.current_mask = best_mask
                
                # Display mask on right subplot
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.mask_display = self.ax_mask.imshow(best_mask, alpha=0.6, cmap='Blues')
                
                # Show points on mask view too
                for point in self.positive_points:
                    self.ax_mask.plot(point[0], point[1], 'g*', markersize=15, markeredgewidth=2)
                for point in self.negative_points:
                    self.ax_mask.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
                
                score = scores[np.argmax(scores)]
                area = np.sum(best_mask)
                self.ax_mask.set_title(f"Segmentación | Score: {score:.3f} | Área: {area} px")
                self.ax_mask.axis('off')
                
            except Exception as e:
                print(f"⚠️ Error en segmentación: {e}")
            
            fig.canvas.draw()
            
        def onclick(self, event):
            if event.inaxes != self.ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            
            # Botón izquierdo (1) = Punto NEGATIVO (rojo)
            if event.button == 1:
                self.negative_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_markers.append(('neg', marker))
                print(f"❌ Punto NEGATIVO agregado: ({x:.0f}, {y:.0f})")
                
            # Botón derecho (3) = Punto POSITIVO (verde)
            elif event.button == 3:
                self.positive_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'g*', markersize=20, markeredgewidth=2)[0]
                self.point_markers.append(('pos', marker))
                print(f"✅ Punto POSITIVO agregado: ({x:.0f}, {y:.0f})")
            
            # Update title with counts
            self.ax_img.set_title(f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar | 's': skip")
            
            # Update segmentation in real-time
            self.update_segmentation()
            
        def onkey(self, event):
            """Handle keyboard events"""
            # Z = Undo last point
            if event.key == 'z':
                if len(self.point_markers) > 0:
                    point_type, marker = self.point_markers.pop()
                    marker.remove()
                    
                    if point_type == 'pos' and len(self.positive_points) > 0:
                        removed = self.positive_points.pop()
                        print(f"↩️  Deshecho punto POSITIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    elif point_type == 'neg' and len(self.negative_points) > 0:
                        removed = self.negative_points.pop()
                        print(f"↩️  Deshecho punto NEGATIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    
                    self.ax_img.set_title(f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar | 's': skip")
                    self.update_segmentation()
            
            # C = Clear all points
            elif event.key == 'c':
                for _, marker in self.point_markers:
                    marker.remove()
                self.point_markers.clear()
                self.positive_points.clear()
                self.negative_points.clear()
                print("🧹 Todos los puntos limpiados")
                self.ax_img.set_title(f"✅ Positivos: 0 | ❌ Negativos: 0 | 'z': deshacer | 'c': limpiar | 's': skip")
                self.update_segmentation()
            
            # S = Skip this image
            elif event.key == 's':
                print(f"⏭️ Saltando imagen: {filename}")
                plt.close(fig)
    
    # Create the selector object with 2 subplots
    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Image with points
    ax_img.imshow(img)
    ax_img.set_title(f"🎯 {filename} | Click derecho = POSITIVO | Click izquierdo = NEGATIVO")
    ax_img.axis('off')
    
    # Right: Real-time mask
    ax_mask.imshow(img)
    ax_mask.set_title("Segmentación (agrega puntos para ver)")
    ax_mask.axis('off')
    
    selector_obj = PointSelector(ax_img, ax_mask)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', selector_obj.onclick)
    fig.canvas.mpl_connect('key_press_event', selector_obj.onkey)
    
    # Instructions
    plt.figtext(0.5, 0.02, 
                "🟢 Click DERECHO: Punto positivo | 🔴 Click IZQUIERDO: Punto negativo | ⌨️ 'z': Deshacer | 'c': Limpiar | 's': Saltar | ENTER/ESC: Siguiente", 
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="blue", linewidth=2))
    
    plt.tight_layout()
    plt.show()
    
    return selector_obj.positive_points, selector_obj.negative_points