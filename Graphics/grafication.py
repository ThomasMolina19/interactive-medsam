import os
import numpy as np
import matplotlib.pyplot as plt
from Segmentation import Masks

def extract_contour_points_3d(segmentations, files, middle_idx, z_spacing=12):
    """
    Extrae los puntos de los contornos y les asigna coordenada Z.
    
    Args:
        segmentations: Diccionario con las segmentaciones {idx: {'mask': ..., ...}}
        files: Lista de archivos de imágenes
        middle_idx: Índice de la imagen del medio
        z_spacing: Separación entre slices en el eje Z
    
    Returns:
        points_3d: Array numpy con puntos (x, y, z)
        slice_info: Lista con información de cada slice
    """
    all_points = []
    slice_info = []
    
    for idx in sorted(segmentations.keys()):
        seg = segmentations[idx]
        mask = seg['mask']
        
        # Calcular coordenada Z relativa al medio
        z = (idx - middle_idx) * z_spacing
        
        # Extraer contornos
        contours = Masks.find_mask_contours(mask)
        
        # Extraer puntos de todos los contornos
        slice_points = []
        for contour in contours:
            # contour tiene shape (N, 1, 2) donde cada punto es [x, y]
            for point in contour:
                x, y = point[0]
                slice_points.append([x, y, z])
        
        all_points.extend(slice_points)
        
        filename = os.path.basename(files[idx])
        slice_info.append({
            'idx': idx,
            'filename': filename,
            'z': z,
            'num_points': len(slice_points)
        })
        
        print(f"  Slice {idx+1}: {filename} → z={z:+4d}, {len(slice_points)} puntos")
    
    return np.array(all_points), slice_info


def plot_3d_contours(points_3d, output_dir, title="Reconstrucción 3D del Húmero"):
    """
    Visualiza los contornos en 3D como nube de puntos.
    
    Args:
        points_3d: Array numpy con puntos (x, y, z)
        output_dir: Directorio para guardar la imagen
        title: Título del gráfico
    """
    if len(points_3d) == 0:
        print("❌ No hay puntos para visualizar")
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar todos los puntos
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                         c=points_3d[:, 2], cmap='viridis', s=1, alpha=0.6)
    
    # Colorbar
    plt.colorbar(scatter, ax=ax, label='Z (profundidad)', shrink=0.5)
    
    # Configurar ejes
    ax.set_xlabel('X (píxeles)')
    ax.set_ylabel('Y (píxeles)')
    ax.set_zlabel('Z (profundidad)')
    ax.set_title(title)
    
    # Ajustar vista
    ax.view_init(elev=20, azim=45)
    
    # Guardar figura
    output_path = os.path.join(output_dir, "reconstruction_3d_points.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"💾 Visualización 3D (puntos) guardada en: {output_path}")
    
    plt.close(fig)

def plot_3d_contours_by_slice(segmentations, files, middle_idx, output_dir, z_spacing=12):
    """
    Visualiza los contornos en 3D, coloreando cada slice diferente.
    Dibuja los contornos como líneas cerradas.
    
    Args:
        segmentations: Diccionario con las segmentaciones
        files: Lista de archivos de imágenes
        middle_idx: Índice de la imagen del medio
        output_dir: Directorio para guardar las imágenes
        z_spacing: Separación entre slices en el eje Z
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colormap para diferentes slices
    num_slices = len(segmentations)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_slices))
    
    for i, idx in enumerate(sorted(segmentations.keys())):
        seg = segmentations[idx]
        mask = seg['mask']
        z = (idx - middle_idx) * z_spacing
        
        # Extraer contornos
        contours = Masks.find_mask_contours(mask)
        
        for contour in contours:
            if len(contour) > 0:
                # Extraer puntos x, y del contorno
                x_pts = contour[:, 0, 0]
                y_pts = contour[:, 0, 1]
                z_pts = np.full_like(x_pts, z, dtype=float)
                
                # Cerrar el contorno (conectar último punto con el primero)
                x_pts = np.append(x_pts, x_pts[0])
                y_pts = np.append(y_pts, y_pts[0])
                z_pts = np.append(z_pts, z_pts[0])
                
                # Dibujar el contorno como línea cerrada
                ax.plot(x_pts, y_pts, z_pts, color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('X (píxeles)')
    ax.set_ylabel('Y (píxeles)')
    ax.set_zlabel('Z (profundidad)')
    ax.set_title(f'Reconstrucción 3D - {num_slices} slices\n(z_spacing = {z_spacing})')
    
    # Guardar múltiples vistas
    views = [
        (20, 45, 'isometrica'),
        (0, 0, 'frontal'),
        (0, 90, 'lateral'),
        (90, 0, 'superior')
    ]
    
    for elev, azim, name in views:
        ax.view_init(elev=elev, azim=azim)
        output_path = os.path.join(output_dir, f"reconstruction_3d_{name}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Vista {name}: {output_path}")
    
    plt.close(fig)

