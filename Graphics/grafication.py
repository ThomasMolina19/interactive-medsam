import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
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


def plot_3d_contours(points_3d, output_dir, title="Reconstrucción 3D del Húmero", interactive=False):
    """
    Visualiza los contornos en 3D como nube de puntos.
    
    Args:
        points_3d: Array numpy con puntos (x, y, z)
        output_dir: Directorio para guardar la imagen
        title: Título del gráfico
        interactive: Si True, muestra la gráfica interactiva para rotar/mover
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
    
    if interactive:
        print("🖱️  Modo interactivo: Arrastra para rotar, scroll para zoom")
        print("   Cierra la ventana para continuar...")
        plt.show()
    else:
        # Guardar figura
        output_path = os.path.join(output_dir, "reconstruction_3d_points.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Visualización 3D (puntos) guardada en: {output_path}")
        plt.close(fig)

def plot_3d_contours_by_slice(segmentations, files, middle_idx, output_dir, z_spacing=12, interactive=False):
    """
    Visualiza los contornos en 3D, coloreando cada slice diferente.
    Dibuja los contornos como líneas cerradas.
    
    Args:
        segmentations: Diccionario con las segmentaciones
        files: Lista de archivos de imágenes
        middle_idx: Índice de la imagen del medio
        output_dir: Directorio para guardar las imágenes
        z_spacing: Separación entre slices en el eje Z
        interactive: Si True, muestra la gráfica interactiva para rotar/mover
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
    
    if interactive:
        print("🖱️  Modo interactivo: Arrastra para rotar, scroll para zoom")
        print("   Cierra la ventana para continuar...")
        plt.show()
    else:
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


def resample_contour(contour, num_points):
    """
    Remuestrea un contorno para que tenga exactamente num_points puntos.
    Usa interpolación para distribuir los puntos uniformemente.
    
    Args:
        contour: Array de puntos del contorno (N, 1, 2) o (N, 2)
        num_points: Número de puntos deseados
    
    Returns:
        Array de puntos remuestreados (num_points, 2)
    """
    # Asegurar formato correcto
    if len(contour.shape) == 3:
        points = contour[:, 0, :]
    else:
        points = contour
    
    if len(points) < 3:
        return None
    
    # Cerrar el contorno si no está cerrado
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Calcular distancia acumulada a lo largo del contorno
    diffs = np.diff(points, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]
    
    if total_length == 0:
        return None
    
    # Normalizar distancias
    cumulative_normalized = cumulative / total_length
    
    # Crear interpoladores para x e y
    interp_x = interp1d(cumulative_normalized, points[:, 0], kind='linear')
    interp_y = interp1d(cumulative_normalized, points[:, 1], kind='linear')
    
    # Generar puntos uniformemente espaciados (sin incluir el último que es igual al primero)
    t_new = np.linspace(0, 1, num_points, endpoint=False)
    
    new_points = np.column_stack([interp_x(t_new), interp_y(t_new)])
    
    return new_points


def create_triangles_between_contours(contour1, contour2, z1, z2):
    """
    Crea triángulos entre dos contornos en diferentes niveles Z.
    
    Args:
        contour1: Puntos del primer contorno (N, 2)
        contour2: Puntos del segundo contorno (N, 2)
        z1: Coordenada Z del primer contorno
        z2: Coordenada Z del segundo contorno
    
    Returns:
        Lista de triángulos (vértices 3D)
    """
    triangles = []
    n = len(contour1)
    
    for i in range(n):
        next_i = (i + 1) % n
        
        # Puntos del contorno inferior
        p1 = [contour1[i][0], contour1[i][1], z1]
        p2 = [contour1[next_i][0], contour1[next_i][1], z1]
        
        # Puntos del contorno superior
        p3 = [contour2[i][0], contour2[i][1], z2]
        p4 = [contour2[next_i][0], contour2[next_i][1], z2]
        
        # Crear dos triángulos para formar un cuadrilátero
        triangles.append([p1, p2, p3])
        triangles.append([p2, p4, p3])
    
    return triangles


def create_cap_triangles(contour, z, is_top=True):
    """
    Crea triángulos para cerrar la tapa superior o inferior del modelo.
    Usa triangulación de abanico desde el centroide.
    
    Args:
        contour: Puntos del contorno (N, 2)
        z: Coordenada Z
        is_top: Si es la tapa superior o inferior
    
    Returns:
        Lista de triángulos
    """
    triangles = []
    n = len(contour)
    
    # Calcular centroide
    centroid = contour.mean(axis=0)
    center = [centroid[0], centroid[1], z]
    
    for i in range(n):
        next_i = (i + 1) % n
        p1 = [contour[i][0], contour[i][1], z]
        p2 = [contour[next_i][0], contour[next_i][1], z]
        
        # Orientación según si es tapa superior o inferior
        if is_top:
            triangles.append([center, p1, p2])
        else:
            triangles.append([center, p2, p1])
    
    return triangles


def plot_3d_solid_mesh(segmentations, files, middle_idx, output_dir, z_spacing=12, 
                       num_points_per_contour=100, color='skyblue', alpha=0.7, 
                       interactive=False, with_caps=True):
    """
    Crea y visualiza una malla sólida 3D conectando los contornos entre slices.
    
    Args:
        segmentations: Diccionario con las segmentaciones {idx: {'mask': ..., ...}}
        files: Lista de archivos de imágenes
        middle_idx: Índice de la imagen del medio
        output_dir: Directorio para guardar las imágenes
        z_spacing: Separación entre slices en el eje Z
        num_points_per_contour: Puntos para remuestrear cada contorno
        color: Color de la superficie
        alpha: Transparencia (0-1)
        interactive: Si True, muestra la gráfica interactiva
        with_caps: Si True, cierra las tapas superior e inferior
    """
    print("\n🔨 Construyendo malla sólida 3D...")
    
    # Extraer contornos de cada slice
    slice_contours = []
    sorted_indices = sorted(segmentations.keys())
    
    for idx in sorted_indices:
        seg = segmentations[idx]
        mask = seg['mask']
        z = (idx - middle_idx) * z_spacing
        
        contours = Masks.find_mask_contours(mask)
        
        if contours and len(contours) > 0:
            # Tomar el contorno más grande
            largest_contour = max(contours, key=lambda c: len(c))
            
            # Remuestrear para tener puntos uniformes
            resampled = resample_contour(largest_contour, num_points_per_contour)
            
            if resampled is not None:
                slice_contours.append({
                    'z': z,
                    'contour': resampled,
                    'idx': idx
                })
                print(f"  ✓ Slice {idx+1}: z={z:+4d}, {len(resampled)} puntos")
    
    if len(slice_contours) < 2:
        print("❌ Se necesitan al menos 2 slices para crear malla sólida")
        return
    
    print(f"\n🔺 Creando triángulos entre {len(slice_contours)} slices...")
    
    # Crear todos los triángulos
    all_triangles = []
    
    # Triángulos entre slices consecutivos
    for i in range(len(slice_contours) - 1):
        c1 = slice_contours[i]
        c2 = slice_contours[i + 1]
        
        triangles = create_triangles_between_contours(
            c1['contour'], c2['contour'], c1['z'], c2['z']
        )
        all_triangles.extend(triangles)
    
    # Tapas superior e inferior
    if with_caps:
        # Tapa inferior (primer slice)
        cap_bottom = create_cap_triangles(
            slice_contours[0]['contour'], 
            slice_contours[0]['z'], 
            is_top=False
        )
        all_triangles.extend(cap_bottom)
        
        # Tapa superior (último slice)
        cap_top = create_cap_triangles(
            slice_contours[-1]['contour'], 
            slice_contours[-1]['z'], 
            is_top=True
        )
        all_triangles.extend(cap_top)
    
    print(f"  Total: {len(all_triangles)} triángulos generados")
    
    # Crear figura
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear colección de polígonos 3D
    mesh = Poly3DCollection(all_triangles, alpha=alpha, linewidths=0.1, edgecolors='darkblue')
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    
    # Configurar límites de los ejes
    all_points = np.array([p for tri in all_triangles for p in tri])
    
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # Añadir margen
    margin = 20
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    ax.set_xlabel('X (píxeles)')
    ax.set_ylabel('Y (píxeles)')
    ax.set_zlabel('Z (profundidad)')
    ax.set_title(f'Reconstrucción 3D Sólida\n{len(slice_contours)} slices, {len(all_triangles)} triángulos')
    
    if interactive:
        print("\n🖱️  Modo interactivo: Arrastra para rotar, scroll para zoom")
        print("   Cierra la ventana para continuar...")
        plt.show()
    else:
        # Guardar múltiples vistas
        views = [
            (20, 45, 'isometrica'),
            (0, 0, 'frontal'),
            (0, 90, 'lateral'),
            (90, 0, 'superior')
        ]
        
        for elev, azim, name in views:
            ax.view_init(elev=elev, azim=azim)
            output_path = os.path.join(output_dir, f"solid_mesh_3d_{name}.png")
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"💾 Vista {name}: {output_path}")
        
        plt.close(fig)
    
    print(f"\n✅ Malla sólida 3D completada")
    return all_triangles


def export_mesh_to_stl(triangles, output_path):
    """
    Exporta la malla de triángulos a formato STL para impresión 3D o software CAD.
    
    Args:
        triangles: Lista de triángulos (cada uno con 3 vértices 3D)
        output_path: Ruta del archivo STL de salida
    """
    print(f"\n📦 Exportando malla a STL: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("solid mesh\n")
        
        for tri in triangles:
            # Calcular normal del triángulo
            v0, v1, v2 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_length = np.linalg.norm(normal)
            
            if norm_length > 0:
                normal = normal / norm_length
            else:
                normal = np.array([0, 0, 1])
            
            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("    outer loop\n")
            
            for vertex in tri:
                f.write(f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid mesh\n")
    
    print(f"✅ Archivo STL guardado: {output_path}")
    print(f"   Triángulos: {len(triangles)}")



