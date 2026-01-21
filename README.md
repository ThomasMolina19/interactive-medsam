# 🏥 Medical Image Segmentation with SAM

Herramienta de segmentación de imágenes médicas utilizando **SAM** (Segment Anything Model). Soporta dos modos de operación: segmentación de **una sola imagen** o **volúmenes completos** con propagación automática y reconstrucción 3D.

## 🎯 Dos Modos de Operación

| Característica | `one_segmentation.py` | `segment_sam_propagation.py` |
|----------------|----------------------|------------------------------|
| **Propósito** | 🖼️ Segmentar **una sola imagen** | 📦 Segmentar **datasets completos** |
| **Entrada** | Una imagen PNG/JPG | Carpeta con múltiples imágenes |
| **Interacción** | Selección manual de puntos | Un click en slice central |
| **Salida** | Máscara + visualización | Máscaras + Reconstrucción 3D + STL |
| **Uso típico** | Pruebas, imágenes individuales | Volúmenes CT/MRI completos |

---

## 🖼️ Opción 1: Segmentación de Una Sola Imagen

**Script:** `one_segmentation.py`

Ideal para segmentar una imagen individual de forma interactiva con vista previa en tiempo real.

### Configuración

Edita las rutas en el archivo (líneas 17 y 22):

```python
# Ruta al checkpoint de SAM
ckpt = "Checkpoints/sam_vit_b_01ec64.pth"

# Ruta a tu imagen
img = np.array(Image.open("tu_imagen.png").convert("RGB"))
```

### Ejecución

```bash
python one_segmentation.py
```

### Uso Interactivo

1. Se abre una ventana con **dos paneles**: imagen original y vista previa de segmentación
2. **Click derecho**: Agregar punto positivo (⭐ verde) - marca el objeto a segmentar
3. **Click izquierdo**: Agregar punto negativo (❌ rojo) - excluye regiones
4. **Tecla 'z'**: Deshacer último punto
5. **Tecla 'c'**: Limpiar todos los puntos
6. **Cerrar ventana**: Finalizar y ver resultados

### Salida

- Visualización de 6 paneles comparando imagen original, máscara raw y máscara refinada
- Estadísticas en consola (área, score, número de puntos)

---

## 📦 Opción 2: Segmentación de Datasets Completos

**Script:** `segment_sam_propagation.py`

Diseñado para segmentar **volúmenes CT/MRI completos** con propagación automática bidireccional desde la slice central, incluyendo reconstrucción 3D.

### Características

- ✅ Procesa **todas las imágenes** de una carpeta automáticamente
- 🔄 **Propagación bidireccional**: desde la slice central hacia arriba y abajo
- 📊 **Métricas de calidad**: Dice coefficient entre slices consecutivas
- 🎨 **Reconstrucción 3D**: Nube de puntos, contornos y malla sólida
- 💾 **Exportación STL**: Para impresión 3D o software CAD
- 📋 **Resumen estadístico**: Archivo con métricas de cada slice

### Configuración

Edita las rutas en el archivo (líneas 35-37):

```python
# Ruta al checkpoint de SAM
ckpt = "Checkpoints/sam_vit_b_01ec64.pth"

# Carpeta con las imágenes PNG/JPG del volumen
data_dir = "DATA/D1/pngs"

# Carpeta donde se guardarán los resultados
output_dir = "DATA/D1_propagation_results"
```

### Ejecución

```bash
python segment_sam_propagation.py
```

### Flujo de Trabajo

1. **Paso 1**: Se abre la **slice central** del volumen
2. **Paso 2**: Segmentas interactivamente (igual que `one_segmentation.py`)
3. **Paso 3**: Al cerrar, **propaga automáticamente** a todas las demás slices
4. **Paso 4**: Genera **reconstrucción 3D** y exporta modelo STL

### Salida

```
DATA/D1_propagation_results/
├── I01_seg.png                    # Overlay de cada slice
├── I01_mask.png                   # Máscara binaria
├── ...
├── contour_points_3d.npy          # Puntos 3D (numpy)
├── contour_points_3d.csv          # Puntos 3D (CSV)
├── reconstruction_3d_points.png   # Vista 3D nube de puntos
├── reconstruction_3d_contours_*.png
├── solid_mesh_3d_*.png            # Vistas de malla sólida
├── modelo_3d.stl                  # Modelo para impresión 3D
└── propagation_summary.txt        # Estadísticas completas
```

---

## ⚡ Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/ThomasMolina19/medsam-unal-project.git
cd medsam-unal-project

# 2. Instalar dependencias
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Descargar checkpoint SAM ViT-B (~375 MB)
mkdir -p Checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P Checkpoints/

# 4. Ejecutar según tu necesidad
python one_segmentation.py           # Una sola imagen
python segment_sam_propagation.py    # Dataset completo
```

## 🔧 Requisitos

### Sistema
- Python 3.8+
- PyTorch 2.0+
- Dispositivo: CUDA GPU, Apple Silicon (MPS), o CPU

### Librerías Principales
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scipy>=1.10.0
Pillow>=9.5.0
pydicom>=2.4.0
```

## 📦 Instalación Completa

### Paso 1: Crear entorno virtual

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Paso 3: Descargar checkpoint SAM

```bash
mkdir -p Checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P Checkpoints/
```

Otras opciones:
- **ViT-H (Huge)**: [Descargar](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (~2.4 GB)
- **ViT-L (Large)**: [Descargar](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (~1.2 GB)

## 📁 Estructura del Proyecto

```
medsam-unal-project/
├── one_segmentation.py             # 🖼️ Segmentación de UNA imagen
├── segment_sam_propagation.py      # 📦 Segmentación de DATASETS completos
├── requirements.txt
├── README.md
│
├── DCM/                            # Módulo de carga de imágenes
│   └── load_dicom_as_image.py
│
├── Graphics/                       # Módulo de visualización
│   ├── grafication.py              # Reconstrucción 3D, exportación STL
│   └── interface.py                # Interfaz interactiva
│
├── Segmentation/                   # Módulo de segmentación
│   ├── Masks.py                    # Operaciones con máscaras
│   ├── Metrics.py                  # Cálculo de Dice, IoU
│   ├── propagation.py              # Lógica de propagación
│   ├── segment_image.py            # Segmentación con SAM
│   └── negative_points.py
│
├── Checkpoints/                    # Checkpoints de SAM
│   └── sam_vit_b_01ec64.pth
│
└── DATA/                           # Datos de entrada/salida
    ├── D1/pngs/
    ├── D1_propagation_results/
    └── ...
```

## 🖱️ Controles de la Interfaz

| Acción | Control |
|--------|---------|
| Punto positivo (objeto) | Click **derecho** |
| Punto negativo (excluir) | Click **izquierdo** |
| Deshacer | Tecla `z` |
| Limpiar todo | Tecla `c` |
| Finalizar | Cerrar ventana |

## 🖥️ Soporte de Dispositivos

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

- **MPS** (Apple Silicon M1/M2/M3): Detección automática
- **CUDA** (NVIDIA GPU): Cambiar a `device = "cuda"`
- **CPU**: Fallback automático

## 🐛 Solución de Problemas

### "No module named 'segment_anything'"
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### "Checkpoint not found"
```bash
ls -lh Checkpoints/sam_vit_b_01ec64.pth
```

### Segmentación de baja calidad
- Agregar más puntos positivos en el objeto
- Usar puntos negativos para excluir regiones no deseadas

### Errores de memoria
- Usar SAM ViT-B en lugar de ViT-H
- Cerrar otras aplicaciones

## �� Referencias

- **SAM**: Kirillov, A., et al. (2023). "Segment Anything" [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **MedSAM**: Ma, J., et al. (2023). "Segment Anything in Medical Images" [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- **Segment Anything**: https://github.com/facebookresearch/segment-anything

## 👥 Autor

**Thomas Molina Molina**  
Universidad Nacional de Colombia  
Tópicos en Geometría Computacional

## 📝 Licencia

Proyecto de código abierto para uso educativo y de investigación.
