# 🏥 Medical Image Segmentation with SAM - Volume Propagation

Herramienta de segmentación de volúmenes médicos (CT/MRI) utilizando **SAM** (Segment Anything Model) con propagación automática bidireccional desde la slice central.

## 🎯 Características Principales

- **Segmentación de Volúmenes Completos**: Un solo click en la slice central propaga a todas las demás
- **Propagación Bidireccional**: Hacia arriba y abajo desde la slice central
- **Vista Previa en Tiempo Real**: Ver resultados de segmentación instantáneamente
- **Reconstrucción 3D**: Genera malla sólida, nube de puntos y contornos
- **Exportación STL**: Para impresión 3D o software CAD
- **Métricas de Calidad**: Dice coefficient, IoU, tasa de éxito
- **Soporte DICOM**: Carga automática con windowing Hounsfield
- **Soporte Multi-Dispositivo**: CUDA, MPS (Apple Silicon), y CPU

## ⚡ Quick Start

\`\`\`bash
# 1. Clonar el repositorio
git clone https://github.com/ThomasMolina19/medsam-unal-project.git
cd medsam-unal-project

# 2. Instalar dependencias
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Descargar checkpoint SAM ViT-B (~375 MB)
mkdir -p Checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P Checkpoints/

# 4. Ejecutar segmentación
python segment_sam_propagation.py
\`\`\`

## 🔧 Requisitos

### Sistema
- Python 3.8+
- PyTorch 2.0+
- Dispositivo: CUDA GPU, Apple Silicon (MPS), o CPU

### Librerías Principales
\`\`\`
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scipy>=1.10.0
Pillow>=9.5.0
pydicom>=2.4.0
\`\`\`

## 📦 Instalación Completa

### Paso 1: Crear entorno virtual

\`\`\`bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
\`\`\`

### Paso 2: Instalar dependencias

\`\`\`bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
\`\`\`

### Paso 3: Descargar checkpoint SAM

Descarga SAM ViT-B (recomendado, ~375 MB):

\`\`\`bash
mkdir -p Checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P Checkpoints/
\`\`\`

Otras opciones disponibles:
- **ViT-H (Huge)**: [Descargar](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (~2.4 GB)
- **ViT-L (Large)**: [Descargar](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (~1.2 GB)

## 🚀 Uso

### Paso 1: Configurar rutas

Edita \`segment_sam_propagation.py\` (líneas 35-37):

\`\`\`python
# Ruta al checkpoint de SAM
ckpt = "Checkpoints/sam_vit_b_01ec64.pth"

# Carpeta con las imágenes PNG/JPG del volumen
data_dir = "DATA/D1/pngs"

# Carpeta donde se guardarán los resultados
output_dir = "DATA/D1_propagation_results"
\`\`\`

### Paso 2: Configurar umbrales (opcional)

\`\`\`python
# Líneas 42-43
SIMILARITY_THRESHOLD = 0.25  # Advertencia leve (25% diferencia)
WARNING_THRESHOLD = 0.35     # Advertencia severa (35% diferencia)
\`\`\`

### Paso 3: Ejecutar

\`\`\`bash
python segment_sam_propagation.py
\`\`\`

### Paso 4: Interacción

1. Se abre una ventana con la **slice central** del volumen
2. **Click derecho**: Agregar punto positivo (⭐ verde) en el objeto a segmentar
3. **Click izquierdo**: Agregar punto negativo (❌ rojo) para excluir regiones
4. **Tecla 'z'**: Deshacer último punto
5. **Tecla 'c'**: Limpiar todos los puntos
6. **Cerrar ventana**: Iniciar propagación automática

### Paso 5: Esperar resultados

La propagación procesará todas las slices automáticamente y generará:
- Máscaras binarias de cada slice
- Overlays con la segmentación
- Reconstrucción 3D (nube de puntos, contornos, malla sólida)
- Modelo STL para impresión 3D
- Resumen estadístico

## 📊 Salida

\`\`\`
DATA/D1_propagation_results/
├── I01_seg.png                    # Overlay de segmentación
├── I01_mask.png                   # Máscara binaria
├── ...
├── contour_points_3d.npy          # Puntos 3D (numpy)
├── contour_points_3d.csv          # Puntos 3D (CSV)
├── reconstruction_3d_points.png   # Vista 3D nube de puntos
├── reconstruction_3d_contours_*.png  # Vistas de contornos
├── solid_mesh_3d_*.png            # Vistas de malla sólida
├── modelo_3d.stl                  # Modelo para impresión 3D
└── propagation_summary.txt        # Estadísticas completas
\`\`\`

### Métricas Generadas

- **Dice coefficient**: Similitud entre slices consecutivas
- **IoU**: Intersection over Union
- **Score**: Confianza del modelo SAM
- **Área**: Tamaño de la máscara en píxeles
- **Tasa de éxito**: % de slices segmentadas correctamente

## 📁 Estructura del Proyecto

\`\`\`
medsam-unal-project/
├── segment_sam_propagation.py      # 🔄 Script principal
├── requirements.txt                # Dependencias
├── README.md                       # Documentación
│
├── DCM/                            # Módulo de carga de imágenes
│   └── load_dicom_as_image.py      # Soporte DICOM, PNG, JPG
│
├── Graphics/                       # Módulo de visualización
│   ├── grafication.py              # Reconstrucción 3D, exportación STL
│   └── interface.py                # Interfaz interactiva con puntos
│
├── Segmentation/                   # Módulo de segmentación
│   ├── Masks.py                    # Operaciones con máscaras
│   ├── Metrics.py                  # Cálculo de Dice, IoU
│   ├── propagation.py              # Lógica de propagación
│   ├── segment_image.py            # Segmentación con SAM
│   └── negative_points.py          # Cálculo de puntos negativos
│
├── Checkpoints/                    # Checkpoints de SAM
│   └── sam_vit_b_01ec64.pth        # SAM ViT-B (~375 MB)
│
└── DATA/                           # Datos de entrada/salida
    ├── D1/pngs/                    # Volumen 1 (imágenes PNG)
    ├── D1_propagation_results/     # Resultados del volumen 1
    └── ...
\`\`\`

## 🔍 Módulos del Proyecto

### \`DCM/load_dicom_as_image.py\`
- \`load_dicom_as_image()\`: Carga DICOM con windowing Hounsfield
- \`read_image_file()\`: Carga PNG/JPG como array RGB
- \`get_dataset_files()\`: Obtiene lista ordenada de archivos del volumen

### \`Graphics/grafication.py\`
- \`extract_contour_points_3d()\`: Extrae puntos de contornos con coordenada Z
- \`plot_3d_contours()\`: Visualización 3D como nube de puntos
- \`plot_3d_contours_by_slice()\`: Contornos 3D coloreados por slice
- \`plot_3d_solid_mesh()\`: Genera malla sólida 3D
- \`export_mesh_to_stl()\`: Exporta a formato STL

### \`Graphics/interface.py\`
- \`interactive_sam_point_selector()\`: Interfaz de selección de puntos con vista previa en tiempo real

### \`Segmentation/Masks.py\`
- \`refine_medical_mask()\`: Postprocesamiento morfológico
- \`calculate_mask_center()\`: Calcula centroide de la máscara
- \`find_mask_contours()\`: Encuentra contornos con OpenCV
- \`save_segmentation_result()\`: Guarda visualización con overlay

### \`Segmentation/propagation.py\`
- \`propagate_segmentation()\`: Propaga segmentación hacia arriba/abajo

### \`Segmentation/segment_image.py\`
- \`segment_image()\`: Segmentación con múltiples puntos
- \`segment_with_point()\`: Segmentación con un solo punto
- \`segment_first_image()\`: Segmentación interactiva de la primera imagen

### \`Segmentation/Metrics.py\`
- \`dice_coefficient()\`: Calcula coeficiente Dice entre máscaras
- \`iou_score()\`: Calcula Intersection over Union

## ��️ Soporte de Dispositivos

El script detecta automáticamente el mejor dispositivo disponible:

\`\`\`python
device = "mps" if torch.backends.mps.is_available() else "cpu"
\`\`\`

- **MPS** (Apple Silicon M1/M2/M3): Detección automática
- **CUDA** (NVIDIA GPU): Cambiar a \`device = "cuda"\`
- **CPU**: Fallback automático

## 🐛 Solución de Problemas

### "No module named 'segment_anything'"
\`\`\`bash
pip install git+https://github.com/facebookresearch/segment-anything.git
\`\`\`

### "Checkpoint not found"
Verifica que el archivo existe:
\`\`\`bash
ls -lh Checkpoints/sam_vit_b_01ec64.pth
\`\`\`

### Segmentación de baja calidad
- Agregar más puntos positivos en el objeto
- Usar puntos negativos para excluir regiones no deseadas
- Ajustar umbrales \`SIMILARITY_THRESHOLD\` y \`WARNING_THRESHOLD\`

### MPS no disponible (Mac)
El script automáticamente usará CPU. Para verificar:
\`\`\`python
import torch
print(torch.backends.mps.is_available())
\`\`\`

### Errores de memoria
- Usar SAM ViT-B en lugar de ViT-H
- Cerrar otras aplicaciones
- Reducir el tamaño de las imágenes de entrada

## 📚 Referencias

### Papers
- **SAM**: Kirillov, A., et al. (2023). "Segment Anything" [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **MedSAM**: Ma, J., et al. (2023). "Segment Anything in Medical Images" [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)

### Repositorios
- **Segment Anything (SAM)**: https://github.com/facebookresearch/segment-anything
- **MedSAM**: https://github.com/bowang-lab/MedSAM
- **Este Proyecto**: https://github.com/ThomasMolina19/medsam-unal-project

## 👥 Autor

**Thomas Molina Molina**  
Universidad Nacional de Colombia  
Tópicos en Geometría Computacional

## 📝 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.
