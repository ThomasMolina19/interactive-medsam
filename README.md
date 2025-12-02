# Interactive Medical Image Segmentation with SAM & MedSAM

Herramientas interactivas de segmentación de imágenes médicas utilizando **SAM** (Segment Anything Model) y **MedSAM** con preprocesamiento y postprocesamiento avanzado específicamente optimizado para aplicaciones de imagenología médica.

## 🎯 Características Principales

- **Segmentación Basada en Puntos (Tiempo Real)**: Vista previa en vivo con prompts de puntos positivos/negativos
- **Segmentación con Bounding Box**: Interfaz intuitiva para seleccionar regiones de interés
- **Vista Previa en Tiempo Real**: Ver resultados de segmentación instantáneamente
- **Funcionalidad Deshacer/Limpiar**: Corrección fácil con atajos de teclado ('z', 'c')
- **Mejora de Imágenes Médicas**: CLAHE para CT/MRI, ajuste de contraste automático
- **Postprocesamiento Avanzado**: Operaciones morfológicas para refinar máscaras
- **Generación Multi-Máscara**: Genera múltiples propuestas y selecciona la mejor
- **Visualización Completa**: Comparación lado a lado con 6 vistas diferentes
- **Soporte Multi-Dispositivo**: CUDA, MPS (Apple Silicon), y CPU

## ⚡ Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/ThomasMolina19/medsam-unal-project.git
cd medsam-unal-project

# 2. Instalar dependencias
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Descargar checkpoints (ver sección de instalación)
mkdir Checkpoints
# Descargar sam_vit_h_4b8939.pth o medsam_vit_b.pth

# 4. Ejecutar segmentación interactiva
python segment_sam_points.py        # SAM con puntos
python segment_medsam_points.py     # MedSAM con puntos (recomendado)
python segment_one_medsam.py        # MedSAM con bounding box
```

## 🔧 Requisitos

### Sistema
- Python 3.8+
- PyTorch 2.0+
- Dispositivo: CUDA GPU, Apple Silicon (MPS), o CPU

### Librerías Principales
- `torch` - Framework de deep learning
- `segment-anything` - Modelo SAM de Meta
- `numpy` - Operaciones numéricas
- `matplotlib` - Visualización e interfaz interactiva
- `opencv-python` (cv2) - Procesamiento de imágenes
- `scikit-image` - Operaciones morfológicas
- `scipy` - Funciones científicas
- `Pillow` (PIL) - Carga de imágenes
- `pydicom` - Lectura de archivos DICOM (opcional)

## 📦 Installation

### Step 0: Create and activate a virtual environment (recommended)

Using a virtual environment isolates project dependencies and prevents conflicts with system packages. Execute all subsequent commands with the environment activated.

#### macOS / Linux

```bash
# From the repo root
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# (Optional) Update pip
python -m pip install --upgrade pip
```

#### Windows

```cmd
# From the repo root
python -m venv .venv

# Activate the environment
.venv\Scripts\activate

# (Optional) Update pip
python -m pip install --upgrade pip
```

### Step 1: Clone the repository

```bash
git clone https://github.com/ThomasMolina19/interactive-medsam.git
cd interactive-medsam
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Step 4: Download SAM/MedSAM checkpoints

You can use either SAM (standard) or MedSAM (medical-optimized) checkpoints.

#### **Option A: SAM (Segment Anything Model) - Recommended**

Download SAM checkpoints from the official repository:

1. Visit [SAM Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
2. Choose a model size:
   - **ViT-H (Huge)**: Best quality, ~2.4 GB - [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
   - **ViT-L (Large)**: Good balance, ~1.2 GB - [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
   - **ViT-B (Base)**: Faster, ~375 MB - [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

3. Create checkpoints directory and move the file:
   ```bash
   mkdir -p checkpoints
   mv ~/Downloads/sam_vit_*.pth checkpoints/
   ```

#### **Option B: MedSAM (Medical Segment Anything)**

Download the pre-trained MedSAM model checkpoint (~2.4 GB):

#### **Option 1: Direct Download from Official Sources**

1. Visit the [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)
2. Navigate to the "Model Checkpoints" section in the README
3. Download from one of these sources:
   - **Google Drive**: [Download medsam_vit_b.pth](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)
   - **Hugging Face**: [MedSAM Models](https://huggingface.co/wanglab/medsam)

4. Create checkpoints directory and move the file:
   ```bash
   mkdir -p checkpoints
   mv ~/Downloads/medsam_vit_b.pth checkpoints/
   ```

#### **Option 2: Using gdown (Google Drive CLI)**

```bash
# Install gdown
pip install gdown

# Create checkpoints directory
mkdir -p checkpoints

# Download from Google Drive (check MedSAM repo for current file ID)
gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam_vit_b.pth
```

**Note:** The Google Drive file ID may change. Check the [MedSAM repository](https://github.com/bowang-lab/MedSAM) for the current download link.

#### **Option 3: Using Hugging Face Hub**

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download using Python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='wanglab/medsam', filename='medsam_vit_b.pth', local_dir='checkpoints/')"
```

#### **Verify the download:**

```bash
# Check file exists and size (~2.4 GB)
ls -lh checkpoints/medsam_vit_b.pth

# Expected output:
# -rw-r--r--  1 user  staff   2.4G  Oct  3 10:30 checkpoints/medsam_vit_b.pth
```

**Expected checkpoint path structure:**
```
interactive-medsam/
├── checkpoints/
│   └── medsam_vit_b.pth          # ~2.4 GB
├── segment_medical_image.py
├── requirements.txt
└── README.md
```

**Important Notes:**
- The checkpoint file is large (~2.4 GB), ensure you have sufficient disk space
- Download may take several minutes depending on your internet connection
- Always download from official sources to ensure model integrity
- The checkpoint is based on SAM's ViT-B (Vision Transformer Base) architecture

## 🚀 Uso

### Opción 1: SAM con Puntos Interactivos (Tiempo Real) ⭐

**Script:** `segment_sam_points.py`

La forma más interactiva e intuitiva con retroalimentación en tiempo real.

#### Características:
- Modelo: SAM ViT-H (generalista)
- Entrada: Imágenes PNG/JPG
- Mejora: Ajuste de contraste con OpenCV
- Interfaz: Dual-panel con vista previa en vivo

#### Paso 1: Configurar rutas

Edita el script `segment_sam_points.py` y actualiza:

```python
# Línea 2: Ruta al repositorio de SAM (si es necesario)
sys.path.append('path/to/segment-anything')

# Línea 18: Ruta al checkpoint de SAM
ckpt = "Checkpoints/sam_vit_h_4b8939.pth"

# Línea 26: Ruta a tu imagen
img = np.array(Image.open("path.png").convert("RGB"))
```

#### Paso 2: Ejecutar

```bash
python segment_sam_points.py
```

#### Paso 3: Selección interactiva con vista previa en tiempo real

La herramienta abre una **interfaz de doble panel**:

**Panel Izquierdo**: Imagen original donde colocas los puntos
**Panel Derecho**: Vista previa de segmentación en vivo (¡se actualiza instantáneamente!)

**Controles:**
- 🟢 **Click DERECHO**: Agregar punto POSITIVO (marca el objeto deseado)
- 🔴 **Click IZQUIERDO**: Agregar punto NEGATIVO (excluir regiones no deseadas)
- ⌨️ **Tecla 'z'**: Deshacer último punto
- ⌨️ **Tecla 'c'**: Limpiar todos los puntos
- ✅ **ENTER o cerrar ventana**: Finalizar y ver resultados

**Flujo de trabajo:**
1. Click derecho en el objeto a segmentar (ej: hueso, órgano)
2. Ver la segmentación aparecer instantáneamente en el panel derecho
3. Agregar más puntos positivos para refinar
4. Click izquierdo en áreas a excluir si es necesario
5. Usar 'z' para deshacer errores
6. Cerrar cuando estés satisfecho para ver resultados detallados

**Ejemplo:**
```
🎯 Segmentando un húmero:
1. Click derecho en centro del hueso → vista previa instantánea
2. Click derecho en bordes del hueso → refinamiento
3. Click izquierdo en fondo si se incluyó → exclusión
4. Presionar 'z' si cometiste un error
5. Cerrar ventana → ver visualización final con 6 vistas
```

### Opción 2: MedSAM con Puntos Interactivos (Producción) 🏥

**Script:** `segment_medsam_points.py`

Versión robusta y profesional con MedSAM especializado en imágenes médicas.

#### Características:
- Modelo: MedSAM ViT-B (especializado en medicina)
- Carga robusta: `strict=False`, modo evaluación
- Mejora: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Soporte para DICOM con windowing Hounsfield
- Postprocesamiento morfológico avanzado

#### Paso 1: Configurar rutas

Edita el archivo `segment_medsam_points.py`:

```python
# Línea 39: Ruta al checkpoint de MedSAM
CKPT_PATH = "Checkpoints/medsam_vit_b.pth"

# Línea 40: Ruta a tu imagen
IMG_PATH = "path.png"

# Línea 43: (Opcional) Espaciado de píxeles para métricas físicas
PIXEL_SPACING_MM = (0.7, 0.7)  # Para cálculos en mm²
```

#### Paso 2: Ejecutar

```bash
python segment_medsam_points.py
```

#### Paso 3: Interacción

Same dual-panel interface as SAM version:
- Click derecho: puntos positivos (verde)
- Click izquierdo: puntos negativos (rojo)
- 'z': deshacer, 'c': limpiar
- Vista previa en vivo

#### Ventajas de MedSAM:
- Mejor para anatomías complejas
- Entrenado específicamente en imágenes médicas
- Carga robusta del checkpoint
- Métricas físicas (mm²) si hay spacing

### Opción 3: MedSAM con Bounding Box 📦

**Script:** `segment_one_medsam.py`

Segmentación rápida usando selección rectangular.

#### Paso 1: Configurar rutas

```python
# Línea 19: Ruta al checkpoint
ckpt = "path/Checkpoints/medsam_vit_b.pth"

# Línea 30: Ruta a la imagen
img = np.array(Image.open("path.png").convert("RGB"))
```

#### Paso 2: Ejecutar

```bash
python segment_one_medsam.py
```

#### Paso 3: Selección de región

1. **Seleccionar Región**: Se abrirá una ventana con tu imagen
2. **Dibujar Bounding Box**: Click y arrastrar para crear un rectángulo
3. **Ajustar**: Arrastrar los bordes para redimensionar
4. **Confirmar**: Cerrar la ventana cuando estés satisfecho
5. **Resultados**: Ver los resultados en visualización de 6 paneles

## 📊 Comparación de Scripts

| Característica | `segment_sam_points.py` | `segment_medsam_points.py` | `segment_one_medsam.py` |
|----------------|-------------------------|----------------------------|-------------------------|
| **Modelo** | SAM ViT-H | MedSAM ViT-B | MedSAM ViT-B |
| **Entrada** | Puntos interactivos | Puntos interactivos | Bounding box |
| **Mejora** | Contraste OpenCV | CLAHE | Contraste OpenCV |
| **Vista previa** | ✅ Tiempo real | ✅ Tiempo real | ❌ Solo final |
| **Carga robusta** | ❌ | ✅ strict=False | ✅ |
| **DICOM windowing** | ❌ | ✅ Opcional | ❌ |
| **Métricas físicas** | ❌ | ✅ mm² con spacing | ❌ |
| **Mejor para** | Imágenes generales | Imágenes médicas | Segmentación rápida |
| **multimask_output** | True (3 máscaras) | False (1 máscara) | True (3 máscaras) |

## 📊 Output

The tool provides comprehensive visualization:

### Row 1: Original Results
- Original medical image
- Raw MedSAM segmentation with bounding box
- Binary mask (raw output)

### Row 2: Enhanced Results
- Contrast-enhanced image
- Refined segmentation overlay
- Cleaned binary mask

### Console Output
```
🎯 Interactive box selection starting...
✅ Final selected box: [150 200 450 500]
🎯 Segmentation completed on mps
📦 Box coordinates: [150 200 450 500]
📏 Mask area: 45678 pixels
⭐ Best mask score: 0.9845
🎭 Total masks generated: 3
```

## 🏗️ Detalles Técnicos

### Preprocesamiento de Imágenes

#### `segment_sam_points.py` y `segment_one_medsam.py`:
```python
# Ajuste de contraste con OpenCV
img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
```
- **alpha=1.2**: Factor de contraste (multiplicador)
- **beta=10**: Ajuste de brillo (offset)
- Simple y rápido para imágenes generales

#### `segment_medsam_points.py`:
```python
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe_rgb(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convierte a LAB, aplica CLAHE al canal L
    # Mejor para imágenes médicas con detalles finos
```
- **clip_limit=2.0**: Limita la amplificación del contraste
- **tile_grid_size=(8,8)**: Tamaño de las regiones locales
- Adaptativo: cada región se mejora independientemente
- **Opcional**: Función para windowing Hounsfield (DICOM)

### Pipeline de Segmentación

#### Puntos Interactivos (SAM/MedSAM):
1. Carga y preprocesamiento de imagen
2. Configuración del predictor (`predictor.set_image()`)
3. Selección interactiva de puntos (GUI dual-panel)
4. Predicción en tiempo real por cada punto agregado
5. Selección de mejor máscara (score más alto)
6. Postprocesamiento y refinamiento
7. Visualización de 6 vistas comparativas

#### Bounding Box (MedSAM):
1. Carga y preprocesamiento de imagen
2. Selección interactiva de bounding box (GUI)
3. Predicción con box completo (`predictor.predict(box=...)`)
4. Selección de mejor máscara
5. Postprocesamiento
6. Visualización de resultados

### Refinamiento de Máscaras
- **Remoción de objetos pequeños**: Filtra objetos < 500 píxeles
- **Relleno de huecos**: Operaciones morfológicas binarias
- **Suavizado**: Kernel en forma de disco (radio=2)
- **Opening/Closing**: Reducción de ruido y relleno de gaps

## 🖥️ Device Support

The script automatically detects and uses the best available device:

- **MPS** (Apple Silicon M1/M2/M3): Automatic detection
- **CUDA** (NVIDIA GPU): Change line 12 to `device = "cuda"`
- **CPU**: Automatic fallback

## 💾 Saving Results

To save the segmentation mask, uncomment these lines at the end of the script:

```python
refined_mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8))
refined_mask_pil.save("segmentation_result.png")
print("💾 Mask saved as 'segmentation_result.png'")
```

## 📁 Estructura del Proyecto

```
medsam-unal-project/
├── Checkpoints/
│   ├── sam_vit_h_4b8939.pth       # SAM ViT-Huge checkpoint (~2.4 GB)
│   ├── sam_vit_b_01ec64.pth       # SAM ViT-Base checkpoint (~375 MB)
│   └── medsam_vit_b.pth           # MedSAM ViT-B checkpoint (~2.4 GB)
├── DATA/                           # Carpeta de datos (imágenes DICOM/PNG)
│   └── Data/
│       └── HumeroData/
│           └── IM-0008-0016.dcm
├── segment_sam_points.py           # ⭐ SAM con puntos (tiempo real)
├── segment_medsam_points.py        # 🏥 MedSAM con puntos (robusto)
├── segment_one_medsam.py           # 📦 MedSAM con bounding box
├── requirements.txt                # Dependencias de Python
├── README.md                       # Este archivo
└── Latex/                          # (Opcional) Documentación LaTeX
    └── informe_entrega1.tex
```

## 🔍 Funciones Clave

### `interactive_point_selector(img, predictor)`
Segmentación interactiva basada en puntos con vista previa en tiempo real.

**Implementado en:**
- `segment_sam_points.py` (SAM)
- `segment_medsam_points.py` (MedSAM - versión mejorada)

**Características:**
- Interfaz dual-panel (imagen + máscara en vivo)
- Prompts de puntos positivos/negativos
- Retroalimentación instantánea
- Funcionalidad deshacer/limpiar ('z', 'c')
- Visualización de score y área

**Controles:**
- Click derecho: Puntos positivos (estrellas verdes ⭐)
- Click izquierdo: Puntos negativos (X rojas ❌)
- Tecla 'z': Deshacer último punto
- Tecla 'c': Limpiar todos los puntos

### `interactive_box_selector(img)`
Interfaz GUI para selección de región de interés con RectangleSelector de matplotlib.

**Implementado en:**
- `segment_one_medsam.py`

**Características:**
- Visualización de coordenadas en tiempo real
- Cajas redimensionables y arrastrables
- Retroalimentación visual con overlays coloridos
- Modo interactivo (ajustable después de crear)

### `refine_medical_mask(mask)`
Pipeline de postprocesamiento para refinamiento de máscaras.

**Implementado en todos los scripts**

**Operaciones:**
- Remoción de objetos pequeños (`min_size=500`)
- Relleno de huecos (`binary_fill_holes`)
- Suavizado morfológico (opening + closing con `disk(2)`)

## 🎓 Casos de Uso

### Investigación Médica
- **Segmentación de huesos**: Análisis de húmero en imágenes CT
- **Detección de tumores**: Identificación de regiones anormales
- **Análisis cuantitativo**: Mediciones de área, volumen

### Aplicaciones Clínicas
- **Análisis ROI**: Extracción de regiones de interés específicas
- **Herramientas de medición**: Cálculos de área en píxeles o mm²
- **Estudios anatómicos**: Análisis comparativo de estructuras

### Educación
- **Enseñanza de análisis de imágenes médicas**: Demostraciones interactivas
- **Comparación de modelos**: SAM vs MedSAM en casos reales
- **Prototipos rápidos**: Anotación para datasets de entrenamiento

### Medicina de Precisión
- **Segmentación específica del paciente**: Refinamiento con puntos interactivos
- **Planificación quirúrgica**: Identificación precisa de estructuras
- **Seguimiento longitudinal**: Comparación de estudios en el tiempo

## 🆕 Características del Proyecto

### Scripts Disponibles (3 Herramientas)

1. **`segment_sam_points.py`** - SAM Generalista
   - Segmentación con puntos interactivos
   - Vista previa en tiempo real
   - Modelo SAM ViT-H
   - Contraste simple con OpenCV

2. **`segment_medsam_points.py`** - MedSAM Profesional
   - Segmentación con puntos (versión robusta)
   - CLAHE para mejora adaptativa
   - Carga de checkpoint tolerante a errores
   - Soporte opcional para windowing DICOM
   - Métricas físicas (mm²) con pixel spacing

3. **`segment_one_medsam.py`** - Bounding Box Rápido
   - Segmentación con caja rectangular
   - Interfaz de arrastrar y soltar
   - Redimensionable e interactivo
   - Procesamiento más rápido

### Mejoras Implementadas
- ✨ **Segmentación basada en puntos** con vista previa en tiempo real
- 🔄 **Funcionalidad deshacer/limpiar** para corrección fácil
- 📊 **Interfaz dual-panel** para retroalimentación instantánea
- ⌨️ **Atajos de teclado** ('z' para deshacer, 'c' para limpiar)
- 🎯 **Prompts positivos/negativos** para control preciso
- 🚀 **Soporte SAM y MedSAM** en scripts separados
- 🏥 **CLAHE para imágenes médicas** (MedSAM version)
- 🔧 **Carga robusta de checkpoints** con strict=False

## 📚 Referencias

### Papers
- **MedSAM**: Ma, J., et al. (2023). "Segment Anything in Medical Images" [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- **SAM**: Kirillov, A., et al. (2023). "Segment Anything" [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

### Repositorios
- **MedSAM Official**: https://github.com/bowang-lab/MedSAM
- **Segment Anything (SAM)**: https://github.com/facebookresearch/segment-anything
- **Este Proyecto**: https://github.com/ThomasMolina19/medsam-unal-project

### Recursos Adicionales
- **SAM Demo**: https://segment-anything.com/
- **MedSAM Hugging Face**: https://huggingface.co/wanglab/medsam

## 🐛 Solución de Problemas

### "No module named 'segment_anything'"
Instalar SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### "Checkpoint not found" o errores de ruta
Verifica que la ruta del checkpoint coincida con la ubicación del archivo descargado:
- **SAM**: `Checkpoints/sam_vit_h_4b8939.pth`
- **MedSAM**: `Checkpoints/medsam_vit_b.pth`

### Errores al cargar el checkpoint de MedSAM
Si ves "Missing keys" o "Unexpected keys", es normal. El script `segment_medsam_points.py` usa `strict=False` para manejar esto automáticamente.

### MPS no disponible (Mac)
El script automáticamente usará CPU. Para GPU NVIDIA:
```python
device = "cuda"  # Cambiar en línea 13 (sam_points) o línea 49 (medsam_points)
```

### Baja calidad de segmentación
- **Método de puntos**: Agregar más puntos positivos o negativos para excluir regiones
- **Método de box**: Ajustar el bounding box para que se ajuste mejor
- **Preprocesamiento**: 
  - OpenCV: Modificar `alpha` y `beta` (línea 29 en sam_points/one_medsam)
  - CLAHE: Ajustar `clip_limit` y `tile_grid_size` (línea 58 en medsam_points)
- **Postprocesamiento**: Ajustar parámetros en `refine_medical_mask()` (líneas 109-122)

### La segmentación no se actualiza en tiempo real
- Asegúrate de estar haciendo click en el panel izquierdo (panel de imagen)
- Verifica que matplotlib esté en modo interactivo (por defecto)
- Intenta cerrar y reabrir el script
- Verifica que hay al menos un punto positivo (click derecho)

### Los puntos no se colocan
- Usa el botón correcto del mouse:
  - **Click DERECHO** = Positivo (verde)
  - **Click IZQUIERDO** = Negativo (rojo)
- Verifica que estás haciendo click dentro del área de la imagen
- Revisa la consola para mensajes de error

### Error: "unexpected keyword argument 'strict'"
Tu versión de PyTorch es antigua. Actualiza:
```bash
pip install --upgrade torch torchvision
```

### Imagen muy oscura o muy clara después del preprocesamiento
Ajusta los parámetros:
- **OpenCV**: `alpha=1.0, beta=0` (sin cambios)
- **CLAHE**: `clip_limit=1.0` (menos agresivo) o `clip_limit=3.0` (más agresivo)

## 👥 Autores

**Thomas Molina Molina**  
Universidad Nacional de Colombia

---

## 📝 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.