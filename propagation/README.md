# Propagation - Módulo de Segmentación SAM con Propagación

Módulo modular para segmentación de imágenes médicas usando SAM (Segment Anything Model) con propagación automática de centros.

## Descripción

Este módulo permite segmentar una carpeta completa de imágenes médicas (CT, MRI, etc.) de forma semi-automática:

1. El usuario segmenta interactivamente la imagen del medio
2. El algoritmo propaga la segmentación hacia arriba y abajo
3. Usa el centro de cada máscara como punto de referencia para la siguiente imagen
4. Detecta y advierte sobre cambios significativos entre frames

## Estructura del Módulo

```
propagation/
├── __init__.py         # Exports públicos del módulo
├── config.py           # Constantes y configuración
├── ui.py               # Interfaz de usuario (Finder macOS)
├── model.py            # Wrapper del modelo SAM
├── mask_utils.py       # Utilidades para máscaras
├── segmentation.py     # Lógica de segmentación
├── propagation.py      # Lógica de propagación
├── visualization.py    # Visualización y guardado
├── main.py             # Punto de entrada principal
└── README.md           # Este archivo
```

## Instalación

### Dependencias

```bash
pip install numpy pillow matplotlib torch opencv-python scipy scikit-image
```

### SAM (Segment Anything)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Descarga un checkpoint de SAM:
- [vit_b (default)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## Uso

### Ejecución Principal

```bash
cd /path/to/interactive-medsam
python -m propagation.main
```

Se abrirá el Finder para seleccionar:
1. **Checkpoint SAM** (.pth)
2. **Carpeta de imágenes** (JPG/PNG)
3. **Carpeta de salida**

### Uso como Biblioteca

```python
from propagation import (
    SAMModel,
    get_user_paths,
    segment_with_point,
    propagate_direction,
    calculate_dice_coefficient,
    save_segmentation_result
)

# Cargar modelo
model = SAMModel("path/to/checkpoint.pth")

# Segmentar con un punto
mask, score = segment_with_point(model, image, [x, y])

# Calcular métricas
dice = calculate_dice_coefficient(mask1, mask2)
```

## Módulos

### `config.py`

Constantes configurables:

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `SIMILARITY_THRESHOLD` | 0.20 | Umbral para advertencias leves (20%) |
| `WARNING_THRESHOLD` | 0.30 | Umbral para advertencias severas (30%) |
| `IMAGE_EXTENSIONS` | `['*.jpg', ...]` | Extensiones soportadas |
| `RETRY_OFFSETS` | `[(0,0), (-10,0), ...]` | Offsets para reintentos |
| `MIN_MASK_SIZE` | 500 | Tamaño mínimo de máscara (px) |

### `model.py`

```python
class SAMModel:
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b")
    def set_image(self, image)
    def predict(self, point_coords, point_labels, multimask_output=True)
```

### `mask_utils.py`

| Función | Descripción |
|---------|-------------|
| `refine_medical_mask(mask)` | Limpia y suaviza máscara |
| `calculate_mask_center(mask)` | Calcula centroide [x, y] |
| `calculate_dice_coefficient(m1, m2)` | Coeficiente Dice (0-1) |
| `calculate_iou(m1, m2)` | Intersection over Union (0-1) |
| `calculate_negative_point(mask, center)` | Punto negativo fuera de máscara |

### `segmentation.py`

| Función | Descripción |
|---------|-------------|
| `segment_with_point(model, img, point)` | Segmenta con 1 punto |
| `segment_with_points(model, img, pos, neg)` | Segmenta con múltiples puntos |
| `segment_with_retry(model, img, center, offsets)` | Segmenta con reintentos |

### `propagation.py`

| Función | Descripción |
|---------|-------------|
| `read_image_file(filepath)` | Lee imagen como RGB array |
| `propagate_direction(model, files, ...)` | Propaga en una dirección |

### `visualization.py`

| Función | Descripción |
|---------|-------------|
| `save_segmentation_result(...)` | Guarda visualización PNG |
| `save_summary(...)` | Guarda resumen TXT |
| `print_final_summary(...)` | Imprime resumen en consola |

## Flujo de Trabajo

```
┌─────────────────────────────────────────────────────────────┐
│                    1. CONFIGURACIÓN                         │
│  Usuario selecciona: checkpoint, carpeta imgs, carpeta out  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              2. SEGMENTACIÓN INICIAL                        │
│  Usuario segmenta imagen del medio con segment_sam_points   │
│  Se calcula centro de la máscara                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  3a. PROPAGAR ARRIBA    │     │  3b. PROPAGAR ABAJO     │
│  (backward)             │     │  (forward)              │
│                         │     │                         │
│  Para cada imagen:      │     │  Para cada imagen:      │
│  • Segmentar con centro │     │  • Segmentar con centro │
│  • Calcular Dice/IoU    │     │  • Calcular Dice/IoU    │
│  • Si diff > 30%:       │     │  • Si diff > 30%:       │
│    reintentar con       │     │    reintentar con       │
│    punto negativo       │     │    punto negativo       │
│  • Actualizar centro    │     │  • Actualizar centro    │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    4. RESULTADOS                            │
│  • Visualizaciones PNG por imagen                           │
│  • propagation_summary.txt con estadísticas                 │
└─────────────────────────────────────────────────────────────┘
```

## Salida

El módulo genera en la carpeta de salida:

- **`{nombre}_seg.png`**: Visualización por cada imagen con:
  - Panel izquierdo: Imagen original
  - Panel central: Overlay con puntos marcados
  - Panel derecho: Máscara binaria

- **`propagation_summary.txt`**: Resumen con:
  - Configuración usada
  - Estadísticas de Dice (promedio, min, max)
  - Lista de imágenes con advertencias

## Ejemplo de Salida

```
SAM Complete Folder Segmentation Summary
======================================================================
Dataset: /path/to/images
Checkpoint: /path/to/sam_vit_b.pth
Output: /path/to/output
Middle image: slice_050.png (index 51)
Total images: 100
Successfully segmented: 100
Images with warnings: 3
Images with severe warnings: 1

Dice Statistics:
  - Average: 0.945
  - Min: 0.712
  - Max: 0.998
======================================================================
```

## Controles (segment_sam_points.py)

Durante la segmentación inicial interactiva:

| Acción | Control |
|--------|---------|
| Punto positivo | Click derecho |
| Punto negativo | Click izquierdo |
| Deshacer | Tecla `z` |
| Limpiar todo | Tecla `c` |
| Guardar y continuar | Cerrar ventana |

## Dispositivos Soportados

El módulo detecta automáticamente:
- **MPS** (Apple Silicon) - Preferido en Mac
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)

## Licencia

MIT License
