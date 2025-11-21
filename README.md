# 🏥 Interactive MedSAM: Segmentación de Húmero en Imágenes Médicas

Pipeline automático para la detección y segmentación del húmero en secuencias de imágenes médicas utilizando técnicas avanzadas de visión por computador y el modelo MedSAM.

## 🚀 Pipeline de 3 Fases

### 🔍 Fase 1: Mejora de Contraste (CLAHE)
- **Objetivo**: Mejorar la visibilidad del húmero en la imagen
- **Técnica**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Proceso**:
  - Convierte la imagen a escala de grises
  - Aplica ecualización adaptativa de histograma
  - Normaliza la intensidad para resaltar estructuras óseas
  - Resalta el anillo oscuro característico del húmero

### 🎯 Fase 2: Detección Automática del Húmero
- **Objetivo**: Localizar con precisión el húmero en la imagen
- **Técnicas**:
  - Transformada de Hough Circular para detección de círculos
  - Fusión de múltiples candidatos para mejorar precisión
  - Tracking temporal para consistencia entre frames
  - Refinamiento de centro basado en características locales
- **Características evaluadas**:
  - Tamaño del círculo (área entre 1500-8000 píxeles)
  - Contraste del anillo óseo (darkness score)
  - Consistencia temporal (tracking entre frames)
  - Posición relativa en la imagen
- **Algoritmos avanzados**:
  - Corrección backward para el primer frame
  - Filtrado de outliers tipo "sandwich"
  - Fusión de candidatos cercanos (máximo 50px distancia)

### 🖌️ Fase 3: Segmentación con MedSAM
- **Objetivo**: Generar una máscara precisa del húmero
- **Entradas**:
  - Bounding box de la fase 2 (con margen del 40%)
  - Punto central del húmero detectado
- **Proceso**:
  - El modelo MedSAM genera múltiples propuestas de máscaras
  - Selección de la mejor máscara basada en scoring combinado:
    - Score de confianza de SAM (50% del peso)
    - Área similar a la esperada (30% del peso)
    - Circularidad del contorno (20% del peso)
- **Point Prompts**: Usa el centro del húmero como punto positivo para guiar la segmentación

## 📊 Métricas de Rendimiento
- **Tasa de detección**: 100% en secuencias de prueba
- **Score SAM promedio**: 0.547 (54.7% de confianza)
- **Consistencia temporal**: Desplazamiento promedio de 6.9px entre frames
- **Mejora significativa**: Las máscaras ahora capturan más área del húmero

## 🛠️ Instalación y Uso

### Instalación

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Instalar dependencias adicionales
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install gdown

# Descargar modelo MedSAM
mkdir -p checkpoints
gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam_vit_b.pth
```

### Ejecución del Pipeline

```bash
# Pipeline completo automático (detección + segmentación)
python scripts/test_automatic_sam.py

# Detección de húmero con tracking temporal
python scripts/test_temporal_tracking.py

# Debug de candidatos de detección
python scripts/debug_candidates.py
```

## 📁 Estructura del Proyecto

```
interactive-medsam/
├── src/detection/
│   ├── humerus_detector.py        # Algoritmos de detección
│   └── __init__.py
├── scripts/
│   ├── test_automatic_sam.py      # Pipeline completo
│   ├── test_temporal_tracking.py  # Tracking temporal
│   └── debug_candidates.py        # Debug de detección
├── checkpoints/
│   └── medsam_vit_b.pth          # Modelo MedSAM (~2.4GB)
├── test_data/
│   ├── I01.png, I02.png, ...      # Imágenes de prueba
└── test_results/
    └── automatic_sam_pipeline.png # Resultados visuales
```

## 🔧 Scripts Principales

### `test_automatic_sam.py` - Pipeline Completo
Ejecuta las 3 fases automáticamente:
1. Procesa secuencia de imágenes (I01.png a I05.png)
2. Aplica CLAHE para mejora de contraste
3. Detecta húmero con tracking temporal
4. Segmenta con MedSAM usando point prompts
5. Genera visualización comparativa

**Salida**:
- Detección exitosa: 5/5 imágenes
- Score SAM promedio: 0.547
- Archivo: `test_results/automatic_sam_pipeline.png`

### `test_temporal_tracking.py` - Solo Detección
Implementa el sistema de tracking temporal:
- Detección en cada frame individual
- Corrección backward para el primer frame
- Filtrado de outliers
- Consistencia temporal entre frames

### `debug_candidates.py` - Análisis de Detección
Visualiza todos los candidatos de detección:
- Muestra círculos detectados por Hough Transform
- Colores por score de confianza
- Ayuda a entender el proceso de selección

## 🎛️ Parámetros Configurables

### Detección de Húmero
- **Área mínima**: 1500 píxeles (radio ~22px)
- **Área máxima**: 8000 píxeles (radio ~50px)
- **Distancia máxima para fusión**: 50 píxeles
- **Margen del bounding box**: 40%
- **Umbral de score mínimo**: 0.7

### Tracking Temporal
- **Distancia máxima para tracking**: 35 píxeles
- **Ratio máximo de cambio de tamaño**: 0.3
- **Corrección backward**: Activada para primer frame

### Selección de Máscara SAM
- **Peso score SAM**: 50%
- **Peso similitud de área**: 30%
- **Peso circularidad**: 20%

## 📈 Mejoras Implementadas

1. **Detección Robusta**: Fusión de múltiples candidatos cercanos
2. **Tracking Temporal**: Consistencia entre frames secuenciales
3. **Point Prompts**: Guía precisa para SAM usando centro detectado
4. **Selección Inteligente**: Scoring combinado para mejor máscara
5. **Corrección Backward**: Ajuste del primer frame usando frames posteriores

## 🏗️ Arquitectura Técnica

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Imagen RGB    │───▶│   CLAHE          │───▶│   Detección     │
│   Original      │    │   Enhancement    │    │   Hough Circles │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           │
│   Bounding Box  │───▶│   MedSAM         │───▶│   Máscara       │
│   + Point       │    │   Segmentation   │    │   Final         │
│   Prompts       │    └──────────────────┘    └─────────────────┘
└─────────────────┘
```

## 🔬 Resultados Experimentales

Con el pipeline implementado se logra:
- **100% detección exitosa** en secuencias de 5 imágenes
- **Consistencia temporal** con desviación promedio de 6.9 píxeles
- **Máscaras más grandes** que capturan mejor la anatomía del húmero
- **Score SAM mejorado** mediante selección inteligente de máscaras

## 📝 Notas de Implementación

- El pipeline es completamente automático (no requiere intervención manual)
- Optimizado para secuencias de imágenes médicas (MRI, CT, X-ray)
- Compatible con CPU, GPU y Apple Silicon (MPS)
- Resultados guardados en `test_results/` con visualizaciones comparativas

## 📄 Licencia

MIT License

## 👤 Autores

**Thomas Molina Molina** - *Desarrollador principal*

**Gustavo Adolfo Pérez** - *Colaborador*
