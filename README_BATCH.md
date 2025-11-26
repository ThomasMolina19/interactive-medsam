# 🔬 Batch DICOM Segmentation with SAM

Este script permite procesar **múltiples archivos DICOM** de una carpeta usando Segment Anything Model (SAM) para segmentación médica interactiva.

## 🌟 Características

- ✅ **Procesamiento por lotes** de carpetas completas de DICOMs
- ✅ **Segmentación interactiva** punto por punto
- ✅ **Saltar imágenes** no deseadas con tecla 's'  
- ✅ **Múltiples formatos** de salida (máscara, overlay, original)
- ✅ **Metadatos detallados** en JSON para cada imagen
- ✅ **Resumen completo** del procesamiento
- ✅ **Manejo de errores** robusto

## 📋 Requisitos

### Dependencias Python
```bash
pip install -r batch_requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### O usar el script de instalación
```bash
chmod +x setup_batch.sh
./setup_batch.sh
```

## 🚀 Uso Rápido

1. **Configurar rutas** en `batch_dicom_segmentation.py`:
   ```python
   DICOM_FOLDER = "/ruta/a/tus/dicom/"
   CHECKPOINT_PATH = "/ruta/al/modelo/sam_vit_h_4b8939.pth"
   OUTPUT_FOLDER = "batch_segmentation_results"
   ```

2. **Ejecutar**:
   ```bash
   python batch_dicom_segmentation.py
   ```

## 🎯 Controles Interactivos

| Acción | Control | Descripción |
|--------|---------|-------------|
| **Punto Positivo** | Click Derecho | Marca región de interés ✅ |
| **Punto Negativo** | Click Izquierdo | Excluye región ❌ |
| **Deshacer** | Tecla `z` | Elimina último punto |
| **Limpiar** | Tecla `c` | Borra todos los puntos |
| **Saltar imagen** | Tecla `s` | Pasa a la siguiente imagen |
| **Continuar** | Cerrar ventana | Procesa la imagen actual |

## 📁 Estructura de Salida

Para cada DICOM procesado se generan:

```
batch_segmentation_results/
├── IM-0008-0011_mask.png          # Máscara binaria
├── IM-0008-0011_overlay.png       # Imagen con overlay
├── IM-0008-0011_original.png      # Imagen original
├── IM-0008-0011_info.json         # Metadatos detallados
├── ...
└── processing_summary.json        # Resumen completo
```

## 📊 Archivo de Información (JSON)

Cada imagen procesada genera un archivo `_info.json` con:

```json
{
  "filename": "IM-0008-0011.dcm",
  "processing_date": "2025-11-10T...",
  "positive_points": [[x1, y1], [x2, y2]],
  "negative_points": [[x3, y3]],
  "mask_area_pixels": 15420,
  "image_dimensions": [512, 512],
  "score": 0.8945,
  "files_generated": {
    "mask": "IM-0008-0011_mask.png",
    "overlay": "IM-0008-0011_overlay.png",
    "original": "IM-0008-0011_original.png"
  }
}
```

## 📈 Resumen de Procesamiento

Al final se genera `processing_summary.json`:

```json
{
  "processing_date": "2025-11-10T...",
  "total_files": 20,
  "processed_files": [...],
  "skipped_files": ["image1.dcm", "image2.dcm"],
  "failed_files": [...]
}
```

## 🔧 Personalización

### Cambiar carpetas
```python
DICOM_FOLDER = "/tu/carpeta/dicom"
OUTPUT_FOLDER = "mis_resultados"
```

### Modificar post-procesamiento
```python
def refine_medical_mask(mask):
    # Personaliza la limpieza de máscaras
    mask_clean = morphology.remove_small_objects(mask, min_size=1000)  # Cambiar tamaño mínimo
    # ... más modificaciones
    return mask_clean
```

## 🩺 Optimización para Imágenes Médicas

El script incluye mejoras específicas para DICOMs:

- **Normalización DICOM**: Usa ventanas DICOM si están disponibles
- **Mejora de contraste**: Aplicación automática para mejor visualización
- **Limpieza de máscaras**: Elimina objetos pequeños y rellena huecos
- **Suavizado morfológico**: Mejora bordes de segmentación

## ⚡ Consejos de Uso

1. **Para carpetas grandes**: El procesamiento es secuencial, puedes interrumpir con Ctrl+C
2. **Memoria**: El modelo SAM usa ~2.4GB de VRAM/RAM
3. **Velocidad**: Aproximadamente 1-2 minutos por imagen (dependiendo de complejidad)
4. **Calidad**: Más puntos = mejor segmentación, pero más tiempo

## 🐛 Solución de Problemas

### Error: "No DICOM files found"
- Verifica que la carpeta contenga archivos `.dcm`, `.DCM`, `.dicom`, o `.DICOM`
- Revisa los permisos de la carpeta

### Error: "Failed to load DICOM"
- Archivo DICOM corrupto o formato no estándar
- Prueba con otro visor DICOM para verificar

### Error: "Checkpoint not found"
- Descarga el checkpoint SAM desde [aquí](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- Verifica la ruta en `CHECKPOINT_PATH`

### Performance lento
- Usa GPU si está disponible (CUDA/MPS)
- Reduce `min_size` en `refine_medical_mask()` para máscaras más simples

## 📚 Archivos Relacionados

- `segment_sam_points.py` - Versión para imagen única
- `batch_requirements.txt` - Dependencias Python
- `setup_batch.sh` - Script de instalación automática

## 🤝 Contribuciones

¡Mejoras y sugerencias son bienvenidas! Especialmente para:
- Mejores algoritmos de post-procesamiento
- Soporte para más formatos médicos
- Optimizaciones de rendimiento