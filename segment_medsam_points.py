#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interfaz interactiva de puntos para segmentación con MedSAM/SAM.
Mejoras incluidas:
- Carga robusta del checkpoint (strict=False), eval(), no_grad()
- Selección automática de dispositivo (cuda > mps > cpu)
- Preprocesado con CLAHE (opcional windowing para DICOM si lo necesitas)
- Tipos correctos para predictor (float32 / int64)
- Opción por defecto multimask_output=False (más estable en MedSAM)
- Postproceso booleano (remove_small_objects, fill_holes, opening/closing)
- Reporte de métricas y visualización mejorada

Nota:
- Ajusta las rutas de `CKPT_PATH` e `IMG_PATH` a tus archivos.
- Si trabajas con DICOM, aplica windowing antes de convertir a RGB (se incluye una
  función auxiliar de ejemplo comentada).
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
from skimage import morphology
import cv2

# --- SAM / MedSAM ---
# Asegúrate de tener el repo de "segment-anything" en tu PYTHONPATH o añade su ruta:
# sys.path.append('path/to/segment-anything')
from segment_anything import sam_model_registry, SamPredictor

# =========================
# Configuración del usuario
# =========================
CKPT_PATH = "Checkpoints/medsam_vit_b.pth"
IMG_PATH  = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/medsam-unal-project/dicom_pngs/I13.png"

# Si tienes spacing físico del píxel (p. ej., de DICOM), puedes rellenarlo aquí para reportar área física
PIXEL_SPACING_MM = (None, None)  # (dx_mm, dy_mm), por ejemplo (0.7, 0.7). Déjalo en None si no aplica.

# ===========
# Dispositivo
# ===========
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  Usando dispositivo: {device}")

# ===========================
# Utilidades de preprocesado
# ===========================
def apply_clahe_rgb(img_rgb: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Aplica CLAHE al canal L en espacio LAB y devuelve RGB uint8.
    Espera img_rgb en uint8 [0-255].
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out

# Ejemplo de windowing para CT/MR (si cargas DICOM crudo). Mantener comentado si no lo usas.
# def window_image_hu_to_uint8(img_hu: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
#     """
#     Convierte intensidades HU a uint8 aplicando windowing.
#     """
#     low = window_center - window_width / 2.0
#     high = window_center + window_width / 2.0
#     img = np.clip(img_hu, low, high)
#     img = (img - low) / (high - low + 1e-6)  # [0,1]
#     img_uint8 = (img * 255).astype(np.uint8)
#     # Si el predictor requiere RGB, duplicamos canales:
#     return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

# ========================
# Carga del modelo MedSAM
# ========================
def load_medsam_vit_b(ckpt_path: str, device_str: str = "cpu"):
    # Crea el modelo base ViT-B del SAM
    sam = sam_model_registry["vit_b"]()
    # Carga de estado con tolerancia a llaves (por si el checkpoint difiere mínimamente)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = sam.load_state_dict(state, strict=False)
    if missing:
        print(f"ℹ️  Missing keys: {len(missing)} (no crítico si el modelo funciona)")
    if unexpected:
        print(f"ℹ️  Unexpected keys: {len(unexpected)} (no crítico si el modelo funciona)")

    sam.eval()
    sam = sam.to(device_str)
    torch.set_grad_enabled(False)
    return sam

# ============================
# Postproceso de máscara (bool)
# ============================
def refine_medical_mask(mask: np.ndarray,
                        min_size: int = 500,
                        disk_radius: int = 2) -> np.ndarray:
    """
    Limpia la máscara binaria con operaciones morfológicas.
    - mask: array booleano o 0/1
    """
    mask = mask.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = ndimage.binary_fill_holes(mask)
    selem = morphology.disk(disk_radius)
    mask = morphology.binary_opening(mask, selem)
    mask = morphology.binary_closing(mask, selem)
    return mask

# =========================================
# Lógica de selección interactiva de puntos
# =========================================
def interactive_point_selector(img_rgb_uint8: np.ndarray, predictor: SamPredictor):
    """
    Selector de puntos interactivo con previsualización en tiempo real.
    - Click derecho: POSITIVO (g*)
    - Click izquierdo: NEGATIVO (rx)
    - Tecla 'z': deshacer último punto
    - Tecla 'c': limpiar todos los puntos
    - ENTER/ESC o cerrar ventana: finalizar
    """
    class PointSelector:
        def __init__(self, ax_img, ax_mask):
            self.positive_points = []
            self.negative_points = []
            self.ax_img = ax_img
            self.ax_mask = ax_mask
            self.point_markers = []
            self.mask_display = None

        def update_segmentation(self):
            """Actualiza segmentación en tiempo real."""
            if self.mask_display is not None:
                try:
                    self.mask_display.remove()
                except Exception:
                    pass
                self.mask_display = None

            if len(self.positive_points) == 0 and len(self.negative_points) == 0:
                self.ax_mask.clear()
                self.ax_mask.imshow(img_rgb_uint8)
                self.ax_mask.set_title("Máscara (agrega puntos para ver)")
                self.ax_mask.axis('off')
                fig.canvas.draw()
                return

            # Preparar puntos y etiquetas (SAM espera (x,y) en pixeles float32)
            input_points = []
            input_labels = []
            for p in self.positive_points:
                input_points.append(p)
                input_labels.append(1)
            for p in self.negative_points:
                input_points.append(p)
                input_labels.append(0)

            input_points = np.array(input_points, dtype=np.float32)
            input_labels = np.array(input_labels, dtype=np.int64)

            # Predicción (por defecto 1 máscara estable)
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False
                )
                best_mask = masks[0]
                best_score = float(scores[0])

                self.ax_mask.clear()
                self.ax_mask.imshow(img_rgb_uint8)
                self.mask_display = self.ax_mask.imshow(best_mask, alpha=0.6, cmap='Blues')

                # Pinta puntos también en la vista de máscara
                for p in self.positive_points:
                    self.ax_mask.plot(p[0], p[1], 'g*', markersize=15, markeredgewidth=2)
                for p in self.negative_points:
                    self.ax_mask.plot(p[0], p[1], 'rx', markersize=12, markeredgewidth=3)

                area_px = np.sum(best_mask)
                title = f"Segmentación | Score: {best_score:.3f} | Área: {area_px:.0f} px"
                # Si hay spacing físico, añade área en mm²
                dx_mm, dy_mm = PIXEL_SPACING_MM
                if dx_mm is not None and dy_mm is not None:
                    area_mm2 = area_px * dx_mm * dy_mm
                    title += f" | Área: {area_mm2:.1f} mm²"

                self.ax_mask.set_title(title)
                self.ax_mask.axis('off')

            except Exception as e:
                print(f"⚠️ Error en segmentación: {e}")

            fig.canvas.draw()

        def onclick(self, event):
            if event.inaxes != self.ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return

            x, y = float(event.xdata), float(event.ydata)

            if event.button == 1:
                # Izquierdo = NEGATIVO
                self.negative_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_markers.append(('neg', marker))
                print(f"❌ Punto NEGATIVO agregado: ({x:.0f}, {y:.0f})")

            elif event.button == 3:
                # Derecho = POSITIVO
                self.positive_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'g*', markersize=20, markeredgewidth=2)[0]
                self.point_markers.append(('pos', marker))
                print(f"✅ Punto POSITIVO agregado: ({x:.0f}, {y:.0f})")

            self.ax_img.set_title(
                f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar"
            )
            self.update_segmentation()

        def onkey(self, event):
            if event.key == 'z':
                if len(self.point_markers) > 0:
                    point_type, marker = self.point_markers.pop()
                    try:
                        marker.remove()
                    except Exception:
                        pass

                    if point_type == 'pos' and len(self.positive_points) > 0:
                        removed = self.positive_points.pop()
                        print(f"↩️  Deshecho punto POSITIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    elif point_type == 'neg' and len(self.negative_points) > 0:
                        removed = self.negative_points.pop()
                        print(f"↩️  Deshecho punto NEGATIVO: ({removed[0]:.0f}, {removed[1]:.0f})")

                    self.ax_img.set_title(
                        f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar"
                    )
                    self.update_segmentation()

            elif event.key == 'c':
                for _, marker in self.point_markers:
                    try:
                        marker.remove()
                    except Exception:
                        pass
                self.point_markers.clear()
                self.positive_points.clear()
                self.negative_points.clear()
                print("🧹 Todos los puntos limpiados")
                self.ax_img.set_title(f"✅ Positivos: 0 | ❌ Negativos: 0 | 'z': deshacer | 'c': limpiar")
                self.update_segmentation()

    # Ventana con 2 subplots (imagen y previsualización de máscara)
    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(20, 8))

    ax_img.imshow(img_rgb_uint8)
    ax_img.set_title("🎯 Imagen | Click derecho = POSITIVO | Click izquierdo = NEGATIVO")
    ax_img.axis('off')

    ax_mask.imshow(img_rgb_uint8)
    ax_mask.set_title("Segmentación (agrega puntos para ver)")
    ax_mask.axis('off')

    selector_obj = PointSelector(ax_img, ax_mask)

    fig.canvas.mpl_connect('button_press_event', selector_obj.onclick)
    fig.canvas.mpl_connect('key_press_event', selector_obj.onkey)

    plt.figtext(
        0.5, 0.02,
        "🟢 Derecho: Positivo | 🔴 Izquierdo: Negativo | ⌨️ 'z': Deshacer | 'c': Limpiar | ENTER/ESC: Terminar",
        ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="blue", linewidth=2)
    )

    plt.tight_layout()
    plt.show()

    return selector_obj.positive_points, selector_obj.negative_points

# ======================
# Programa principal (UI)
# ======================
def main():
    # ---- Carga de modelo ----
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"No se encontró el checkpoint en: {CKPT_PATH}")

    sam = load_medsam_vit_b(CKPT_PATH, device)
    predictor = SamPredictor(sam)

    # ---- Carga de imagen ----
    if not os.path.isfile(IMG_PATH):
        raise FileNotFoundError(f"No se encontró la imagen en: {IMG_PATH}")

    # Si es PNG/JPG: cargar y asegurar RGB uint8
    img = np.array(Image.open(IMG_PATH).convert("RGB"))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Preprocesado (CLAHE)
    img_enhanced = apply_clahe_rgb(img)

    # Configurar imagen para predictor (HWC, RGB, uint8)
    predictor.set_image(img_enhanced)

    H, W = img.shape[:2]
    print(f"🖼️  Tamaño imagen: {W}x{H}")

    # ---- Interacción de puntos ----
    print("🎯 Selección de puntos iniciando...")
    print("   - Click DERECHO: Marca puntos POSITIVOS (objeto de interés)")
    print("   - Click IZQUIERDO: Marca puntos NEGATIVOS (para omitir contornos)")
    print("   - Tecla 'z': Deshacer último punto")
    print("   - Tecla 'c': Limpiar todos los puntos")

    positive_points, negative_points = interactive_point_selector(img_enhanced, predictor)

    # Preparar arrays para predictor
    input_points = []
    input_labels = []

    for p in positive_points:
        input_points.append(p)
        input_labels.append(1)
    for p in negative_points:
        input_points.append(p)
        input_labels.append(0)

    if len(input_points) == 0:
        print("⚠️ No se seleccionaron puntos. Saliendo...")
        sys.exit(0)

    input_points = np.array(input_points, dtype=np.float32)
    input_labels = np.array(input_labels, dtype=np.int64)

    print(f"✅ Total de puntos: {len(input_points)}")
    print(f"   - Positivos: {len(positive_points)}")
    print(f"   - Negativos: {len(negative_points)}")

    # ---- Predicción final (una sola máscara estable por defecto) ----
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )
    best_mask = masks[0]
    best_score = float(scores[0])

    # ---- Postproceso ----
    refined_mask = refine_medical_mask(best_mask, min_size=500, disk_radius=2)

    # ---- Visualización de resultados ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Imagen Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_enhanced)
    axes[0, 1].imshow(best_mask, alpha=0.5, cmap='Reds')
    for p in positive_points:
        axes[0, 1].plot(p[0], p[1], 'g*', markersize=15, markeredgewidth=2)
    for p in negative_points:
        axes[0, 1].plot(p[0], p[1], 'rx', markersize=12, markeredgewidth=3)
    axes[0, 1].set_title("SAM Output (puntos)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(best_mask, cmap='gray')
    axes[0, 2].set_title("Máscara cruda")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(img_enhanced)
    axes[1, 0].set_title("Imagen Mejorada (CLAHE)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_enhanced)
    axes[1, 1].imshow(refined_mask, alpha=0.5, cmap='Blues')
    for p in positive_points:
        axes[1, 1].plot(p[0], p[1], 'g*', markersize=15, markeredgewidth=2)
    for p in negative_points:
        axes[1, 1].plot(p[0], p[1], 'rx', markersize=12, markeredgewidth=3)
    axes[1, 1].set_title("Segmentación Refinada")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(refined_mask, cmap='gray')
    axes[1, 2].set_title("Máscara Refinada")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # ---- Resumen ----
    area_px = int(np.sum(refined_mask))
    dx_mm, dy_mm = PIXEL_SPACING_MM
    area_mm2_str = ""
    if dx_mm is not None and dy_mm is not None:
        area_mm2 = area_px * dx_mm * dy_mm
        area_mm2_str = f" | Área: {area_mm2:.1f} mm²"

    print("\n" + "=" * 50)
    print(f"🎯 Segmentación completada en {device}")
    print(f"🟢 Puntos positivos: {len(positive_points)}")
    print(f"🔴 Puntos negativos: {len(negative_points)}")
    print(f"📏 Área máscara: {area_px} px{area_mm2_str}")
    print(f"⭐ Score mejor máscara: {best_score:.4f}")
    print(f"🎭 Total máscaras generadas: {len(masks)}")
    print("=" * 50)

    # ---- Guardado opcional ----
    # refined_mask_pil = Image.fromarray((refined_mask.astype(np.uint8) * 255))
    # refined_mask_pil.save("segmentation_result_points.png")
    # print("💾 Guardado en 'segmentation_result_points.png'")

if __name__ == "__main__":
    main()
