"""Interfaz de usuario para selección de paths usando Finder de macOS."""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox


def _cancel_exit(root, item):
    """Cancela y sale del programa."""
    print(f"❌ No se seleccionó {item}. Cancelado.")
    root.destroy()
    sys.exit(1)


def get_user_paths():
    """
    Solicita al usuario los paths usando Finder de macOS.
    
    Returns:
        tuple: (checkpoint_path, data_dir, output_dir)
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("\n" + "="*70)
    print("🔧 CONFIGURACIÓN DE PATHS (usando Finder)")
    print("="*70)
    
    # 1. Seleccionar checkpoint SAM
    print("\n📦 Selecciona el archivo del CHECKPOINT SAM (.pth)...")
    ckpt = filedialog.askopenfilename(
        title="Seleccionar checkpoint SAM (.pth)",
        filetypes=[("PyTorch checkpoint", "*.pth"), ("Todos los archivos", "*.*")]
    )
    if not ckpt:
        _cancel_exit(root, "checkpoint")
    print(f"   ✅ Checkpoint: {ckpt}")
    
    # 2. Seleccionar directorio de imágenes
    print("\n📁 Selecciona la CARPETA con las imágenes (JPG/PNG)...")
    data_dir = filedialog.askdirectory(title="Seleccionar carpeta de imágenes")
    if not data_dir:
        _cancel_exit(root, "carpeta de imágenes")
    print(f"   ✅ Imágenes: {data_dir}")
    
    # 3. Seleccionar directorio de salida
    print("\n💾 Selecciona la CARPETA de SALIDA para los resultados...")
    output_dir = filedialog.askdirectory(title="Seleccionar carpeta de salida")
    if not output_dir:
        _cancel_exit(root, "carpeta de salida")
    os.makedirs(output_dir, exist_ok=True)
    print(f"   ✅ Salida: {output_dir}")
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("📋 RESUMEN DE CONFIGURACIÓN:")
    print(f"   • Checkpoint: {ckpt}")
    print(f"   • Imágenes:   {data_dir}")
    print(f"   • Salida:     {output_dir}")
    print("="*70)
    
    # Diálogo de confirmación
    confirm = messagebox.askyesno(
        "Confirmar configuración",
        f"¿Confirmar la siguiente configuración?\n\n"
        f"• Checkpoint:\n  {ckpt}\n\n"
        f"• Imágenes:\n  {data_dir}\n\n"
        f"• Salida:\n  {output_dir}"
    )
    
    root.destroy()
    
    if not confirm:
        print("❌ Cancelado por el usuario.")
        sys.exit(1)
    
    print("\n✅ Configuración confirmada!")
    return ckpt, data_dir, output_dir
