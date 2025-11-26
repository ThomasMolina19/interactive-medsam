#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert DICOM files to PNG format.
Creates a 'png_output' subfolder inside the DICOM directory.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
import pydicom
from pathlib import Path


def load_and_convert_dicom(dicom_path):
    """
    Load DICOM file and convert to PNG-ready array (0-255 range)
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        numpy array in uint8 format, or None if failed
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Extract pixel array
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 0-255 range
        if pixel_array.dtype != np.uint8:
            # Check if DICOM has windowing information
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                # Use DICOM windowing
                center = dicom_data.WindowCenter
                width = dicom_data.WindowWidth
                
                # Handle MultiValue (sometimes WindowCenter/Width are arrays)
                if isinstance(center, pydicom.multival.MultiValue):
                    center = float(center[0])
                if isinstance(width, pydicom.multival.MultiValue):
                    width = float(width[0])
                
                min_val = center - width / 2
                max_val = center + width / 2
                
                pixel_array = np.clip(pixel_array, min_val, max_val)
                pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                # Simple min-max normalization
                min_val = pixel_array.min()
                max_val = pixel_array.max()
                
                if max_val > min_val:
                    pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        return pixel_array
        
    except Exception as e:
        print(f"❌ Error loading {dicom_path}: {e}")
        return None


def convert_dicom_folder_to_png(dicom_folder, output_subfolder="png_output"):
    """
    Convert all DICOM files in a folder to PNG format.
    Creates output folder inside the DICOM directory.
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        output_subfolder: Name of subfolder to create for PNG files (default: "png_output")
    """
    
    print("="*70)
    print("🔬 DICOM to PNG Converter")
    print("="*70)
    
    # Verify input folder exists
    if not os.path.exists(dicom_folder):
        print(f"❌ Error: Folder not found: {dicom_folder}")
        sys.exit(1)
    
    print(f"📁 Input folder: {dicom_folder}")
    
    # Create output directory inside the DICOM folder
    output_dir = os.path.join(dicom_folder, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Output folder: {output_dir}")
    
    # Find all DICOM files (without extension or with common DICOM extensions)
    dicom_patterns = ['*', '*.dcm', '*.DCM', '*.dicom', '*.DICOM']
    dicom_files = []
    
    for pattern in dicom_patterns:
        potential_files = glob.glob(os.path.join(dicom_folder, pattern))
        for file_path in potential_files:
            # Skip directories and the output folder itself
            if os.path.isfile(file_path) and output_subfolder not in file_path:
                # Try to read as DICOM to verify
                try:
                    pydicom.dcmread(file_path, stop_before_pixels=True)
                    if file_path not in dicom_files:
                        dicom_files.append(file_path)
                except:
                    pass  # Not a valid DICOM file
    
    if not dicom_files:
        print(f"⚠️ No DICOM files found in {dicom_folder}")
        sys.exit(1)
    
    # Sort files for consistent ordering
    dicom_files.sort()
    
    print(f"📊 Found {len(dicom_files)} DICOM files")
    print("="*70)
    
    # Convert each DICOM to PNG
    successful = 0
    failed = 0
    
    for i, dicom_path in enumerate(dicom_files, 1):
        filename = os.path.basename(dicom_path)
        
        # Generate output filename (remove extension if exists, add .png)
        base_name = os.path.splitext(filename)[0]
        if not base_name:  # If file had no extension
            base_name = filename
        output_filename = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(dicom_files)}] Converting: {filename} -> {output_filename}")
        
        # Convert DICOM to array
        pixel_array = load_and_convert_dicom(dicom_path)
        
        if pixel_array is None:
            print(f"    ❌ Failed to convert")
            failed += 1
            continue
        
        # Save as PNG
        try:
            img = Image.fromarray(pixel_array)
            img.save(output_path)
            print(f"    ✅ Saved ({pixel_array.shape[1]}x{pixel_array.shape[0]} pixels)")
            successful += 1
        except Exception as e:
            print(f"    ❌ Failed to save: {e}")
            failed += 1
    
    # Summary
    print("="*70)
    print("📊 CONVERSION SUMMARY")
    print("="*70)
    print(f"✅ Successfully converted: {successful}/{len(dicom_files)}")
    print(f"❌ Failed: {failed}/{len(dicom_files)}")
    print(f"📂 Output directory: {output_dir}")
    print("="*70)
    
    return successful, failed


def main():
    """
    Main function with argument parsing
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert DICOM files to PNG format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert DICOMs in a specific folder
  python dicom_to_png.py --input /path/to/dicom/folder
  
  # Specify custom output subfolder name
  python dicom_to_png.py --input /path/to/dicom/folder --output my_pngs
        """
    )
    
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to folder containing DICOM files"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="png_output",
        help="Name of subfolder to create for PNG files (default: png_output)"
    )
    
    args = parser.parse_args()
    
    # Run conversion
    convert_dicom_folder_to_png(args.input, args.output)


if __name__ == "__main__":
    main()
