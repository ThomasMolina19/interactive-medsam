#!/bin/bash

# Setup script for Batch DICOM Segmentation
echo "🔬 Setting up Batch DICOM Segmentation environment..."

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r batch_requirements.txt

# Install Segment Anything Model from GitHub
echo "🤖 Installing Segment Anything Model..."
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "✅ Setup complete!"
echo ""
echo "🚀 To run the batch segmentation:"
echo "   python batch_dicom_segmentation.py"
echo ""
echo "📝 Edit the paths in the script before running:"
echo "   - DICOM_FOLDER: Path to your DICOM files"
echo "   - CHECKPOINT_PATH: Path to SAM model checkpoint"
echo "   - OUTPUT_FOLDER: Where to save results"