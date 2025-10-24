# Interactive MedSAM Segmentation

Interactive medical image segmentation tool using MedSAM (Medical Segment Anything Model) with enhanced preprocessing and post-processing specifically optimized for medical imaging applications.

## 🎯 Features

- **Interactive Bounding Box Selection**: User-friendly interface for selecting regions of interest
- **Medical Image Enhancement**: Automatic contrast adjustment optimized for medical images
- **Advanced Post-Processing**: Morphological operations to refine segmentation masks
- **Multi-Mask Generation**: Generates multiple segmentation proposals and selects the best one
- **Comprehensive Visualization**: Side-by-side comparison of original and refined results
- **MPS Support**: Optimized for Apple Silicon (M1/M2/M3)

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU, Apple Silicon (MPS), or CPU

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

### Step 4: Download MedSAM checkpoint

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

## 🚀 Usage

### Single Image Segmentation

#### Step 1: Update file paths

Edit the script `segment_one.py` and update:

```python
# Line 6: Update the SAM repository path (if needed)
sys.path.append('path/to/segment-anything')

# Line 13: Update checkpoint path
ckpt = "checkpoints/medsam_vit_b.pth"

# Line 24: Update your image path
img = np.array(Image.open("path/to/your/medical/image.png").convert("RGB"))
```

#### Step 2: Run the segmentation tool

```bash
python segment_one.py
```

#### Step 3: Interactive segmentation

1. **Select Region**: A window will open showing your medical image
2. **Draw Bounding Box**: Click and drag to create a box around your region of interest
3. **Adjust**: Drag the edges to resize or adjust the box
4. **Confirm**: Close the window when satisfied with the selection
5. **Results**: View the segmentation results in the output visualization

### Batch Processing (Multiple Images)

For processing multiple medical images in a folder:

#### Step 1: Prepare your images

Place all medical images (PNG format) in a folder, for example:
```
dicom_pngs/
├── I01.png
├── I02.png
├── I03.png
└── ...
```

#### Step 2: Update file paths

Edit the script `segment_multiple.py` and update:

```python
# Line 13: Update checkpoint path
ckpt = "checkpoints/medsam_vit_b.pth"

# Line 111: Update input folder path
input_folder = "path/to/your/dicom_pngs"

# Line 112: Update output folder path
output_folder = "path/to/segmentation_results"
```

#### Step 3: Choose processing mode

The script supports two modes:

**Mode 1: Fixed Bounding Box (Default)**
- Uses the same bounding box for all images
- Faster processing
- Ideal for aligned/registered images

```python
# In segment_multiple.py, line 116
use_interactive = False
fixed_box = [150, 200, 450, 500]  # [x_min, y_min, x_max, y_max]
```

**Mode 2: Interactive Box Selection**
- Select bounding box for each image
- More flexible but slower
- Better for varying anatomies

```python
# In segment_multiple.py, line 116
use_interactive = True
```

#### Step 4: Run batch processing

```bash
python segment_multiple.py
```

#### Step 5: Monitor progress

The script will display progress for each image:
```
Processing image 1/50: I01.png
✅ Successfully processed I01.png
Processing image 2/50: I02.png
✅ Successfully processed I02.png
...
```

#### Step 6: View results

Results are saved in the output folder with the structure:
```
segmentation_results/
├── I01_segmentation.png          # Visualization
├── I01_mask.png                   # Binary mask
├── I02_segmentation.png
├── I02_mask.png
└── ...
```

#### Batch Processing Summary

After completion, you'll see statistics:
```
📊 Batch Processing Summary:
✅ Successfully processed: 48/50 images
❌ Failed: 2 images
⏱️  Total time: 5m 23s
⚡ Average time per image: 6.5s
```

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

## 🏗️ Technical Details

### Image Enhancement
- **Contrast adjustment**: `alpha=1.2, beta=10`
- Optimized for medical imaging (CT, MRI, X-rays)

### Segmentation Pipeline
1. Image preprocessing and enhancement
2. Interactive bounding box selection
3. MedSAM inference with multi-mask output
4. Best mask selection based on confidence scores
5. Post-processing and refinement

### Mask Refinement
- **Small object removal**: Filters objects < 500 pixels
- **Hole filling**: Binary morphological operations
- **Smoothing**: Disk-shaped kernel (radius=2)
- **Opening/Closing**: Noise reduction and gap filling

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

## 📁 Project Structure

```
interactive-medsam/
├── checkpoints/
│   └── medsam_vit_b.pth          # MedSAM model checkpoint
├── dicom_pngs/                    # Input images folder
│   ├── I01.png
│   └── ...
├── segmentation_results/          # Output folder (batch processing)
│   ├── I01_segmentation.png
│   ├── I01_mask.png
│   └── ...
├── segment_one.py                 # Single image segmentation
├── segment_multiple.py            # Batch processing script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── examples/                      # (Optional) Example images
    └── sample_medical_image.png
```

## 🔍 Key Functions

### `interactive_box_selector(img)`
Interactive GUI for region of interest selection using matplotlib's RectangleSelector widget.

**Features:**
- Real-time coordinate display
- Resizable and draggable boxes
- Visual feedback with colored overlays

### `refine_medical_mask(mask)`
Post-processing pipeline for mask refinement.

**Operations:**
- Small object removal
- Hole filling
- Morphological smoothing (opening + closing)

## 🎓 Use Cases

- **Medical Research**: Organ segmentation, tumor detection
- **Clinical Applications**: ROI analysis, measurement tools
- **Educational**: Teaching medical image analysis
- **Prototyping**: Quick annotation for training datasets

## 📚 References

- **MedSAM Paper**: [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- **MedSAM Repository**: https://github.com/bowang-lab/MedSAM
- **Segment Anything (SAM)**: https://github.com/facebookresearch/segment-anything
- **SAM Paper**: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

## 🐛 Troubleshooting

### "No module named 'segment_anything'"
Install SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### "Checkpoint not found"
Verify the checkpoint path in line 13 matches your downloaded file location.

### MPS not available
The script will automatically fallback to CPU. For NVIDIA GPU, change line 12 to `device = "cuda"`.

### Low segmentation quality
- Try adjusting the bounding box to better fit the region
- Modify enhancement parameters (alpha, beta) in line 24
- Adjust post-processing parameters in `refine_medical_mask()`

## 👤 Authors
 

**Thomas Molina Molina**

**Gustavo Adolfo Pérez**