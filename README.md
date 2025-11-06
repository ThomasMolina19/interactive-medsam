# Interactive MedSAM Segmentation

Interactive medical image segmentation tool using SAM (Segment Anything Model) and MedSAM with enhanced preprocessing and post-processing specifically optimized for medical imaging applications.

## 🎯 Features

- **Interactive Point-Based Selection**: Real-time segmentation with positive/negative point prompts
- **Interactive Bounding Box Selection**: User-friendly interface for selecting regions of interest
- **Real-Time Preview**: See segmentation results instantly as you add points
- **Undo/Redo Functionality**: Easy correction of point selections with keyboard shortcuts
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

## 🚀 Usage

### Point-Based Segmentation (Real-Time) - NEW! ⭐

The most interactive and intuitive method with real-time feedback.

#### Step 1: Update file paths

Edit the script `segment_sam_points.py` and update:

```python
# Line 6: Update the SAM repository path (if needed)
sys.path.append('path/to/segment-anything')

# Line 19: Update checkpoint path
ckpt = "/path/to/checkpoints/sam_vit_h_4b8939.pth"

# Line 28: Update your image path
img = np.array(Image.open("/path/to/your/medical/image.png").convert("RGB"))
```

#### Step 2: Run the interactive segmentation tool

```bash
python segment_sam_points.py
```

#### Step 3: Interactive point selection with real-time preview

The tool opens a **dual-panel interface**:

**Left Panel**: Original image where you place points
**Right Panel**: Live segmentation preview (updates instantly!)

**Controls:**
- 🟢 **Right Click**: Add POSITIVE point (mark the object you want)
- 🔴 **Left Click**: Add NEGATIVE point (exclude unwanted regions)
- ⌨️ **Press 'z'**: Undo last point
- ⌨️ **Press 'c'**: Clear all points
- ✅ **Close window or ESC**: Finish and view final results

**Workflow:**
1. Right-click on the object you want to segment (e.g., bone, organ)
2. See the segmentation appear instantly on the right panel
3. Add more positive points to refine the selection
4. Left-click on areas to exclude if needed
5. Use 'z' to undo mistakes
6. Close when satisfied to see detailed results

**Example:**
```
🎯 Selecting a humerus bone:
1. Right-click center of bone → instant preview
2. Right-click on bone edges → refinement
3. Left-click on background if included → exclusion
4. Press 'z' if you made a mistake
5. Close window → see final visualization
```

### Single Image Segmentation (Bounding Box)

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
│   ├── sam_vit_h_4b8939.pth       # SAM ViT-Huge checkpoint
│   ├── sam_vit_b_01ec64.pth       # SAM ViT-Base checkpoint (optional)
│   └── medsam_vit_b.pth           # MedSAM checkpoint (optional)
├── dicom_pngs/                     # Input images folder
│   ├── I01.png
│   └── ...
├── segmentation_results/           # Output folder (batch processing)
│   ├── I01_segmentation.png
│   ├── I01_mask.png
│   └── ...
├── segment_sam_points.py           # 🆕 Point-based real-time segmentation
├── segment_one.py                  # Bounding box single image (SAM)
├── segment_one_medsam.py           # Bounding box single image (MedSAM)
├── segment_multiple.py             # Batch processing script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── examples/                       # (Optional) Example images
    └── sample_medical_image.png
```

## 🔍 Key Functions

### `interactive_point_selector(img, predictor)` 🆕
Real-time interactive point-based segmentation with live preview.

**Features:**
- Dual-panel interface (image + live mask)
- Positive/negative point prompts
- Instant segmentation feedback
- Undo/redo functionality (keyboard shortcuts)
- Confidence score and area display

**Controls:**
- Right-click: Positive points (green stars ⭐)
- Left-click: Negative points (red X ❌)
- 'z' key: Undo last point
- 'c' key: Clear all points

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

- **Medical Research**: Organ segmentation, tumor detection, bone analysis
- **Clinical Applications**: ROI analysis, measurement tools, anatomical studies
- **Educational**: Teaching medical image analysis, interactive demonstrations
- **Prototyping**: Quick annotation for training datasets, fast iteration
- **Precision Medicine**: Patient-specific segmentation with point-based refinement

## 🆕 What's New

### Version 2.0 (Current)
- ✨ **Point-based segmentation** with real-time preview
- 🔄 **Undo/redo functionality** for easy correction
- 📊 **Dual-panel interface** for instant feedback
- ⌨️ **Keyboard shortcuts** ('z' for undo, 'c' for clear)
- 🎯 **Positive/negative prompts** for precise control
- 🚀 **SAM support** alongside MedSAM

### Version 1.0
- Interactive bounding box selection
- MedSAM integration
- Batch processing
- Medical image enhancement

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
Verify the checkpoint path matches your downloaded file location. Update the path in your script.

### MPS not available
The script will automatically fallback to CPU. For NVIDIA GPU, change the device to `device = "cuda"`.

### Low segmentation quality
- **Point-based method**: Try adding more positive points or negative points to exclude unwanted regions
- **Box method**: Adjust the bounding box to better fit the region
- Modify enhancement parameters (alpha, beta)
- Adjust post-processing parameters in `refine_medical_mask()`

### Segmentation not updating in real-time
- Ensure you're clicking on the left panel (image panel)
- Check that matplotlib backend is interactive (should be by default)
- Try closing and reopening the script

### Points not being placed
- Make sure you're using the correct mouse button (right for positive, left for negative)
- Verify you're clicking inside the image area
- Check console for error messages

## 👤 Authors
 

**Thomas Molina Molina**

**Gustavo Adolfo Pérez**