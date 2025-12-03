import sys
sys.path.append('path/to/segment-anything')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage
from skimage import morphology
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector


device = "mps" if torch.backends.mps.is_available() else "cpu"

# Here goes the path to your MedSAM model checkpoint
ckpt = "/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/Checkpoints/sam_vit_b_01ec64.pth"

# Load model
sam = sam_model_registry["vit_b"]()
checkpoint = torch.load(ckpt, map_location=device)
sam.load_state_dict(checkpoint)
sam = sam.to(device)

predictor = SamPredictor(sam)

# Line 31: Here goes the path to your medical image
img = np.array(Image.open("/Users/thomasmolinamolina/Downloads/UNAL/MATERIAS/SEMESTRE 6/PALUZNY/DATA/D1/pngs/I14.png").convert("RGB"))

# Enhance contrast for medical images
img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

predictor.set_image(img_enhanced)

H, W = img.shape[:2]

def interactive_box_selector(img):
    """Enhanced interactive bounding box selection"""
    
    class BoxSelector:
        def __init__(self):
            self.box = None
            self.selected = False
            
        def onselect(self, eclick, erelease):
            """Callback for rectangle selection"""
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            # Ensure coordinates are in correct order
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            self.box = np.array([x_min, y_min, x_max, y_max])
            self.selected = True
            
            # Update title with coordinates
            ax.set_title(f"Selected: [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}] - Close window when ready")
            fig.canvas.draw()
    
    # Create the selector object
    selector_obj = BoxSelector()
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title("🎯 Drag to select region of interest")
    ax.axis('off')
    
    # Create rectangle selector with optimal settings
    rectangle_selector = RectangleSelector(
        ax, 
        selector_obj.onselect,
        useblit=True,           # Faster drawing
        button=[1],             # Left mouse button only
        minspanx=10, minspany=10,  # Minimum size
        spancoords='pixels',
        interactive=True,       # Enable resizing
        drag_from_anywhere=True # Drag from inside rectangle
    )
    
    # Beautiful styling
    rectangle_selector.rectprops = dict(
        facecolor='cyan', 
        edgecolor='red',
        alpha=0.4, 
        fill=True,
        linewidth=3
    )
    
    # Instructions
    plt.figtext(0.5, 0.02, 
                "💡 Drag to create box • Drag edges to resize • Close window when done", 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()
    
    # Return selected box or intelligent default
    if selector_obj.box is not None:
        return selector_obj.box.astype(int)
    else:
        # Smart default: center region
        H, W = img.shape[:2]
        margin = min(W, H) // 8
        return np.array([margin, margin, W-margin, H-margin])

# Use the enhanced selector
print("🎯 Interactive box selection starting...")
box = interactive_box_selector(img)
print(f"✅ Final selected box: {box}")

# Generate masks using the selected box
masks, scores, _ = predictor.predict(
    box=box,
    multimask_output=True
)

# Select best mask
best_mask = masks[np.argmax(scores)]

# Post-process mask
def refine_medical_mask(mask):
    """Clean up the segmentation mask for medical images"""
    # Remove small objects
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    
    # Fill holes
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Smooth with morphological operations
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth

refined_mask = refine_medical_mask(best_mask)

# Enhanced visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Original results
axes[0,0].imshow(img)
axes[0,0].set_title("Original Image")
axes[0,0].axis('off')

axes[0,1].imshow(img)
axes[0,1].imshow(best_mask, alpha=0.5, cmap='Reds')
# Add bounding box visualization
rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                linewidth=3, edgecolor='green', facecolor='none')
axes[0,1].add_patch(rect)
axes[0,1].set_title("Raw SAM Output with Selected Box")
axes[0,1].axis('off')

axes[0,2].imshow(best_mask, cmap='gray')
axes[0,2].set_title("Raw Mask")
axes[0,2].axis('off')

# Row 2: Enhanced results
axes[1,0].imshow(img_enhanced)
axes[1,0].set_title("Enhanced Image")
axes[1,0].axis('off')

axes[1,1].imshow(img)
axes[1,1].imshow(refined_mask, alpha=0.5, cmap='Blues')
# Add bounding box to refined view too
rect2 = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                 linewidth=3, edgecolor='yellow', facecolor='none')
axes[1,1].add_patch(rect2)
axes[1,1].set_title("Refined Segmentation")
axes[1,1].axis('off')

axes[1,2].imshow(refined_mask, cmap='gray')
axes[1,2].set_title("Refined Mask")
axes[1,2].axis('off')

plt.tight_layout()
plt.show()

# Results summary
print(f"🎯 Segmentation completed on {device}")
print(f"📦 Box coordinates: {box}")
print(f"📏 Mask area: {np.sum(refined_mask)} pixels")
print(f"⭐ Best mask score: {scores[np.argmax(scores)]:.4f}")
print(f"🎭 Total masks generated: {len(masks)}")

# Save results if needed
# refined_mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8))
# refined_mask_pil.save("segmentation_result.png")
# print("💾 Mask saved as 'segmentation_result.png'")