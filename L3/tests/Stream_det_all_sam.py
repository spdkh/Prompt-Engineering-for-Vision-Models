# ============================================================================
# Run all SAM versions on stream6.jpg with every prompt variety
# ============================================================================
# Prompt varieties supported by Ultralytics SAM (per docs.ultralytics.com):
#   1. Bounding box        — bboxes=[x1, y1, x2, y2]                (from text-prompted detector)
#   2. Single positive point — points=[x, y], labels=[1]
#   3. Multiple positive points — points=[[x1,y1], [x2,y2]], labels=[1, 1]
#   4. Positive + negative points — points=[[[x1,y1], [x2,y2]]], labels=[[1, 0]]
#   5. "Everything" mode   — no prompts; segment all objects in image
#
# Detector (Grounding DINO) generates the bbox/point coordinates from a text
# prompt, then SAM consumes those coordinates to produce masks.
# ============================================================================

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from PIL.ImageFilter import GaussianBlur

from transformers import pipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from ultralytics import SAM


prj_path = Path('Prompt-Engineering-for-Vision-Models')
sys.path.insert(0, str(prj_path))

from utils import preprocess_outputs
from utils import show_boxes_and_labels_on_image
from utils import show_masks_on_image
from utils import make_bbox_annots
from utils import resize_image
from utils import get_model, plot_multy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Configuration
# ============================================================================
versions = [
    'mobile_sam',
    'sam_b', 'sam_l',
    'sam2_b', 'sam2_t', 'sam2_s', 'sam2_l',
    'sam2.1_b', 'sam2.1_t', 'sam2.1_s', 'sam2.1_l',
]

OWL_checkpoint  = 'google/owlvit-base-patch32'
Dino_checkpoint = 'IDEA-Research/grounding-dino-base'
obj_det_model   = Dino_checkpoint

# Single image to evaluate every prompt variety against
image_name = 'stream6'
text_prompt = 'river'  # adjust this if your stream6 image needs a different prompt

# ============================================================================
# Load detector and image once (all SAM runs share the same inputs)
# ============================================================================
detector = pipeline("zero-shot-object-detection", model=obj_det_model, device=device)

raw_image = Image.open(f"images/{image_name}.jpg")
resized_image = resize_image(raw_image, input_size=512)
image_path_resized = f"images/{image_name}_resized.jpg"
resized_image.save(image_path_resized)

W, H = resized_image.size
print(f"Image: {image_name}  ({W}x{H})")

# Run the open-vocabulary detector once → reuse boxes/points across all SAMs
det_results = detector(resized_image, candidate_labels=[text_prompt])
print(f"Detector found {len(det_results)} object(s) for prompt '{text_prompt}'")

# if len(det_results) == 0:
#     raise RuntimeError(
#         f"Detector returned no boxes for prompt '{text_prompt}'. "
#         "Try a different text prompt or check the image."
#     )

# Convert detector output → list of [x1, y1, x2, y2] boxes
input_boxes = []
for r in det_results:
    b = r['box']
    input_boxes.append([b['xmin'], b['ymin'], b['xmax'], b['ymax']])

# Compute box centers — used as positive point prompts
box_centers = [
    [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    for b in input_boxes
]
primary_box    = input_boxes[0]
primary_center = box_centers[0]

# A "negative" point: somewhere clearly OUTSIDE the first detected box
# (tells SAM "do NOT include this region in the mask")
neg_point = [
    max(5, primary_box[0] - 30),  # 30px to the left of the box
    max(5, primary_box[1] - 30),
]

# ============================================================================
# Define every prompt variety
# Each entry: (prompt_name, kwargs_for_model.predict, short_explanation)
# ============================================================================
prompt_varieties = [
    (
        "1_bbox",
        dict(bboxes=primary_box),
        "Single bounding box from Grounding DINO (most common Grounded-SAM mode)",
    ),
    (
        "2_bbox_all",
        dict(bboxes=input_boxes),
        "All bounding boxes from the detector (multi-object)",
    ),
    (
        "3_single_point",
        dict(points=primary_center, labels=[1]),
        "Single positive point at the center of the detected box",
    ),
    (
        "4_multi_points",
        dict(points=box_centers, labels=[1] * len(box_centers)),
        "Multiple positive points (one per detected object) — yields one mask each",
    ),
    (
        "5_multi_points_one_object",
        dict(points=[box_centers[:2] if len(box_centers) >= 2 else [primary_center, primary_center]],
             labels=[[1, 1] if len(box_centers) >= 2 else [1, 1]]),
        "Multiple positive points belonging to ONE object (nested list)",
    ),
    (
        "6_positive_negative_points",
        dict(points=[[primary_center, neg_point]], labels=[[1, 0]]),
        "Positive + negative points: foreground center + background point to exclude",
    ),
    (
        "7_everything",
        dict(),  # no prompt → SAM segments everything
        "Everything mode: no prompt, SAM auto-generates masks for all objects",
    ),
]

# ============================================================================
# Sweep every (SAM version × prompt variety) combination
# ============================================================================
all_imgs   = []
all_titles = []
speed_data = []

for SAM_version in versions:
    print(f"\n{'='*100}\nSAM version: {SAM_version}\n{'='*100}")
    model = SAM(model=get_model(SAM_version))

    for prompt_name, prompt_kwargs, explanation in prompt_varieties:
        print(f"  → Prompt [{prompt_name}]  {explanation}")

        try:
            # "Everything" mode passes only the image; others pass bbox/point kwargs
            result = model.predict(resized_image, **prompt_kwargs)
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            speed_data.append({
                "SAM_version": SAM_version,
                "prompt": prompt_name,
                "total_speed_ms": np.nan,
                "num_masks": 0,
                "error": str(e),
            })
            continue

        # Speed
        spd = result[0].speed
        total_speed = spd['preprocess'] + spd['inference'] + spd['postprocess']

        # Masks
        masks = result[0].masks.data.detach().cpu().numpy() if result[0].masks is not None else np.array([])
        num_masks = len(masks)

        speed_data.append({
            "SAM_version": SAM_version,
            "prompt": prompt_name,
            "total_speed_ms": total_speed,
            "num_masks": num_masks,
            "error": "",
        })

        # Collect masks for visualization
        for idx, mask in enumerate(masks):
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            all_imgs.append(mask)
            all_titles.append(f"{SAM_version}\n{prompt_name}\nmask {idx+1}/{num_masks}")

        print(f"    ✓ {num_masks} mask(s)  |  {total_speed:.1f} ms")

# ============================================================================
# Save speed/result table
# ============================================================================
df_speed = pd.DataFrame(speed_data)
df_speed.to_csv("sam_stream6_prompt_sweep.csv", index=False)

print("\n" + "="*100)
print("Per-prompt speed (ms) by SAM version:")
print("="*100)

# Pivot for an easy-to-read view
pivot = df_speed.pivot(index="SAM_version", columns="prompt", values="total_speed_ms")
print(pivot.round(1))

print("\nMask counts:")
print(df_speed.pivot(index="SAM_version", columns="prompt", values="num_masks"))

# ============================================================================
# Visualize all results in one grid
# ============================================================================
plot_multy(all_imgs, 'all_sam.png', cols=len(prompt_varieties), rows=len(versions), titles=all_titles)