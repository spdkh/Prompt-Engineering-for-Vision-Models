import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Starting water segmentation test...")
# --- Load image ---
image_path = "/project/sdjkhosh/Prompt-Engineering-for-Vision-Models/images/stream5.jpg"  # Update path if needed
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image at: {image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# --- Water color segmentation (HSV ranges tuned for this river scene) ---
# Water here appears as a muted blue-green with reflections
lower1 = np.array([85,  10,  40])
upper1 = np.array([130, 180, 200])

lower2 = np.array([20,  5,  80])   # captures brownish-green reflections near shore
upper2 = np.array([85, 80, 180])

mask1 = cv2.inRange(img_hsv, lower1, upper1)
mask2 = cv2.inRange(img_hsv, lower2, upper2)
raw_mask = cv2.bitwise_or(mask1, mask2)

# --- Constrain to upper ~55% of the image (water is in the upper half) ---
h, w = raw_mask.shape
zone_mask = np.zeros_like(raw_mask)
zone_mask[:int(h * 0.55), :] = 255
raw_mask = cv2.bitwise_and(raw_mask, zone_mask)

print("Initial mask created. Starting morphological cleanup...")

# --- Morphological cleanup ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_clean = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN,  kernel, iterations=2)

# Keep only the largest connected component (the river body)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask_final = np.uint8(labels == largest) * 255
else:
    mask_final = mask_clean
print("Morphological cleanup completed. Building overlay...")

# --- Build overlay ---
overlay = img_rgb.copy()
blue_layer = np.zeros_like(img_rgb)
blue_layer[:, :] = (30, 120, 255)   # vivid blue tint

alpha = 0.45  # transparency
mask_bool = mask_final > 0
overlay[mask_bool] = (
    alpha * blue_layer[mask_bool] + (1 - alpha) * img_rgb[mask_bool]
).astype(np.uint8)

print("Overlay created. Drawing contours...")
# Draw contour outline for crisp edge
contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
cv2.drawContours(overlay_bgr, contours, -1, (0, 80, 255), 3)
overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

print("Contours drawn. Preparing final visualization...")
# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor("#1a1a2e")

titles  = ["Original", "Water Mask", "Overlay"]
images  = [img_rgb, mask_final, overlay]
cmaps   = [None, "Blues", None]

for ax, title, im, cmap in zip(axes, titles, images, cmaps):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

plt.suptitle("Water Detection – River Scene", color="white", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
print("Visualization ready. Saving figure...")
plt.savefig("water_mask_result.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Saved → water_mask_result.png")