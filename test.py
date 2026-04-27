"""
Stream Detection via Color-Space Thresholding and Centerline Estimation
=======================================================================
Implements the single-frame stream detection pipeline described in:
  - Water pixel identification using RGB color-space thresholding (Eq. 1–2)
  - Binary mask generation
  - Centerline estimation via perspective transformation to world coordinates

Usage:
    python stream_detection.py --image <path_to_image> [--show]
"""

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Intrinsic and extrinsic camera parameters for perspective projection."""
    tilt_deg: float = 45.0          # downward tilt angle (degrees)
    altitude_m: float = 10.0        # UAV altitude above ground (meters)
    fx: float = 1000.0              # focal length x (pixels)
    fy: float = 1000.0              # focal length y (pixels)
    cx: float = 960.0               # principal point x (pixels)
    cy: float = 540.0               # principal point y (pixels)
    image_width: int = 1920
    image_height: int = 1080


@dataclass
class WaterColorRange:
    """
    Defines the water color range C_w in RGB space (Eq. 1).
    Defaults tuned for natural stream/river water under daylight conditions.
    Adjust these for your specific lighting environment.
    """
    R_min=0;  R_max=160;   # moderate red; high end covers the silty beige sections
    G_min=60; G_max=250;   # green is dominant throughout
    B_min=80;  B_max=255;   # blue present but always below green


# ---------------------------------------------------------------------------
# Step 1 – Water color thresholding  (Eq. 1 & 2)
# ---------------------------------------------------------------------------

def compute_water_mask(
    frame: np.ndarray,
    color_range: WaterColorRange,
    morph_kernel_size: int = 5
) -> np.ndarray:
    """
    Build binary mask M(x,y) identifying water pixels.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (as loaded by OpenCV), shape (H, W, 3).
    color_range : WaterColorRange
        Per-channel min/max thresholds defining C_w.
    morph_kernel_size : int
        Kernel size for morphological closing (removes small holes/noise).

    Returns
    -------
    mask : np.ndarray
        Binary mask, dtype uint8, values {0, 255}, shape (H, W).
    """
    # OpenCV loads images in BGR order — split accordingly
    B, G, R = cv2.split(frame)

    # Eq. 1:  C_w = {(R,G,B) | R_min ≤ R ≤ R_max, G_min ≤ G ≤ G_max, B_min ≤ B ≤ B_max}
    r_mask = (R >= color_range.R_min) & (R <= color_range.R_max)
    g_mask = (G >= color_range.G_min) & (G <= color_range.G_max)
    b_mask = (B >= color_range.B_min) & (B <= color_range.B_max)

    # Eq. 2:  M(x,y) = 1 if (R,G,B) ∈ C_w, else 0
    mask = (r_mask & g_mask & b_mask).astype(np.uint8) * 255

    # Morphological closing: fill small gaps, remove salt-and-pepper noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return mask


# ---------------------------------------------------------------------------
# Step 2 – Perspective transformation → world coordinates
# ---------------------------------------------------------------------------

def build_homography(cam: CameraConfig) -> np.ndarray:
    """
    Construct a homography H that maps image pixels to ground-plane world
    coordinates (meters), assuming a flat ground plane and known camera tilt.

    The camera is tilted `tilt_deg` downward from horizontal, mounted at
    `altitude_m` above the ground.

    Returns
    -------
    H : np.ndarray, shape (3, 3)
    """
    tilt_rad = np.deg2rad(cam.tilt_deg)

    # Rotation about the x-axis (camera tilted downward)
    R_tilt = np.array([
        [1,              0,               0],
        [0,  np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0,  np.sin(tilt_rad),  np.cos(tilt_rad)],
    ], dtype=np.float64)

    # Camera intrinsic matrix K
    K = np.array([
        [cam.fx,      0, cam.cx],
        [     0, cam.fy, cam.cy],
        [     0,      0,      1],
    ], dtype=np.float64)

    # Translation: camera is `altitude_m` above the ground
    t = np.array([[0], [0], [cam.altitude_m]], dtype=np.float64)

    # Projection matrix [R | t]
    Rt = np.hstack([R_tilt, t])

    # Full projection P = K [R | t]
    P = K @ Rt

    # Ground-plane homography: drop the Y column (ground plane Z_world = 0)
    H = P[:, [0, 1, 3]]   # columns for X_w, Y_w, 1

    return np.linalg.inv(H)   # invert: image → world


def pixels_to_world(
    pixels: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Map image pixel coordinates to world-plane (X, Y) in meters.

    Parameters
    ----------
    pixels : np.ndarray, shape (N, 2)  — (col, row) pairs
    H      : np.ndarray, shape (3, 3)  — image-to-ground homography

    Returns
    -------
    world_pts : np.ndarray, shape (N, 2)  — (X, Y) in meters
    """
    pts_h = np.column_stack([pixels, np.ones(len(pixels))])  # homogeneous
    world_h = (H @ pts_h.T).T                                 # apply homography
    world_pts = world_h[:, :2] / world_h[:, 2:3]             # normalize
    return world_pts


# ---------------------------------------------------------------------------
# Step 3 – Centerline estimation
# ---------------------------------------------------------------------------

def estimate_centerline(
    mask: np.ndarray,
    H: np.ndarray,
    n_slices: int = 20
) -> Optional[np.ndarray]:
    """
    Estimate stream centerline path P = {(X_i, Y_i)} in world coordinates.

    Strategy: divide the image into horizontal slices; for each slice,
    compute the centroid of water pixels → one centerline point per slice.

    Parameters
    ----------
    mask     : np.ndarray  — binary water mask, shape (H, W)
    H        : np.ndarray  — image-to-world homography
    n_slices : int         — number of horizontal divisions

    Returns
    -------
    path : np.ndarray, shape (M, 2), world-coordinate centerline points,
           or None if no water is detected.
    """
    h, w = mask.shape
    slice_h = h // n_slices

    image_centers = []
    for i in range(n_slices):
        y0, y1 = i * slice_h, (i + 1) * slice_h
        row_slice = mask[y0:y1, :]
        ys, xs = np.where(row_slice == 255)
        if len(xs) == 0:
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys)) + y0    # offset back to full-image coords
        image_centers.append([cx, cy])

    if not image_centers:
        return None

    image_centers = np.array(image_centers, dtype=np.float64)
    world_centers = pixels_to_world(image_centers, H)
    return world_centers                # shape (M, 2)


def fit_centerline_polyline(path: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Optionally smooth the centerline by fitting a polynomial in world space.

    Returns smoothed (X, Y) points sampled along the fitted curve.
    """
    if len(path) < degree + 1:
        return path
    t = np.linspace(0, 1, len(path))
    t_fine = np.linspace(0, 1, 200)
    px = np.polyfit(t, path[:, 0], degree)
    py = np.polyfit(t, path[:, 1], degree)
    X_smooth = np.polyval(px, t_fine)
    Y_smooth = np.polyval(py, t_fine)
    return np.column_stack([X_smooth, Y_smooth])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def apply_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 120, 255),   # BGR — default: vivid orange-blue
    alpha: float = 0.55,
    contour_color: tuple = (0, 230, 255),
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Blend a colored mask over the detected water region and draw its contour.

    The result shows the original image everywhere except over water pixels,
    where a semi-transparent colored fill is blended in, with a bright contour
    drawn around the boundary of each detected region.

    Parameters
    ----------
    frame            : BGR input image
    mask             : binary water mask (uint8, values 0/255)
    color            : BGR fill color for the water region
    alpha            : opacity of the colored fill (0 = invisible, 1 = solid)
    contour_color    : BGR color for the region boundary
    contour_thickness: thickness of the boundary line in pixels

    Returns
    -------
    result : np.ndarray — annotated BGR image
    """
    result = frame.copy()

    # --- Colored fill: alpha-blend only over water pixels ---
    color_layer = np.zeros_like(frame, dtype=np.uint8)
    color_layer[:] = color                          # flood the layer with fill color
    water_px = mask == 255
    result[water_px] = cv2.addWeighted(
        frame, 1 - alpha, color_layer, alpha, 0
    )[water_px]

    # --- Contour around each detected water region ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, contour_color, contour_thickness)

    return result


def show_masked_image(
    frame: np.ndarray,
    mask: np.ndarray,
    save_path: Optional[str] = None,
) -> np.ndarray:
    masked = apply_mask_overlay(frame, mask)

    n_water = int(np.sum(mask == 255))
    pct = 100.0 * n_water / mask.size
    label = f"Water: {n_water:,} px  ({pct:.1f}%)"
    cv2.putText(masked, label, (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(masked, label, (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 80, 200),   2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, masked)          # save masked only, not composite
        print(f"[INFO] Masked image saved → {save_path}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Stream Detection — Mask Overlay", fontsize=14, fontweight="bold")
    axes[0].imshow(cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Frame")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected Water Region (mask overlay)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    return masked                               # return masked only


def draw_centerline_on_image(
    frame: np.ndarray,
    mask: np.ndarray,
    centerline_img: Optional[np.ndarray]
) -> np.ndarray:
    """
    Overlay water mask (blue tint) and centerline (red) on the original frame.
    """
    overlay = apply_mask_overlay(frame, mask)   # reuse the new overlay function

    # Draw centerline points in the image
    if centerline_img is not None and len(centerline_img) > 1:
        pts = centerline_img.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False,
                      color=(0, 0, 255), thickness=3)
        for pt in centerline_img:
            cv2.circle(overlay, tuple(pt.astype(int)), 6, (0, 255, 255), -1)

    return overlay


def visualize_results(
    frame: np.ndarray,
    mask: np.ndarray,
    world_path: Optional[np.ndarray],
    centerline_img_pts: Optional[np.ndarray]
) -> None:
    """Display a 2×2 figure: original | mask | overlay | world-space path."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Stream Detection Pipeline", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Water Mask M(x,y)")
    axes[0, 1].axis("off")

    overlay = draw_centerline_on_image(frame, mask, centerline_img_pts)
    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Overlay: Water Mask + Centerline")
    axes[1, 0].axis("off")

    if world_path is not None:
        smoothed = fit_centerline_polyline(world_path)
        axes[1, 1].plot(world_path[:, 0],   world_path[:, 1],
                        "o", color="steelblue", label="Centerline pts", ms=5)
        axes[1, 1].plot(smoothed[:, 0], smoothed[:, 1],
                        "-", color="crimson",  label="Smoothed path", lw=2)
        axes[1, 1].set_title("World-Coordinate Path P = {(Xᵢ, Yᵢ)}")
        axes[1, 1].set_xlabel("X (m)"); axes[1, 1].set_ylabel("Y (m)")
        axes[1, 1].legend(); axes[1, 1].grid(True)
        axes[1, 1].set_aspect("equal")
    else:
        axes[1, 1].text(0.5, 0.5, "No stream detected",
                        ha="center", va="center", fontsize=12)
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def detect_stream(
    image_path: str,
    color_range: Optional[WaterColorRange] = None,
    cam: Optional[CameraConfig] = None,
    show: bool = True,
    save_mask_path: Optional[str] = None,
) -> dict:
    """
    Full single-frame stream detection pipeline.

    Parameters
    ----------
    image_path      : path to input image
    color_range     : WaterColorRange — defaults used if None
    cam             : CameraConfig   — defaults used if None
    show            : if True, display all visualization panels
    save_mask_path  : if provided, save the mask-overlay composite image here

    Returns
    -------
    dict with keys:
        'mask'        — binary water mask (np.ndarray)
        'world_path'  — centerline in world coordinates (np.ndarray or None)
        'water_ratio' — fraction of frame classified as water (float)
        'masked_img'  — BGR image with colored mask overlay applied (np.ndarray)
    """
    if color_range is None:
        color_range = WaterColorRange()
    if cam is None:
        cam = CameraConfig()

    # --- Load image ---
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    print(f"[INFO] Image loaded: {frame.shape[1]}x{frame.shape[0]} px")

    # --- Step 1: Water mask (Eq. 1 & 2) ---
    mask = compute_water_mask(frame, color_range)
    water_ratio = float(np.sum(mask == 255)) / mask.size
    print(f"[INFO] Water pixels: {water_ratio*100:.1f}% of frame")

    # --- Step 2: Perspective homography ---
    H = build_homography(cam)

    # --- Step 3: Centerline in world coordinates ---
    world_path = estimate_centerline(mask, H, n_slices=20)

    # Back-project world centerline to image pixels for visualization
    centerline_img_pts = None
    if world_path is not None:
        H_inv = np.linalg.inv(H)
        pts_h = np.column_stack([world_path, np.ones(len(world_path))])
        img_h = (H_inv @ pts_h.T).T
        centerline_img_pts = (img_h[:, :2] / img_h[:, 2:3]).astype(np.float32)
        print(f"[INFO] Centerline estimated: {len(world_path)} points")
    else:
        print("[WARN] No stream detected in this frame.")

    # --- Step 4: Mask overlay image ---
    # Always build the masked image so it is available in the returned dict.
    masked_img = show_masked_image(
        frame, mask,
        # centerline_img=centerline_img_pts,
        save_path=save_mask_path,
    ) if show or save_mask_path else apply_mask_overlay(frame, mask)

    # --- Full pipeline visualization ---
    if show:
        visualize_results(frame, mask, world_path, centerline_img_pts)

    return {
        "mask": mask,
        "world_path": world_path,
        "water_ratio": water_ratio,
        "masked_img": masked_img,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-frame stream detection")
    parser.add_argument("--image", help="Path to input image",
                        default="images/stream6.jpg")
    parser.add_argument("--show",      action="store_true", default=False,
                        help="Display all result panels (default: True; use --no-show to disable)")
    parser.add_argument("--save-mask", default=None, metavar="PATH",
                        help="Save the mask-overlay composite image to this path (e.g. result.jpg)")
    parser.add_argument("--altitude",  type=float, default=10.0,
                        help="UAV altitude in meters (default: 10.0)")
    parser.add_argument("--tilt",      type=float, default=45.0,
                        help="Camera tilt in degrees (default: 45.0)")
    args = parser.parse_args()

    if args.save_mask is None:
        args.save_mask = args.image.rsplit(".", 1)[0] + "_mask.jpg"
    cam = CameraConfig(tilt_deg=args.tilt, altitude_m=args.altitude)
    results = detect_stream(
        args.image,
        cam=cam,
        show=args.show,
        save_mask_path=args.save_mask,
    )

    if results["world_path"] is not None:
        print("\nWorld-coordinate centerline path P = {(Xᵢ, Yᵢ)}:")
        for i, (x, y) in enumerate(results["world_path"]):
            print(f"  P[{i:02d}] = ({x:.3f} m, {y:.3f} m)")


