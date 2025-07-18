import os
import json
import random

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from huggingface_hub import hf_hub_download

import numpy as np
from PIL import Image
import torch
from skimage import measure


def preprocess_outputs(output):
    input_scores = [x["score"] for x in output]
    input_labels = [x["label"] for x in output]
    input_boxes = []
    for i in range(len(output)):
        input_boxes.append([*output[i]["box"].values()])
    input_boxes = [input_boxes]
    return input_scores, input_labels, input_boxes


def resize_image(image, input_size):
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    return image


def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def box_prompt(masks, bbox):
    h = masks.shape[1]
    w = masks.shape[2]

    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    # IoUs = torch.zeros(len(masks), dtype=torch.float32)
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index].cpu().numpy(), max_iou_index


def point_prompt(masks, points, point_label):  # numpy 处理
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]

    onemask = np.zeros((h, w))
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation['segmentation']
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and point_label[i] == 1:
                onemask[mask] = 1
            if mask[point[1], point[0]] == 1 and point_label[i] == 0:
                onemask[mask] = 0
    onemask = onemask >= 1
    return onemask, 0


def show_masks_on_image(image, masks):
    # Create a mask image (assuming binary mask)
    #image_with_mask = Image.open(image_path).convert("RGBA")
    image_with_mask = image.convert("RGBA")

    for mask in masks:
        #mask = mask.cpu().numpy()

        height, width = mask.shape
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]

        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)

        width, height = image_with_mask.size
        mask_image = mask_image.resize((width, height))

        # Overlay the mask on the image
        image_with_mask = Image.alpha_composite(
            image_with_mask,
            mask_image)

    # Display the result
    image_with_mask.show()


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(image, masks):
    # Create a mask image (assuming binary mask)
    #image_with_mask = Image.open(image_path).convert("RGBA")
    image_with_mask = image.convert("RGBA")

    for mask in masks:
        #mask = mask.cpu().numpy()

        height, width = mask.shape
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]

        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)

        width, height = image_with_mask.size
        mask_image = mask_image.resize((width, height))

        # Overlay the mask on the image
        image_with_mask = Image.alpha_composite(
            image_with_mask,
            mask_image)

    # Display the result
    return image_with_mask


def show_binary_mask(masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    fig, ax = plt.subplots(figsize=(15, 15))
    idx = scores.tolist().index(max(scores))
    mask = masks[idx].cpu().detach()
    ax.imshow(np.array(masks[0,:,:]), cmap='gray')
    score = scores[idx]
    ax.title.set_text(f"Score: {score.item():.3f}")
    ax.axis("off")
    plt.show()


def generate_image(
    image,
    mask,
    prompt,
    pipe,
    seed,
    guidance_scale=7.5,
    negative_prompt="low resolution, ugly",
    num_inference_steps=120,
    strength=1,
):
    # resize for inpainting
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))

    generator = torch.Generator(device).manual_seed(seed)

    result = pipe(
        image=in_image,
        mask_image=in_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
    )
    result = result.images[0]

    return result.resize((w, h))


def show_generated_image(generated_image):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(generated_image, interpolation="nearest")
    plt.tight_layout()


def make_birthday_card(img, filename=None):
    h, w, _ = np.array(generated_image).shape
    x = [randint(50, w) for x in range(50)]
    y = [randint(50, h) for x in range(50)]
    color_list = [
        "darkmagenta",
        "orangered",
        "greenyellow",
        "lightpink",
        "orange",
        "gold",
        "forestgreen",
        "blue",
        "mediumturquoise",
        "magenta",
        "dodgerblue",
    ]
    c = [choice(color_list) for x in range(50)]

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.text(
        x=(w / 12),
        y=((9 * h) / 10),
        s="Happy Birthday!",
        c="tomato",
        path_effects=[pe.withStroke(linewidth=3, foreground="oldlace")],
        fontsize=58,
        fontname="serif",
    )
    plt.scatter(x=x, y=y, s=50, c=c, marker=(5, 1), alpha=0.7)
    plt.axis("off")
    if filename:
        plt.savefig(filename)
    plt.show()


def show_boxes_and_labels_on_image(raw_image, boxes, labels, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for i, box in enumerate(boxes):
        show_box(box, plt.gca())
        plt.text(
            x=box[0],
            y=box[1] - 12,
            s=f"{labels[i]}: {scores[i]:,.4f}",
            c="beige",
            path_effects=[pe.withStroke(linewidth=4, foreground="darkgreen")],
        )
    plt.axis("on")
    plt.show()


def show_multiple_masks_on_image(raw_image, masks, scores):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.array(raw_image))
    for idx in range(len(masks[0])):
        mask = masks[0][idx][0].cpu().detach()
        show_mask(mask, ax, random_color=True)
    ax.axis("off")
    plt.show()


def make_coco_boxes(detections_boxes):

    """Convert torch tensor Pascal VOC bboxes to COCO format for Comet annotations"""

    list_boxes = detections_boxes
    coco_boxes = [
        [
            list_boxes[0],
            list_boxes[1],
            (list_boxes[2] - list_boxes[0]),
            (list_boxes[3] - list_boxes[1]),
        ]
    ]
    return coco_boxes


def make_bbox_annots(input_scores, input_labels, input_boxes, image_metadata):

    if len(input_boxes[0]) == 0:
        return None

    annotations = [
        {
            "name": "bbox annots",
            "data": [],
            "metadata": image_metadata,
        }
    ]

    for i in range(len(input_boxes[0])):
        annotations[0]["data"].append(
            {
                "label": input_labels[i],
                "score": round((input_scores[i] * 100), 2),
                # bboxes in pascal_voc format, return in coco format for Comet annotations
                "boxes": make_coco_boxes(input_boxes[0][i]),
                "points": None,
            }
        )
    annotations = json.loads(json.dumps(annotations))
    return annotations


def make_mask_annots(input_masks,
                     input_labels,
                     image_metadata
                    ):

    if len(input_masks[0]) == 0:
        return None

    annotations = [
        {
            "name": "mask annots",
            "data": [],
            "meta_data": image_metadata,
        }
    ]

    for i in range(len(input_masks)):
        annotations[0]["data"].append(
            {
                "label": input_labels[i],
                "score": 100.00,
                "points": make_sam_mask(input_masks[i]),
            }
        )
    annotations = json.loads(json.dumps(annotations))
    return annotations


def get_model_path(model_key):
    """ Get the local path of the model or download it if not available.

    Args:
        name (_type_): options: fastsam, mobilesam, sam-vit-h

    Returns:
        _type_: _description_
    """
    MODEL_PATHS = {
        "fastsam": {
            "repo_id": "Uminosachi/FastSAM",
            "filename": "FastSAM-s.pt",
            "local_filename": "fastsam-s.pt",
            "local_dir": "models/"
        },
        "mobilesam": {
            "repo_id": "Uminosachi/MobileSAM",
            "filename": "mobile_sam.pt",
            "local_filename": "mobile_sam.pt",
            "local_dir": "models/"
        }
    }

    if model_key not in MODEL_PATHS:
        raise ValueError(f"Model '{model_key}' not found in MODEL_PATHS.")

    model_info = MODEL_PATHS[model_key]
    local_dir = model_info["local_dir"]
    local_path = os.path.join(local_dir, model_info["local_filename"])

    os.makedirs(local_dir, exist_ok=True)

    # Download if not already present
    if not os.path.exists(local_path):
        print(f"Downloading '{model_key}' to '{local_path}'...")
        downloaded_path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            cache_dir=local_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        os.rename(downloaded_path, local_path)

    return local_path