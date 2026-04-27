import ultralytics


import time
import torch
from ultralytics import SAM


print(ultralytics.checks())


# Load model
model_path = "sam2.1_b.pt"

# Run inference function
def run_inference(device, img_url):
    model = SAM(model_path)
    model.to(device)
    start = time.time()
    results = model(img_url)
    end = time.time()
    return results, end - start

img = "https://ultralytics.com/images/bus.jpg"

# CPU
cpu_results, cpu_time = run_inference("cpu", img)

# GPU (only if available)
if torch.cuda.is_available():
    gpu_results, gpu_time = run_inference("cuda", img)
else:
    gpu_results, gpu_time = None, None

# Show one result
cpu_results[0].show()

print(f"CPU time: {cpu_time:.4f} sec")
if gpu_results:
    print(f"GPU time: {gpu_time:.4f} sec")

print("YOLO-SAM inference completed.")