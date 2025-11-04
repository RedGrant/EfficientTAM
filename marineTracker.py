import os
import cv2
import torch
import numpy as np
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

# -----------------------------
# CONFIGURATION
# -----------------------------
video_path = ("/media/pedroguedes/UBUNTU_SSD/PHD/Thesis/Software/Tracking/yolov8_model/input_videos/"
              "adaptive_gaussian_11_opening_KNN/Fish_net_1200.mp4")
checkpoint = "../checkpoints/efficienttam_ti.pt"   # use FP32 version if converted
model_cfg = "configs/efficienttam/efficienttam_ti.yaml"
output_dir = "./outputs_test"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD VIDEO INFO
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.release()
print(f"Video loaded: {video_path}")
print(f"Resolution: {W}x{H}, FPS: {fps}")

# -----------------------------
# BUILD MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

predictor = build_efficienttam_video_predictor(model_cfg, checkpoint, device=device)

# -----------------------------
# INIT STATE FROM VIDEO
# -----------------------------
inference_state = predictor.init_state(video_path=video_path, offload_video_to_cpu=True,   # keep raw frames on CPU
    offload_state_to_cpu=False)  # keep intermediate features on GPU)

# -----------------------------
# ADD DUMMY INIT POINTS
# -----------------------------
# We need *something* to start propagation — one fake click in the first frame.
# Later, this will come from YOLO.
points = np.array([[W//2, H//2]], dtype=np.float32)  # center pixel
labels = np.array([1], np.int32)

#predictor.add_new_points_or_box(
#    inference_state=inference_state,
#    frame_idx=0,
#    obj_id=0,
#    points=points,
#    labels=labels,
#)
#print("Initialized dummy click at video center.")

frame_idx = 7  # YOLO detections come from frame 7
annotation_path = ("/media/pedroguedes/UBUNTU_SSD/PHD/Thesis/Software/Tracking/yolov8_model/"
                   "video_annotations/Fish_net_1200/7_Fish_net_1200.txt")  # <-- change this!

if not os.path.exists(annotation_path):
    raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

# Read YOLO annotation lines for frame 7
with open(annotation_path, "r") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

print(f"Loaded {len(lines)} detections from {annotation_path}")

for line in lines:
    # Expected YOLO format: class x_center y_center width height track_id
    parts = line.split()
    if len(parts) < 6:
        print(f"Skipping malformed line: {line}")
        continue

    cls, xc, yc, w_box, h_box, track_id = map(float, parts)

    # Convert normalized [0,1] → pixel coordinates
    xc_pix = xc * W
    yc_pix = yc * H
    w_pix = w_box * W
    h_pix = h_box * H

    # Use center point of box as click input
    points = np.array([[xc_pix, yc_pix]], dtype=np.float32)
    labels = np.array([1], np.int32)

    # Add this object to EfficientTAM for tracking
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=int(track_id),  # use YOLO track_id to keep consistency
        points=points,
        labels=labels,
    )

    print(f"✅ Added object {int(track_id)} from frame {frame_idx}: "
          f"class={int(cls)}, center=({xc_pix:.1f},{yc_pix:.1f}), "
          f"size=({w_pix:.1f},{h_pix:.1f})")

print(f"Initialized from YOLO boxes on frame {frame_idx}.")


# -----------------------------
# RUN PROPAGATION
# -----------------------------
video_segments = {}
print("Running segmentation propagation...")
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    masks = [(out_mask_logits[i] > 0.0).cpu().numpy() for i in range(len(out_obj_ids))]
    video_segments[out_frame_idx] = masks
print("Propagation complete!")

# -----------------------------
# SAVE SEGMENTED VIDEO
# -----------------------------
def overlay_mask(frame, mask, color):
    overlay = frame.copy()
    color = np.array(color, dtype=np.uint8)
    # Expand mask to 3 channels
    mask_3c = np.stack([mask]*3, axis=-1)
    # Blend: use np.where
    overlay = np.where(mask_3c, (overlay * 0.4 + color * 0.6).astype(np.uint8), overlay)
    return overlay

cap = cv2.VideoCapture(video_path)
out_path = os.path.join(output_dir, "efficienttam_segmentation.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

colors = [(255,0,0), (0,255,0), (0,0,255)]
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame_idx in video_segments:
        for j, mask in enumerate(video_segments[frame_idx]):
            frame_rgb = overlay_mask(frame_rgb, mask.squeeze() > 0, colors[j % len(colors)])
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)
    frame_idx += 1

cap.release()
writer.release()
print(f"Segmented video saved at: {out_path}")
