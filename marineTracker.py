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

predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=0,
    points=points,
    labels=labels,
)

print("Initialized dummy click at video center.")

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
