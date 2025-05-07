import torch
import cv2
import numpy as np
import pyrealsense2 as rs
import math
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

# Configure RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Define paths
sam2_checkpoint = "/home/corey/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Load the predictor model
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        results = model(frame, stream=True)

        target_box = None
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if classNames[cls] not in ["bottle", "cup"]:
                    continue

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break  # Process only one target object

        if target_box is not None and not if_init:
            predictor.load_first_frame(frame)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=target_box)
            if_init = True
        elif if_init:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
            for i in range(len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                all_mask = cv2.bitwise_or(all_mask, out_mask)

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

pipeline.stop()
cv2.destroyAllWindows()
