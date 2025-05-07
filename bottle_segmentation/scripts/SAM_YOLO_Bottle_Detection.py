#!/usr/bin/env python3
import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

# Initialize ROS node
rospy.init_node("yolo_sam2_tracker", anonymous=True)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Load SAM2 predictor
sam2_checkpoint = "/home/corey/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# YOLO class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]

# Initialization
bridge = CvBridge()
if_init = False

# Inference context
inference_context = torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)

def image_callback(msg):
    global if_init

    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        results = model(frame, stream=True)
        target_box = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if classNames[cls] not in ["bottle", "cup"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, classNames[cls], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break

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

        cv2.imshow("YOLO + SAM2 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("User requested shutdown")

# Subscribe to the image topic
rospy.Subscriber("/cam_base/color/image_raw", Image, image_callback, queue_size=1, buff_size=2**24)

# Keep node alive
rospy.loginfo("YOLO + SAM2 tracker node started...")
rospy.spin()
cv2.destroyAllWindows()
