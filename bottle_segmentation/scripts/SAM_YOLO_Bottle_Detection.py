#!/usr/bin/env python3
import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor
from geometry_msgs.msg import Point, Polygon, Point32, Vector3
from std_msgs.msg import Bool  # ✅ Added

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
              "scissors", "teddy bear", "hair drier", "toothbrush"] # (unchanged - full list omitted here)

# Initialization
bridge = CvBridge()
if_init = False
prev_center = None
prev_time = rospy.Time.now()

# Publishers
mask_pub = rospy.Publisher("/segmentation/mask", Image, queue_size=1)
center_pub = rospy.Publisher("/segmentation/center", Point, queue_size=1)
bbox_pub = rospy.Publisher("/segmentation/bbox", Polygon, queue_size=1)
velocity_pub = rospy.Publisher("/segmentation/velocity", Vector3, queue_size=1)
overlay_pub = rospy.Publisher("/segmentation/overlay_image", Image, queue_size=1)
is_tracking_pub = rospy.Publisher("/segmentation/is_tracking", Bool, queue_size=1)  # ✅ Added publisher

def image_callback(msg):
    global if_init, prev_center, prev_time

    # ✅ Publish current tracking state
    is_tracking_pub.publish(Bool(data=if_init))

    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    original_frame = frame.copy()
    curr_time = rospy.Time.now()

    overlay_frame = original_frame.copy()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        results = model(frame, stream=True)
        target_box = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls < 0 or cls >= len(classNames):
                    continue
                if classNames[cls] not in ["bottle", "cup","frisbee"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
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

            mask_msg = bridge.cv2_to_imgmsg(all_mask, encoding="mono8")
            mask_pub.publish(mask_msg)

            M = cv2.moments(all_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            center_pub.publish(Point(x=cx, y=cy, z=0))

            ys, xs = np.where(all_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x1, y1 = int(np.min(xs)), int(np.min(ys))
                x2, y2 = int(np.max(xs)), int(np.max(ys))
                bbox = Polygon(points=[
                    Point32(x=x1, y=y1), Point32(x=x2, y=y1),
                    Point32(x=x2, y=y2), Point32(x=x1, y=y2)
                ])
                bbox_pub.publish(bbox)

            if prev_center is not None:
                dt = (curr_time - prev_time).to_sec()
                vx = (cx - prev_center[0]) / dt if dt > 0 else 0
                vy = (cy - prev_center[1]) / dt if dt > 0 else 0
                velocity_pub.publish(Vector3(x=vx, y=vy, z=0))
            prev_center = (cx, cy)
            prev_time = curr_time

            color_mask = np.zeros_like(frame, dtype=np.uint8)
            color_mask[all_mask > 0] = (0, 0, 255)  # BGR: RED

            overlay = color_mask
            overlay_frame = cv2.addWeighted(original_frame, 1.0, overlay, 0.5, 0)
            cv2.circle(overlay_frame, (cx, cy), 5, (0, 255, 0), -1)

    overlay_msg = bridge.cv2_to_imgmsg(overlay_frame, encoding="bgr8")
    overlay_pub.publish(overlay_msg)

    cv2.imshow("YOLO + SAM2 Tracking", overlay_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        rospy.signal_shutdown("User requested shutdown")

rospy.Subscriber("/cam_base/color/image_raw", Image, image_callback, queue_size=1, buff_size=2**24)

rospy.loginfo("YOLO + SAM2 tracker node started...")
rospy.spin()
cv2.destroyAllWindows()
