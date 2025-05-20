#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import FastSAM, SAM

class SAMSegmentationNode:
    def __init__(self):
        rospy.init_node('sam_segmentation_node')

        self.fast_mode = rospy.get_param("~fast", True)
        self.run_once = rospy.get_param("~run_once", False)
        self.model_path = rospy.get_param("~model_path", "FastSAM-s.pt" if self.fast_mode else "sam2.1_b.pt")
        self.rgb_topic = rospy.get_param("~image_topic", "/cam_base/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/cam_base/depth/image_rect_raw")

        if self.fast_mode:
            rospy.loginfo("Loading FastSAM model...")
            self.model = FastSAM(self.model_path)
        else:
            rospy.loginfo("Loading SAM model...")
            self.model = SAM(self.model_path)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_depth = None

        rospy.Subscriber(self.rgb_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscribed to {self.rgb_topic} and {self.depth_topic}")

        if self.run_once:
            self.run_segmentation_once()
        else:
            self.run_segmentation_loop()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth_image.copy()
        except Exception as e:
            rospy.logerr(f"Failed to convert depth image: {e}")

    def run_segmentation_once(self):
        rospy.loginfo("Waiting for image and depth...")
        rate = rospy.Rate(10)
        timeout = rospy.Time.now() + rospy.Duration(10)

        while not rospy.is_shutdown() and (self.latest_image is None or self.latest_depth is None):
            if rospy.Time.now() > timeout:
                rospy.logerr("Timeout waiting for image and depth.")
                return
            rate.sleep()

        rospy.loginfo("Image and depth received. Running segmentation...")
        self.run_sam_on_image(self.latest_image, self.latest_depth, display=True)

    def run_segmentation_loop(self):
        rospy.loginfo("Running segmentation loop (no display)...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.latest_image is not None and self.latest_depth is not None:
                self.run_sam_on_image(self.latest_image, self.latest_depth, display=False)
            rate.sleep()

    def run_sam_on_image(self, image, depth, display=True):
        try:
            results = self.model(image)
            masks = results[0].masks.data.cpu().numpy()
        except Exception as e:
            rospy.logwarn(f"Segmentation failed or no masks: {e}")
            return

        overlay = image.copy()
        text_locations = []
        avg_depths = []

        for idx, mask in enumerate(masks):
            resized_mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_indices = resized_mask.astype(bool)

            # Color overlay (for display only)
            if display:
                color = np.random.randint(0, 255, size=(3,))
                colored_mask = np.zeros_like(overlay)
                for i in range(3):
                    colored_mask[:, :, i] = resized_mask * color[i]
                overlay[mask_indices] = cv2.addWeighted(overlay[mask_indices], 0.1, colored_mask[mask_indices], 0.9, 0)

            # Resize depth to match mask if needed
            if depth.shape[:2] != resized_mask.shape:
                resized_depth = cv2.resize(depth, (resized_mask.shape[1], resized_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                resized_depth = depth

            # Compute average depth
            depth_values = resized_depth[mask_indices]
            valid_depths = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
            avg_depth = np.mean(valid_depths) if valid_depths.size > 0 else float('nan')

            avg_depths.append(avg_depth)
            yx = np.argwhere(mask_indices)
            center = np.mean(yx, axis=0).astype(int) if yx.size > 0 else [0, 0]
            text_locations.append((center[1], center[0]))  # x, y

            #rospy.loginfo(f"Mask {idx}: Avg depth = {avg_depth:.2f} m")

        if display:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(overlay)
            for (x, y), depth_val in zip(text_locations, avg_depths):
                if not np.isnan(depth_val):
                    plt.text(x, y, f"{depth_val:.2f} m", color='white',
                             fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
            plt.axis("off")
            plt.title("Segmentation with Depth Overlay")
            plt.show(block=False)
            plt.pause(15)
            plt.close()
        else:
            rospy.loginfo(f"Segmented {len(masks)} objects (no display).")

if __name__ == "__main__":
    try:
        SAMSegmentationNode()
    except rospy.ROSInterruptException:
        pass
