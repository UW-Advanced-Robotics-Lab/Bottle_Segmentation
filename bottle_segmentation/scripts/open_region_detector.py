#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import FastSAM, SAM
import time  # <-- Added for timing display

class SAMSegmentationNode:
    def __init__(self):
        rospy.init_node('sam_segmentation_node')

        self.fast_mode = rospy.get_param("~fast", True)
        self.model_path = rospy.get_param("~model_path", "FastSAM-s.pt" if self.fast_mode else "sam2.1_b.pt")
        self.rgb_topic = rospy.get_param("~image_topic", "/cam_base/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/cam_base/depth/image_rect_raw")
        self.continuous = rospy.get_param("~continuous", True)

        mode_str = "continuous" if self.continuous else "one-shot"
        rospy.loginfo(f"Segmentation node initialized in {mode_str} mode.")

        if self.fast_mode:
            rospy.loginfo("Loading FastSAM model...")
            self.model = FastSAM(self.model_path)
        else:
            rospy.loginfo("Loading SAM model...")
            self.model = SAM(self.model_path)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_depth = None
        self.selected_mask_idx = None
        self.target_depth = None

        rospy.Subscriber(self.rgb_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscribed to {self.rgb_topic} and {self.depth_topic}")

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

    def run_segmentation_loop(self):
        rospy.loginfo("Waiting for image and depth...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (self.latest_image is None or self.latest_depth is None):
            rate.sleep()

        masks, avg_depths, centers = self.run_sam_on_image(self.latest_image, self.latest_depth, preview=True)

        if masks is None or len(masks) == 0:
            rospy.logerr("No masks found for selection.")
            return

        self.selected_mask_idx = self.select_mask(masks, centers)

        if self.selected_mask_idx is None:
            rospy.logerr("No mask was selected.")
            return

        self.target_depth = avg_depths[self.selected_mask_idx]
        rospy.loginfo(f"Target mask selected with average depth {self.target_depth:.2f} m.")

        if self.continuous:
            while not rospy.is_shutdown():
                if self.latest_image is not None and self.latest_depth is not None:
                    self.run_sam_on_image(self.latest_image, self.latest_depth, preview=False)
                    rospy.loginfo("Segmentation completed in continuous mode.")
                rate.sleep()
        else:
            if self.latest_image is not None and self.latest_depth is not None:
                self.run_sam_on_image(self.latest_image, self.latest_depth, preview=False)
            rospy.loginfo("Segmentation completed once. Exiting.")

    def run_sam_on_image(self, image, depth, preview=False):
        try:
            results = self.model(image, verbose=False)
            masks = results[0].masks.data.cpu().numpy()
        except Exception as e:
            rospy.logwarn(f"Segmentation failed or no masks: {e}")
            return None, None, None

        overlay = image.copy()
        centers = []
        avg_depths = []

        foreground_mask_total = np.zeros(image.shape[:2], dtype=np.uint8)
        background_mask_total = np.zeros(image.shape[:2], dtype=np.uint8)

        for idx, mask in enumerate(masks):
            resized_mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_indices = resized_mask.astype(bool)

            if depth.shape[:2] != resized_mask.shape:
                resized_depth = cv2.resize(depth, (resized_mask.shape[1], resized_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                resized_depth = depth

            yx = np.argwhere(mask_indices)
            if yx.size == 0:
                avg_depth = float('nan')
                center = [0, 0]
            else:
                center = np.mean(yx, axis=0).astype(int)
                y, x = center
                region = resized_depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3].flatten()
                valid_depths = region[np.isfinite(region) & (region > 0)]
                avg_depth = np.mean(valid_depths) if valid_depths.size > 0 else float('nan')

            centers.append((x, y))
            avg_depths.append(avg_depth)

            if preview:
                color = np.random.randint(0, 255, size=(3,))
                for c in range(3):
                    overlay[:, :, c] = np.where(mask_indices, color[c], overlay[:, :, c])
            else:
                if idx == self.selected_mask_idx:
                    background_mask_total[mask_indices] = 1
                    continue

                if not np.isnan(avg_depth) and self.target_depth is not None:
                    if avg_depth < self.target_depth - 0.1:
                        background_mask_total[mask_indices] = 1
                    else:
                        foreground_mask_total[mask_indices] = 1


        if preview:
            import matplotlib.pyplot as plt
            self.clicked_index = None

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(overlay)
            for i, (x, y) in enumerate(centers):
                ax.text(x, y, f"{i}", color='white', fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.5))

            def onclick(event):
                x_click, y_click = int(event.xdata), int(event.ydata)
                for idx, (x, y) in enumerate(centers):
                    if abs(x - x_click) < 15 and abs(y - y_click) < 15:
                        self.clicked_index = idx
                        plt.close()

            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.title("Click a mask index to select target object")
            plt.axis("off")
            plt.show()
            return masks, avg_depths, centers

        else:
            if self.continuous:
                return None, None, None  # Don't display, just return

            # One-shot mode: show blended image for 15 seconds
            composite_mask = np.zeros_like(image)
            composite_mask[background_mask_total.astype(bool)] = [0, 0, 255]
            composite_mask[foreground_mask_total.astype(bool)] = [0, 255, 0]
            blended = cv2.addWeighted(image, 0.6, composite_mask, 0.4, 0)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(blended)
            plt.title("Foreground (Green) vs Background (Red)")
            plt.axis("off")
            plt.draw()
            plt.pause(15)
            plt.close()
            rospy.loginfo(f"Segmented {len(masks)} objects. Foreground mask generated.")
            return None, None, None

    def select_mask(self, masks, centers):
        rospy.loginfo("Waiting for user to click on a mask...")
        while self.clicked_index is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        return self.clicked_index


if __name__ == "__main__":
    try:
        SAMSegmentationNode()
    except rospy.ROSInterruptException:
        pass
