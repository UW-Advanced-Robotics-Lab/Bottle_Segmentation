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

        self.fast_mode = rospy.get_param("~fast", False)
        self.run_once = rospy.get_param("~run_once", True)
        self.model_path = rospy.get_param("~model_path", "FastSAM-s.pt" if self.fast_mode else "sam2.1_b.pt")
        self.topic_name = rospy.get_param("~image_topic", "/cam_base/color/image_raw")

        # Load model
        if self.fast_mode:
            rospy.loginfo("Loading FastSAM model...")
            self.model = FastSAM(self.model_path)
        else:
            rospy.loginfo("Loading SAM model...")
            self.model = SAM(self.model_path)

        self.bridge = CvBridge()
        self.latest_image = None

        rospy.Subscriber(self.topic_name, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscribed to {self.topic_name}")

        if self.run_once:
            self.run_segmentation_once()
        else:
            self.run_segmentation_loop()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def run_segmentation_once(self):
        rospy.loginfo("Waiting for image...")
        rate = rospy.Rate(10)
        timeout = rospy.Time.now() + rospy.Duration(10)

        while not rospy.is_shutdown() and self.latest_image is None:
            if rospy.Time.now() > timeout:
                rospy.logerr("Timeout waiting for image.")
                return
            rate.sleep()

        rospy.loginfo("Image received. Running segmentation...")
        self.run_sam_on_image(self.latest_image, display=True)

    def run_segmentation_loop(self):
        rospy.loginfo("Running segmentation loop (no display)...")
        rate = rospy.Rate(1000)  # Max 10 Hz; you can increase if needed
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                self.run_sam_on_image(self.latest_image, display=False)
            rate.sleep()

    def run_sam_on_image(self, image, display=True):
        try:
            results = self.model(image)
            masks = results[0].masks.data.cpu().numpy()
        except Exception as e:
            rospy.logwarn(f"Segmentation failed or no masks: {e}")
            return

        if display:
            overlay = image.copy()
            for mask in masks:
                resized_mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                color = np.random.randint(0, 255, size=(3,))
                colored_mask = np.zeros_like(overlay)
                for i in range(3):
                    colored_mask[:, :, i] = resized_mask * color[i]
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 1.0, 0)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(overlay)
            plt.axis("off")
            plt.title("Segmentation Overlay")
            plt.show(block=False)
            plt.pause(15)  # Show for 3 seconds
            plt.close()   # Close the figure to release resources
        else:
            rospy.loginfo(f"Segmented {len(masks)} objects (no display).")

if __name__ == "__main__":
    try:
        SAMSegmentationNode()
    except rospy.ROSInterruptException:
        pass
