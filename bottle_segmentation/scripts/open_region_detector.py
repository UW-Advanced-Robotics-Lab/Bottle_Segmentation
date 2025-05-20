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

