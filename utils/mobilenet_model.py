import os, argparse
from pathlib import Path
from typing import Tuple, Dict
import random
from collections import deque
import cv2
import numpy as np
from time import perf_counter

from .model import Model


class Detection:
    def __init__(self, xmin, ymin, xmax, ymax, score, id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = id

    def bottom_left_point(self):
        return self.xmin, self.ymin

    def top_right_point(self):
        return self.xmax, self.ymax

    def get_coords(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

class MobileNetModel(Model):
	def __init__(self, model_path, device, image_size=640):
		super().__init__(model_path, device, image_size)
		self.name = "MobileNet"
		
	def put(self, infer_request, image, resized_image):
		outputs = infer_request.get_output_tensor().data[0]

		detections = [Detection(xmin, ymin, xmax, ymax, score, label)
                for _, label, score, xmin, ymin, xmax, ymax in outputs[0]] 

		self.outputs.append((detections,image,resized_image))

		self.labels = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
						"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
						"sheep", "sofa", "train", "tvmonitor" ]

	def preprocess(self, image, width, height):
	
		image = cv2.resize(image, (width, height))

		# Convert HWC to CHW
		image = image.transpose(2, 0, 1)
		image = image.astype(np.float32) 
		image = np.expand_dims(image, axis=0)

		return image

	def plot_one_box(self, box:np.ndarray, img:np.ndarray,  color=(255,0,0), line_thickness:int = 1):
		tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
		c1, c2 = (int(box.xmin), int(box.ymin)), (int(box.xmax), int(box.ymax))
		cv2.rectangle(img, c1, c2, color=color, thickness=tl, lineType=cv2.LINE_AA)
		
		tf = max(tl - 1, 1)  # font thickness
		label = self.labels[int(box.id)]
		distance = self.cap.get_distance(c1[0] + (c2[0]-c1[0])/2, c1[1] + (c2[1]-c1[1])/2)
		if distance is not None:
			label = f'{self.labels[int(box.id)]} at {distance:.2f}m'
		else:
			label = f'{self.labels[int(box.id)]} {int(box.score*100)}%'

		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

		return img

	def postprocess(self, outputs, threshold=0.5):

		detections, image, resized_image = outputs
		detections = [d for d in detections if d.score > threshold]
		for detection in detections:
			detection.xmin *= image.shape[1]
			detection.xmax *= image.shape[1]
			detection.ymin *= image.shape[0]
			detection.ymax *= image.shape[0]
			self.plot_one_box(detection, image)
    
		return image


	
