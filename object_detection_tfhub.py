import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

# Load the model
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# COCO labels for the SSD Mobilenet V2 model
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Classes to detect
TARGET_CLASSES = {'knife', 'gun'}


def detect_objects(image):
    img_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    img_tensor = tf.expand_dims(img_tensor, 0)
    detections = detector(img_tensor)
    return detections


def process_image(file_path):
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detect_objects(image_rgb)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    height, width, _ = image.shape

    for i in range(len(detection_scores)):
        if detection_scores[i] >= 0.3:
            class_id = detection_classes[i]
            class_name = COCO_LABELS.get(class_id, 'Unknown')
            if class_name in TARGET_CLASSES:
                ymin, xmin, ymax, xmax = detection_boxes[i]
                (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv2.putText(image, class_name, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    while True:
        file_path = filedialog.askopenfilename(title="Select an Image",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            break
        process_image(file_path)


if __name__ == "__main__":
    open_file_dialog()
