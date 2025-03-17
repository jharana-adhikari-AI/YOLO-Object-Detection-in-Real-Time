# 📌 YOLO Object Detection in Real-Time
This project implements real-time object detection using YOLO (You Only Look Once) with Python, OpenCV, and the Ultralytics library. It allows you to train a YOLO model, detect objects in real-time, and export the trained model for deployment.

## 1️⃣ What is YOLO?
YOLO (You Only Look Once) is a state-of-the-art deep learning algorithm for real-time object detection. Unlike traditional object detection models that scan an image multiple times, YOLO processes the image in a single pass, making it extremely fast and efficient.

## Why YOLO?
Real-time speed (Processes images in milliseconds)
High accuracy (Used in security, automotive, and robotics)
Single-pass detection (Detects multiple objects in one go)
Example Use Cases: Self-driving cars, surveillance systems, face recognition, traffic monitoring, and more.

## 2️⃣ Libraries Used
Library	Purpose: 
- Ultralytics	Provides the YOLOv8 model for easy training and inference
- OpenCV	Handles video processing and image manipulation
- NumPy	Efficient array and matrix operations
- TensorFlow	Supports deep-learning computations

  
#### How These Libraries Work Together
- Ultralytics loads and trains the YOLO model.
- OpenCV captures video frames and displays detected objects.
- NumPy processes numerical data for model input.
- TensorFlow provides deep learning support for the model.


## 3️⃣ What This Project Does
This project allows you to:
✅ Train a YOLOv8 model on a dataset (e.g., COCO dataset)
✅ Perform real-time object detection on live video
✅ Export the model to ONNX format for deployment


## 4️⃣ How YOLO Works in This Project
### - Training Phase:
     -  The YOLO model is trained on labeled images to recognize objects.
     -  It learns object features, positions, and categories.

### - Detection Phase: 
     -  The model takes a frame from a video.
     -  It identifies objects and their bounding boxes in a single pass.
     -  The detected objects are labeled and displayed in real time.

### - Exporting Phase: 
     - The trained model is converted into ONNX format for easy deployment.
