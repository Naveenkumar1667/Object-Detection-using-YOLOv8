**Object Detection Project**

**Overview**

This project implements real-time object detection using Python and the YOLOv8 model. It can detect multiple objects from webcam or video input, displaying bounding boxes and logging detected objects with confidence scores. Optimized for speed and accuracy, it’s suitable for practical applications and further enhancements.

**✨ Features**

✅ Real-time object detection using YOLOv8
✅ Logs detected objects with confidence scores
✅ Supports webcam and video input
✅ Easily extendable for custom datasets and objects
✅ Lightweight and optimized for real-time performance

**⚙️How It Works**

->Input Capture – Captures frames from a webcam or video file.
->Object Detection – Processes each frame through YOLOv8 to predict object classes and bounding boxes.
->Confidence Filtering – Filters detections based on confidence scores for accuracy.
->Display & Logging – Shows bounding boxes and labels on video; logs detected objects with confidence scores.
->Real-time Updates – Repeats for each frame, enabling continuous detection.

**🛠 Dependencies**

->Python 3.8+
->OpenCV
->YOLOv8 (Ultralytics)
->NumPy
->Pandas (optional for logging)

