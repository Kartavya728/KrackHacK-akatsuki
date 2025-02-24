# 🚦 YOLO-Based Traffic Detection System

## 📌 Concise Summary of the Code
This repository contains a **YOLO-based deep learning model** integrated with **Streamlit** for an interactive UI. It detects **vehicles, pedestrians, traffic lights, lanes, and more** in images and videos.

---
## Data Set: https://datasetninja.com/bdd100k#download
## PPT : https://gamma.app/docs/Real-Time-Object-Detection-in-Traffic-Systems-A-YOLO-Based-Approa-937lb3fj0t2soqr?mode=doc
## 🔑 Key Features

✅ **Two Modes:**  
- **Moving Camera Model**: Assists drivers by detecting traffic lights and driving areas in real-time.  
- **Static Camera Model**: Analyzes pre-recorded videos or live feeds to extract traffic data.  

✅ **Supports Multiple Input Methods:**  
- Upload a **video**  
- Use a **live camera feed**  
- Provide a **YouTube link**  

✅ **Automated Traffic Analysis:**  
- Helps in traffic monitoring and urban planning by counting vehicles and analyzing road conditions.  

---
## ⚙️ Tech Stack
- **Deep Learning:** YOLO (You Only Look Once) by Ultralytics  
- **Frontend:** Streamlit   
- **Computer Vision:** OpenCV, VidGear (CamGear)  
- **Data Processing:** NumPy  
- **Model Training:** Custom YOLO model trained on 70,023 images and validated with 9,977 images  

## 🛠 Technical Overview
This code implements a **YOLO-based object detection system** with a **Streamlit frontend** for real-time and offline video analysis. It utilizes:

- **Ultralytics YOLO** to detect objects like vehicles, pedestrians, traffic signals, and lanes.  
- **OpenCV (`cv2`)** and **VidGear (`CamGear`)** for video processing.  
- **Annotator Class** to draw bounding boxes around detected objects.  
- **NumPy**, **tempfile**, and **time.sleep()** for optimized performance and smoother processing.  
- The model was trained on **70,023 images** and validated with **9,977 images**, ensuring **robust detection accuracy** for various traffic scenarios.  

## Model Analysis
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/F1_curve.png)
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/PR_curve.png)
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/P_curve.png)
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/R_curve.png)
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/confusion_matrix.px)
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/confusion_matrix_normalized.png)
![alt](https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Data%20cleaning/traffic_detection/yolov8m_finetune4/results.png)

---

## 🚀 Steps to Run the Code
## Either open the folder model-training and run the files moving_camera.py and static_camera.py or 
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Kartavya728/KrackHacK-akatsuki
```
### 2️⃣ Install Dependencies
Navigate to the repository and install the required dependencies:
```bash 
pip install ultralytics opencv-python streamlit numpy vidgear
```
Check if all dependencies are downloaded else download the required library
### 3️⃣ Run the Application
```bash
streamlit run app.py
```
### Ensure that you change the paths accordingly..
### Video for running the code-- https://www.dailymotion.com/video/k1SWKtGbifZzs5CxqKY
## 📁 Repository Structure
KrackHacK-akatsuki:.
├───Data cleaning
│   └───traffic_detection
│       ├───yolov8m_finetune4
│       │   └───weights
│       └───yolov8m_finetune42
├───Model Training
└───Website-rendering
## Weights: https://github.com/Kartavya728/KrackHacK-akatsuki/blob/main/Website-rendering/best.pt
