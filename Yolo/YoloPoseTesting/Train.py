
from ultralytics import YOLO
import os 
import pandas as pd
import cv2
import numpy as np

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n-cls.pt") 

results = model.train(
    data=r"C:\Capstone\Yolo\YoloPoseTesting\dataset\train",  # path to folder with class subfolders
    epochs=2,    # start with 2, you can adjust later
    imgsz=320,    # smaller image size → faster training
    batch=2       # small batch for CPU
)

# Save trained model
model_path = "C:\\Capstone\\Yolo\\YoloPoseTesting\\runs\\train\\my_yolo_model.pt"
model.save(model_path)

print("Training complete!")
