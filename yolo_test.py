import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

folder_path = r"C:\Users\voice\Documents\YOLOv8_Project\car_test"

image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    results = model(image_path)

    for result in results:
        img = result.plot()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(image_file)  
    plt.axis("off") 
    plt.show()