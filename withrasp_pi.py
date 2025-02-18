import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(1, cv2.CAP_V4L2)

cap1.set(3, 640)
cap1.set(4, 480)
cap2.set(3, 640)
cap2.set(4, 480)

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    results1 = model(frame1)
    results2 = model(frame2)

    for result in results1:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for result in results2:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - Camera 1", frame1)
    cv2.imshow("YOLOv8 - Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()