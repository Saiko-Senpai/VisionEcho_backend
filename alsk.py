import cv2
from ultralytics import YOLO

# Load your YOLOv8 model (you can replace with 'yolov8n.pt', 'yolov8m.pt', etc.)
model = YOLO('yolov8x.pt')

# Choose input source:
# 0 = Laptop webcam
# 'http://<your-ip>:8080/video' = IP Webcam from mobile
source = "http://192.168.29.3:4747/video"


# Initialize video capture
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("❌ Error: Couldn't open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run object detection
    results = model(frame, conf=0.45)[0]  # Adjust conf threshold if needed

    # Draw results
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        conf = float(box.conf[0])              # Confidence
        cls = int(box.cls[0])                  # Class ID
        label = model.names[cls]               # Class name

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
