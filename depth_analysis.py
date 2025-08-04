import cv2
import torch
import numpy as np
import pyttsx3

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv5 model to device
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
yolo_model.to(device)
yolo_model.eval()

# Load MiDaS model to device
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.to(device)
midas.eval()

# Load MiDaS transform
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Define sectors for generic obstacle detection
def get_sectors(width):
    return {
        "left": (0, width // 3),
        "ahead": (width // 3, 2 * width // 3),
        "right": (2 * width // 3, width)
    }

# Generate spoken descriptions of detected obstacles
def generate_descriptions(frame, detections, depth_map):
    height, width = frame.shape[:2]
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    descriptions = []

    for det in detections:
        if det[4] > 0.5:
            label = yolo_model.names[int(det[5])]
            x1, y1, x2, y2 = map(int, det[:4])
            box_depth = depth_normalized[y1:y2, x1:x2]
            if box_depth.size > 0:
                avg_depth = np.mean(box_depth)
                if avg_depth > 0.5:
                    distance_meters = 5 - 5 * avg_depth
                    steps = max(1, int(distance_meters / 0.5))
                    x_center = (x1 + x2) / 2
                    if x_center < width / 3:
                        position = "to your left"
                    elif x_center > 2 * width / 3:
                        position = "to your right"
                    else:
                        position = "ahead"
                    description = f"{label} {position}, approximately {steps} steps away"
                    descriptions.append(description)

    # Check for generic obstacles
    sectors = get_sectors(width)
    for pos, (start, end) in sectors.items():
        sector_depth = depth_normalized[:, start:end]
        if sector_depth.size > 0:
            max_depth = np.max(sector_depth)
            if max_depth > 0.7:
                descriptions.append(f"obstacle {pos}")

    return descriptions


assert torch.cuda.is_available(), "CUDA GPU is not available!"


# Main loop
cap = cv2.VideoCapture(0)
print("📷 Webcam open. Press SPACE for navigation assistance, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press SPACE for navigation assistance / ESC to exit", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO inference on device
        results = yolo_model(frame_rgb)
        detections = results.xyxy[0].to("cpu").numpy()

        # MiDaS depth estimation
        input_image = midas_transform(frame_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.to("cpu").numpy()

        # Generate and speak descriptions
        descriptions = generate_descriptions(frame, detections, depth_map)
        full_description = " and ".join(descriptions) if descriptions else "No obstacles detected."
        print("🧠 Navigation Assistance:", full_description)
        engine.say(full_description)
        engine.runAndWait()

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
