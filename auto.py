import cv2
import time
from blip_module import caption_from_frame
from blip_module import get_non_neutral_emotion

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("⏱️ Capturing every 3 seconds. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        caption = caption_from_frame(frame)
        emotion = get_non_neutral_emotion(frame)
        if emotion != 'None':
            print("🧠 Scene Caption:", caption, emotion)
        else:
            print("🧠 Scene Caption:", caption)

        time.sleep(3)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
