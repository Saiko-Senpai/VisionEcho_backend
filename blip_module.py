from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
from deepface import DeepFace

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_scene_caption_from_image(image, processor, model):
    image = image.convert('RGB')
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def get_non_neutral_emotion(frame):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if results and isinstance(results, list):
            dominant_emotion = results[0].get('dominant_emotion', '').lower()
            if dominant_emotion != 'neutral':
                return dominant_emotion
    except Exception as e:
        pass  # silently fail on detection errors
    return None

def caption_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return generate_scene_caption_from_image(pil_img, processor, model)

# Capture from webcam
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("📷 Webcam open. Press SPACE to capture an image and get its scene description.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Press SPACE to capture / ESC to exit", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            caption = generate_scene_caption_from_image(pil_img, processor, model)
            print("🧠 Scene Caption:", caption)

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
