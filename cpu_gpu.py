import cv2
import torch
import numpy as np
import pyttsx3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")