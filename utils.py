import cv2
import torch

def preprocess_frame(self, frame, device):
    # make it gray
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 25% downsampling
    frame = frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    # thresholding to make everything black and white
    _, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
    # 0-1 image
    return torch.tensor(frame, dtype=torch.float32, device=device).div_(255)
    