import torch
from ultralytics import YOLO

def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    return model, device

def detect_objects(model, image, device):
    result = model(image, conf=0.5, device=device)[0]
    return result
