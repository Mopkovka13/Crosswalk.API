import torch
from Linear import Linear


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)