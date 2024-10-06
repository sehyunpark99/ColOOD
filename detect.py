import os
import tqdm
import random
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from preprocessor.base_preprocessor import Transform
from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info

CONF_THRES = 0.15
NMS_THRES = 0.3

def get_detections(image, model, device):
    transform = Transform()
    model.eval()  # Set model to evaluation mode

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = transforms.ToTensor()(image)
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        # Apply non-max suppression on GPU
        outputs = non_max_suppression(outputs, conf_thres=CONF_THRES, iou_thres=NMS_THRES)

    return outputs

def detect_and_crop(frame, model, device):
    detections = get_detections(frame, model, device)

    if detections[0] is not None:
        for detection in detections[0]:
            x1, y1, x2, y2, conf, class_id = detection[:6]
            if conf > 0.5:  # Threshold for confidence
                label = f"{int(class_id)}: {conf:.2f}"
                color = (0, 255, 0)
                # Ensure coordinates are integers and within the frame bounds
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cropped_frame = frame[y1:y2, x1:x2]
                return cropped_frame, [x1, y1, x2, y2]
    return frame, detections[0][0:4] # Return original frame if no detections