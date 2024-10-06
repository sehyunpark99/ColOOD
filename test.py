import os

# Set environment variables for PyTorch thread management
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['Pytorch_NUM_THREADS'] = '1'

import streamlit as st
import time
import cv2
import numpy as np
import argparse
import tempfile
import torch
import torchvision.transforms as transforms
from PIL import Image
from network import get_class_model, get_obj_detect_model
from detect import get_detections, detect_and_crop
from utils.utils import print_environment_info

from classify import classification

# # Limit PyTorch threads
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

def run():
    img = Image.open('/home/shpark/Colood/data/img/test_image.png').convert('RGB')

    # Perform polyp detection and cropping
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj_model = get_obj_detect_model(checkpoint='/home/shpark/Colood/ckpt/yolo_best.pth', device=device, args=None)
    detections = get_detections(img, obj_model, device)
    print(detections)
    cropped_frame = detect_and_crop(img, obj_model, device)
    # result_frame.image(cropped_frame, channels="BGR")

    # Classify the image
    
    class_model = get_class_model(num_classes=2, device=device, checkpoint='/home/shpark/Colood/ckpt/deit_best.pth')
    ood_model = get_class_model(num_classes=2, device=device, checkpoint='/home/shpark/Colood/ckpt/ood_best.ckpt')
    classification_result, confidence = classification(class_model, ood_model, cropped_frame, device)
    print(classification_result, confidence)


if __name__ == "__main__":
    run()

