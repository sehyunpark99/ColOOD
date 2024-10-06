import os
import cv2
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import load_model
from utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, \
    print_environment_info
from utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config
from torchinfo import summary


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="/home/shpark/YOLO-OB/config/Config.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str,
                        default="/home/shpark/YOLO-OB/yolo_OB_best_model_ap_0.98669.pth",
                        help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/sun.data",
                        help="Path to data config file (.data)")  # new_mix

    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of CPU threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.4, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.15, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="IOU threshold for non-maximum suppression")
    parser.add_argument('--cuda_idx', type=str, default='0', help='cuda')
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # # Limit PyTorch threads
    torch.set_num_threads(args.n_cpu)
    torch.set_num_interop_threads(args.n_cpu)

    # Define the transformation to be applied to each frame
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def get_predictions(image, model, transform, device):
        image = transform(image).to(device)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(image)
            # Apply non-max suppression on GPU
            outputs = non_max_suppression(outputs, conf_thres=args.conf_thres, iou_thres=args.nms_thres)

        return outputs

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # valid_path = data_config["test"]
    class_names = ["AD", "HP"]

    # Load Model: Yolov3 Darknet based
    # args -> cuda_idx
    model = load_model(args.model, args.weights, args)
    model.to(device)
    model.eval()

    # Open video file
    cap = cv2.VideoCapture("/home/datasets/colon_video/Colon_Video/V0011.mpg")
    start_time = time.time()

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    max_frames = 500 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        print(f'Frame: {frame_count}')
        
        # Get predictions
        detections = get_predictions(frame, model, transform, device)

        # Process detections
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

        # Write the frame with detection boxes
        out.write(frame)
        frame_count += 1

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    latency = end_time - start_time
    print(f'Duration: {latency}')


if __name__ == "__main__":
    run()
