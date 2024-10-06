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
import torch
from PIL import Image, ImageDraw
from network import get_class_model, get_obj_detect_model
from detect import get_detections, detect_and_crop
from classify import classification
from streamlit_drawable_canvas import st_canvas

# Initialize session state variables
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'cropped_frame' not in st.session_state:
    st.session_state.cropped_frame = None
if 'canvas_result' not in st.session_state:
    st.session_state.canvas_result = None

def crop_image(image, coords):
    top_left, bottom_right = coords
    cropped_image = image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    return cropped_image

def display_classification_result(result, confidence):
    st.markdown(
        f"""
        <div style="border:2px solid #4CAF50; padding: 16px; border-radius: 10px; text-align: center; font-size: 24px; background-color: #f9f9f9;">
            <b>Classification Result:</b> <span style="color: #4CAF50;">{result}</span><br>
            <b>Confidence:</b> <span style="color: #4CAF50;">{confidence}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def process_video(selected_video, video_dir, device, class_model, ood_model, crop_mode):
    video_path = os.path.join(video_dir, selected_video)
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        st.write(f"Processing {selected_video}...")

        stframe = st.empty()
        result_frame = st.empty()

        current_frame = 0
        stop_button = st.button("Stop")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Display the original frame
            stframe.image(frame, channels="BGR")
            time.sleep(0.1)  # Simulate real-time display

            current_frame += 1
            print(f'Current Frame: {current_frame}')

            # Check if the stop button is pressed
            if stop_button:
                st.session_state.last_frame = frame  # Store the last read frame
                st.write(f"Video {selected_video} stopped at frame {current_frame}.")
                break

        cap.release()

        if st.session_state.last_frame is not None:
            st.write("Captured frame at the moment of stop:")
            stframe.image(st.session_state.last_frame, channels="BGR")

            # Convert the last frame to PIL image
            img = Image.fromarray(cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB))

            if crop_mode == "Object Detection":
                # Perform polyp detection and cropping
                obj_model = get_obj_detect_model(checkpoint='/home/shpark/Colood/ckpt/yolo_best.pth', device=device, args=None)
                cropped_frame, bbox = detect_and_crop(img, obj_model, device)
                
                # Draw bounding box on the original frame
                bbox = tuple(bbox[0][0:4])
                img_with_bbox = draw_bounding_box(img, bbox)
                st.image(img_with_bbox, caption="Frame with Bounding Box", use_column_width=True)
                
                # Display cropped frame
                cropped_frame_np = np.array(cropped_frame)
                result_frame.image(cropped_frame_np, channels="RGB")

                st.session_state.cropped_frame = cropped_frame

            elif crop_mode == "Click to Crop":
                st.write("Draw a rectangle on the image to crop.")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="red",
                    background_image=img,
                    update_streamlit=True,
                    height=img.height,
                    width=img.width,
                    drawing_mode="rect",
                    key="canvas",
                )
                
                if canvas_result.json_data is not None:
                    shapes = canvas_result.json_data["objects"]
                    if len(shapes) > 0:
                        shape = shapes[0]
                        top_left = (shape["left"], shape["top"])
                        bottom_right = (shape["left"] + shape["width"], shape["top"] + shape["height"])
                        st.session_state.canvas_result = (top_left, bottom_right)
                
                if st.session_state.canvas_result:
                    cropped_frame = crop_image(img, st.session_state.canvas_result)
                    st.image(cropped_frame, caption="Cropped Image", use_column_width=True)
                    st.session_state.cropped_frame = cropped_frame

            # Proceed to classification only if the user clicks the "Classify" button
            if st.button("Classify"):
                if st.session_state.cropped_frame is not None:
                    # Classify the image
                    classification_result, confidence = classification(class_model, ood_model, st.session_state.cropped_frame, device)
                    display_classification_result(classification_result, confidence)
                else:
                    st.write("No cropped frame available for classification.")
        else:
            st.write(f"No frames to process in {selected_video}.")
    else:
        st.write(f"Failed to open {selected_video}.")

def process_uploaded_image(uploaded_file, device, class_model, ood_model, crop_mode):
    if uploaded_file is not None:
        # Convert the uploaded file to a PIL image
        img = Image.open(uploaded_file).convert('RGB')

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        if crop_mode == "Object Detection":
            # Perform polyp detection and cropping
            obj_model = get_obj_detect_model(checkpoint='/home/shpark/Colood/ckpt/yolo_best.pth', device=device, args=None)
            cropped_frame, bbox = detect_and_crop(img, obj_model, device)
            
            # Draw bounding box on the original image
            bbox = tuple(bbox[0][0:4])
            img_with_bbox = draw_bounding_box(img, bbox)
            st.image(img_with_bbox, caption="Uploaded Image with Bounding Box", use_column_width=True)
            
            # Display cropped frame
            cropped_frame_np = np.array(cropped_frame)
            st.image(cropped_frame_np, caption='Cropped Image', use_column_width=True)

            st.session_state.cropped_frame = cropped_frame

        elif crop_mode == "Click to Crop":
            st.write("Draw a rectangle on the image to crop.")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="red",
                background_image=img,
                update_streamlit=True,
                height=img.height,
                width=img.width,
                drawing_mode="rect",
                key="canvas",
            )
            
            if canvas_result.json_data is not None:
                shapes = canvas_result.json_data["objects"]
                if len(shapes) > 0:
                    shape = shapes[0]
                    top_left = (shape["left"], shape["top"])
                    bottom_right = (shape["left"] + shape["width"], shape["top"] + shape["height"])
                    st.session_state.canvas_result = (top_left, bottom_right)
            
            if st.session_state.canvas_result:
                cropped_frame = crop_image(img, st.session_state.canvas_result)
                st.image(cropped_frame, caption="Cropped Image", use_column_width=True)
                st.session_state.cropped_frame = cropped_frame

        # Proceed to classification only if the user clicks the "Classify" button
        if st.button("Classify"):
            if st.session_state.cropped_frame is not None:
                # Classify the image
                classification_result, confidence = classification(class_model, ood_model, st.session_state.cropped_frame, device)
                display_classification_result(classification_result, confidence)
            else:
                st.write("No cropped frame available for classification.")

def process_existing_image(selected_image, image_dir, device, class_model, ood_model, crop_mode):
    image_path = os.path.join(image_dir, selected_image)
    img = Image.open(image_path).convert('RGB')

    # Display the existing image
    st.image(img, caption=f'Image: {selected_image}', use_column_width=True)

    if crop_mode == "Object Detection":
        # Perform polyp detection and cropping
        obj_model = get_obj_detect_model(checkpoint='/home/shpark/Colood/ckpt/yolo_best.pth', device=device, args=None)
        cropped_frame, bbox = detect_and_crop(img, obj_model, device)
        
        # Draw bounding box on the original image
        if bbox.numel() <= 0:  # Check if bbox is empty
            st.write("No object detected.")
        else:
            bbox = tuple(bbox[0][0:4])
            img_with_bbox = draw_bounding_box(img, bbox)
            st.image(img_with_bbox, caption="Existing Image with Bounding Box", use_column_width=True)
            
            # Display cropped frame
            cropped_frame_np = np.array(cropped_frame)
            st.image(cropped_frame_np, caption='Cropped Image', use_column_width=True)

            st.session_state.cropped_frame = cropped_frame

    elif crop_mode == "Click to Crop":
        st.write("Draw a rectangle on the image to crop.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="red",
            background_image=img,
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="rect",
            key="canvas",
        )
        
        if canvas_result.json_data is not None:
            shapes = canvas_result.json_data["objects"]
            if len(shapes) > 0:
                shape = shapes[0]
                top_left = (shape["left"], shape["top"])
                bottom_right = (shape["left"] + shape["width"], shape["top"] + shape["height"])
                st.session_state.canvas_result = (top_left, bottom_right)
        
        if st.session_state.canvas_result:
            cropped_frame = crop_image(img, st.session_state.canvas_result)
            st.image(cropped_frame, caption="Cropped Image", use_column_width=True)
            st.session_state.cropped_frame = cropped_frame

    # Proceed to classification only if the user clicks the "Classify" button
    if st.button("Classify"):
        if st.session_state.cropped_frame is not None:
            # Classify the image
            classification_result, confidence = classification(class_model, ood_model, st.session_state.cropped_frame, device)
            display_classification_result(classification_result, confidence)
        else:
            st.write("No cropped frame available for classification.")

def draw_bounding_box(image, bbox):
    if bbox is not None:
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = bbox
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
    return image

def run():
    st.title("Video and Image Classification App")

    # Directory containing the videos
    video_dir = '/home/shpark/Colood/data/video'
    # Directory containing the existing images
    image_dir = '/home/shpark/Colood/data/img'

    # Sections
    st.sidebar.title("Select Mode")
    mode = st.sidebar.selectbox("Choose mode", ["Video", "Upload Image", "Existing Images"])
    crop_mode = st.sidebar.selectbox("Choose crop mode", ["Object Detection", "Click to Crop"])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    class_model = get_class_model(num_classes=2, device=device, checkpoint='/home/shpark/Colood/ckpt/deit_best.pth')
    ood_model = get_class_model(num_classes=2, device=device, checkpoint='/home/shpark/Colood/ckpt/ood_best.ckpt')

    if mode == "Video":
        # Get list of video files in the directory
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi', '.mpg'))]
        selected_video = st.selectbox("Select a video file", video_files)
        if selected_video:
            process_video(selected_video, video_dir, device, class_model, ood_model, crop_mode)
    elif mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            process_uploaded_image(uploaded_file, device, class_model, ood_model, crop_mode)
    elif mode == "Existing Images":
        # Get list of existing images in the directory
        existing_images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        selected_image = st.selectbox("Select an existing image file", existing_images)
        if selected_image:
            process_existing_image(selected_image, image_dir, device, class_model, ood_model, crop_mode)

if __name__ == "__main__":
    run()
