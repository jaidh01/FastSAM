import streamlit as st
import torch
import numpy as np
import cv2
import time
from fastsam import FastSAM, FastSAMPrompt

# Set up the Streamlit layout
st.title("Real-time Object Detection with FastSAM")

# Load the model once
model = FastSAM('FastSAM-x.pt')

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

st.write(f"Using device: {DEVICE}")

#st.sidebar.title("Options")
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)
iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.9)
prompt_option = st.selectbox(
    "Select Prompt Type", 
    ("Point Prompt", "Text Prompt", "Box Prompt")
)

# Start and Stop buttons
start_button = st.button("Start", key="start")
stop_button = st.button("Stop", key="stop")

def main(cap):
  cap = cv2.VideoCapture(0)  # Start the video capture

    # Main loop for video stream
  if cap.isOpened():
        stframe = st.empty()  # Placeholder for the video frame
        
        while True:
            suc, frame = cap.read()

            if not suc:
                st.write("Failed to capture video")
                break

            start = time.perf_counter()

            # Run inference with FastSAM
            everything_results = model(
                source=frame,
                device=DEVICE,
                retina_masks=True,
                imgsz=1024,
                conf=confidence_threshold,
                iou=iou_threshold
            )
            
            # Process the prompt based on user selection
            prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)

            if prompt_option == "Point Prompt":
                ann = prompt_process.point_prompt(points=[[620,360]], pointlabel=[1])
            elif prompt_option == "Text Prompt":
                ann = prompt_process.text_prompt(text='a man')
            elif prompt_option == "Box Prompt":
                ann = prompt_process.box_prompt(bboxes=[[200,200,300,300], [500,500,600,600]])

            end = time.perf_counter()
            total_time = end - start
            fps = 1 / total_time

            # Generate the result image with annotations
            img = prompt_process.plot_to_result(img=frame, annotations=ann)

            return img

            # Display FPS on Streamlit
            st.sidebar.write(f"FPS: {fps:.2f}")

            # Display the frame and the annotated result in the Streamlit app
            stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

#             # Streamlit break condition (button press to stop)
            if stop_button:  # Only exit when Stop is pressed
                break

#     # Release resources after exiting the loop
   cap.release()
   cv2.destroyAllWindows()


