import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# Page layout configuration
st.set_page_config(page_title="Video Analysis - AI Detection", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Video Analysis Dashboard")

# Load a pre-trained YOLO model (using YOLOv8 nano for fast browser execution)
@st.cache_resource
def load_model():
    # 'yolov8n.pt' will download automatically on first run. 
    # For sports balls specifically, you can later train or use a custom-trained weights file.
    return YOLO("yolov8n.pt")

model = load_model()

# File Uploader Section
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "ai"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display video player
    st.video(video_path)
    st.markdown("---")

    distance = st.number_input("Distance (m)", min_value=0.1, value=15.0, step=0.5)
    
    if st.button("🤖 Run AI Ball Detection & Speed Estimation", type="primary"):
        with st.spinner("Processing video frames with YOLO AI... Please wait."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            prev_center = None
            max_speed_kmh = 0
            frame_count = 0
            
            # COCO dataset class 32 is 'sports ball' in standard YOLO models
            SPORTS_BALL_CLASS_ID = 32 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run YOLO object detection on the frame
                results = model(frame, verbose=False)
                
                current_center = None
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        # Check if detected object is a sports ball and has high confidence
                        if cls == SPORTS_BALL_CLASS_ID and float(box.conf[0]) > 0.3:
                            # Get bounding box coordinates [x1, y1, x2, y2]
                            xyxy = box.xyxy[0].cpu().numpy()
                            # Calculate center (x, y) of the ball
                            cx = int((xyxy[0] + xyxy[2]) / 2)
                            cy = int((xyxy[1] + xyxy[3]) / 2)
                            current_center = (cx, cy)
                            break
                
                # Calculate speed if we tracked the ball across consecutive frames
                if prev_center is not None and current_center is not None:
                    pixel_distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                    
                    # Assuming a generic pixel-to-meter scale factor for demonstration 
                    # (ideally calibrated against a known court dimension)
                    meters_per_pixel = 0.02 
                    distance_meters = pixel_distance * meters_per_pixel
                    
                    speed_mps = distance_meters * fps
                    speed_kmh = speed_mps * 3.6
                    
                    if speed_kmh > max_speed_kmh:
                        max_speed_kmh = speed_kmh

                prev_center = current_center

            cap.release()

            st.success("AI Analysis Complete!")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Peak AI-Detected Speed", f"{max_speed_kmh:.2f} km/h")
            res_col2.metric("Total Frames Analyzed", frame_count)

else:
    st.info("👆 Upload a video file above to start AI tracking.")
