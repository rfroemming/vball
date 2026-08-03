import streamlit as st
import tempfile
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Page layout configuration
st.set_page_config(page_title="Video Analysis - AI Detection", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Video Analysis Dashboard")

# Load a pre-trained YOLO model (using YOLOv8 nano for fast execution)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# File Uploader Section
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display video player
    st.video(video_path)
    st.markdown("---")

    # Reliable Calibration Inputs
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        # Instead of ball size, use a fixed distance scale reference (e.g., total estimated travel distance or calibration span)
        known_distance_meters = st.number_input("Total Estimated Ball Flight Distance (m)", value=5.0, step=0.5, help="Approximate distance the ball travels from hit to landing.")
    with col_cfg2:
        hit_type = st.selectbox("Hit type", ["Serve", "Spike", "Pass", "Setter Dump"])

    # Slow-motion multiplier configuration
    slow_mo_options = {
        "Normal (1x)": 1.0,
        "Slow-mo (1/2x)": 0.5,
        "Slow-mo (1/4x)": 0.25,
        "Slow-mo (1/8x)": 0.125
    }
    selected_speed_label = st.selectbox("Video recording slow-motion factor", list(slow_mo_options.keys()))
    speed_factor = slow_mo_options[selected_speed_label]

    if st.button("🤖 Run AI Ball Detection & Speed Estimation", type="primary"):
        with st.spinner("Processing video frames with YOLO AI and stabilizing trajectory... Please wait."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30.0  # Fallback default
            
            centers = []
            frame_count = 0
            SPORTS_BALL_CLASS_ID = 32 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                results = model(frame, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == SPORTS_BALL_CLASS_ID and float(box.conf[0]) > 0.3:
                            xyxy = box.xyxy[0].cpu().numpy()
                            cx = int((xyxy[0] + xyxy[2]) / 2)
                            cy = int((xyxy[1] + xyxy[3]) / 2)
                            centers.append((frame_count, cx, cy))
                            break

            cap.release()

            if len(centers) < 2:
                st.error("Could not track the ball across enough frames. Try adjusting confidence or ensuring the ball is clearly visible.")
            else:
                # Find the maximum displacement vector (peak velocity segment between consecutive tracked frames)
                max_pixel_speed = 0.0
                for i in range(1, len(centers)):
                    f1, x1, y1 = centers[i-1]
                    f2, x2, y2 = centers[i]
                    
                    frame_diff = f2 - f1
                    if frame_diff > 0:
                        pix_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        pix_speed_per_frame = pix_dist / frame_diff
                        if pix_speed_per_frame > max_pixel_speed:
                            max_pixel_speed = pix_speed_per_frame

                # Approximate total pixel span of the trajectory for scaling
                total_pixel_span = np.sum([np.sqrt((centers[i][1]-centers[i-1][1])**2 + (centers[i][2]-centers[i-1][2])**2) for i in range(1, len(centers))])
                
                if total_pixel_span > 0:
                    meters_per_pixel = known_distance_meters / total_pixel_span
                    peak_mps = (max_pixel_speed * fps * meters_per_pixel) * speed_factor
                    max_speed_kmh = peak_mps * 3.6
                else:
                    max_speed_kmh = 0.0

                st.success("AI Analysis Complete!")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Estimated Peak Speed", f"{max_speed_kmh:.2f} km/h")
                res_col2.metric("Frames Tracked", len(centers))

else:
    st.info("👆 Upload a video file above to start AI tracking.")
