import streamlit as st
import tempfile
import cv2
import numpy as np
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

    # Configuration options using 20 cm ball size reference
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        known_ball_diameter_cm = st.number_input("Known Ball Diameter (cm)", value=20.0, step=0.5)
    with col_cfg2:
        hit_type = st.selectbox("Hit type", ["Serve", "Spike", "Pass", "Setter Dump"])

    if st.button("🤖 Run AI Ball Detection & Speed Estimation", type="primary"):
        with st.spinner("Processing video frames with YOLO AI and 20 cm ball calibration... Please wait."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30.0  # Fallback default
            
            prev_center = None
            max_speed_kmh = 0.0
            frame_count = 0
            
            # COCO dataset class 32 is 'sports ball' in standard YOLO models
            SPORTS_BALL_CLASS_ID = 32 
            known_diameter_m = known_ball_diameter_cm / 100.0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run YOLO object detection on the frame
                results = model(frame, verbose=False)
                
                current_center = None
                current_ball_pixels = 0
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        # Check if detected object is a sports ball and has sufficient confidence
                        if cls == SPORTS_BALL_CLASS_ID and float(box.conf[0]) > 0.25:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            
                            # Calculate center (x, y) of the ball
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            current_center = (cx, cy)
                            
                            # Use bounding box width/height to estimate ball pixel size diameter
                            box_width = x2 - x1
                            box_height = y2 - y1
                            current_ball_pixels = max(box_width, box_height)
                            break
                
                # Calculate speed dynamically using the 20 cm ball reference scale on consecutive frames
                if prev_center is not None and current_center is not None and current_ball_pixels > 0:
                    pixel_distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                    
                    # Dynamic scale factor calculation: meters per pixel based on 20 cm reference width
                    meters_per_pixel = known_diameter_m / current_ball_pixels
                    
                    distance_meters = pixel_distance * meters_per_pixel
                    speed_mps = distance_meters * fps
                    speed_kmh = speed_mps * 3.6
                    
                    # Filter out unrealistic outliers caused by detection jitter
                    if speed_kmh > max_speed_kmh and speed_kmh < 180.0:
                        max_speed_kmh = speed_kmh

                if current_center is not None:
                    prev_center = current_center

            cap.release()

            st.success("AI Analysis Complete!")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Peak AI-Detected Speed", f"{max_speed_kmh:.2f} km/h")
            res_col2.metric("Total Frames Analyzed", frame_count)

else:
    st.info("👆 Upload a video file above to start AI tracking.")
