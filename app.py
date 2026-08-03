import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# Page layout configuration
st.set_page_config(page_title="Video Analysis - AI Preview", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Video Analysis & Detection Preview")

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

    # Display original video player
    st.video(video_path)
    st.markdown("---")

    # Configuration Inputs
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        known_distance_meters = st.number_input("Flight Distance (m)", value=5.0, step=0.5, help="Approximate distance the ball travels from hit to landing.")
    with col_cfg2:
        true_fps = st.number_input("Recording FPS", value=240.0, step=10.0, help="Set to 240 if recorded in 240fps slow-motion.")
    with col_cfg3:
        # Adjustable confidence slider to catch blurred/fast-moving balls
        conf_threshold = st.slider("YOLO Confidence", min_value=0.05, max_value=0.50, value=0.15, step=0.05, help="Lower values help detect fast, blurred balls.")
    with col_cfg4:
        hit_type = st.selectbox("Hit type", ["Serve", "Spike", "Pass", "Setter Dump"])

    slow_mo_options = {
        "Standard (1x - Use with True FPS)": 1.0,
        "Slow-mo export factor (1/2x)": 0.5,
        "Slow-mo export factor (1/4x)": 0.25,
        "Slow-mo export factor (1/8x)": 0.125
    }
    selected_speed_label = st.selectbox("Timeline Playback Speed Multiplier", list(slow_mo_options.keys()))
    speed_factor = slow_mo_options[selected_speed_label]

    if st.button("🤖 Run AI Detection & Show Preview", type="primary"):
        with st.spinner("Processing frames with lowered confidence filter... Please wait."):
            cap = cv2.VideoCapture(video_path)
            detected_container_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if detected_container_fps == 0:
                detected_container_fps = 30.0

            # Use WebM format for smooth browser playback
            output_preview_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(output_preview_path, fourcc, detected_container_fps, (width, height))

            centers = []
            frame_count = 0
            SPORTS_BALL_CLASS_ID = 32 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                # Run YOLO on the frame
                results = model(frame, verbose=False)
                
                best_box = None
                highest_conf = 0.0

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Filter for sports ball using the adjustable confidence threshold
                        if cls == SPORTS_BALL_CLASS_ID and conf >= conf_threshold:
                            if conf > highest_conf:
                                highest_conf = conf
                                best_box = box.xyxy[0].cpu().numpy()

                # If a ball was found in this frame, record and draw it
                if best_box is not None:
                    x1, y1, x2, y2 = map(int, best_box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    centers.append((frame_count, cx, cy))

                    # Draw visual bounding box and center dot
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Ball {highest_conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(frame)

            cap.release()
            out.release()

            st.success("AI Preview Generation Complete!")

            # Display Annotated Video Preview
            st.subheader("🎥 Annotated YOLO Detection Preview")
            with open(output_preview_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes, format="video/webm")

            if len(centers) < 5:
                st.error(f"Only {len(centers)} frames tracked. Try lowering the YOLO Confidence slider further (e.g., to 0.05) or check if the ball is clear in the video.")
            else:
                # Calculations using true_fps
                max_pixel_speed = 0.0
                best_segment = (0, 0)
                for i in range(1, len(centers)):
                    f1, x1, y1 = centers[i-1]
                    f2, x2, y2 = centers[i]
                    frame_diff = f2 - f1
                    if frame_diff > 0:
                        pix_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        pix_speed_per_frame = pix_dist / frame_diff
                        if pix_speed_per_frame > max_pixel_speed:
                            max_pixel_speed = pix_speed_per_frame
                            best_segment = (f1, f2)

                total_pixel_span = np.sum([np.sqrt((centers[i][1]-centers[i-1][1])**2 + (centers[i][2]-centers[i-1][2])**2) for i in range(1, len(centers))])
                
                if total_pixel_span > 0:
                    meters_per_pixel = known_distance_meters / total_pixel_span
                    peak_mps = (max_pixel_speed * true_fps * meters_per_pixel) * speed_factor
                    max_speed_kmh = peak_mps * 3.6
                else:
                    max_speed_kmh = 0.0

                # Display Metrics and Raw Data Breakdown
                st.markdown("---")
                st.subheader("📊 Calculation Data Breakdown")
                
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Calculated Peak Speed", f"{max_speed_kmh:.2f} km/h")
                res_col2.metric("Total Frames Tracked", len(centers))

                with st.expander("🔍 View Raw Tracking & Math Details"):
                    st.write(f"- **YOLO Confidence Threshold Used:** {conf_threshold}")
                    st.write(f"- **Forced True Recording FPS:** {true_fps}")
                    st.write(f"- **Total Pixel Span of Trajectory:** {total_pixel_span:.2f} px")
                    st.write(f"- **Max Frame-to-Frame Displacement:** {max_pixel_speed:.2f} pixels/frame (between frames {best_segment[0]} and {best_segment[1]})")
                    st.write(f"- **Estimated Scale Factor:** {meters_per_pixel:.6f} meters/pixel")
                    st.write(f"- **Total Frames Processed:** {frame_count}")

else:
    st.info("👆 Upload a video file above to generate the AI detection preview.")
