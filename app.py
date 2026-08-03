import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# Page layout configuration
st.set_page_config(page_title="Volleyball Speed Tracker - Court Calibration", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Volleyball Speed Tracker with Court Calibration")

# Load your custom-trained YOLO model weights
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in the same folder as this app
    return YOLO("best.pt")

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
    st.subheader("📏 Calibration Settings")
    col_cal1, col_cal2 = st.columns(2)
    with col_cal1:
        reference_type = st.selectbox(
            "Calibration Reference Line",
            [
                "Half-Court Length (Center Line to Baseline) - 9.0m",
                "Attack Line to Baseline - 6.0m",
                "Full Court Width - 9.0m",
                "Attack Line to Center Line - 3.0m",
                "Custom Distance"
            ]
        )
    with col_cal2:
        if "9.0m" in reference_type:
            default_ref_meters = 9.0
        elif "6.0m" in reference_type:
            default_ref_meters = 6.0
        elif "3.0m" in reference_type:
            default_ref_meters = 3.0
        else:
            default_ref_meters = 5.0
            
        known_ref_meters = st.number_input("Reference Line Real-World Length (m)", value=default_ref_meters, step=0.5)

    st.markdown("---")
    st.subheader("⚙️ Detection & Timing Settings")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        true_fps = st.number_input("Recording FPS", value=240.0, step=10.0, help="Set to 240 if recorded in 240fps slow-motion.")
    with col_cfg2:
        conf_threshold = st.slider("Custom Model Confidence", min_value=0.01, max_value=0.50, value=0.10, step=0.05)
    with col_cfg3:
        hit_type = st.selectbox("Hit type", ["Serve", "Spike", "Pass", "Setter Dump"])

    slow_mo_options = {
        "Standard (1x - Use with True FPS)": 1.0,
        "Slow-mo export factor (1/2x)": 0.5,
        "Slow-mo export factor (1/4x)": 0.25,
        "Slow-mo export factor (1/8x)": 0.125
    }
    selected_speed_label = st.selectbox("Timeline Playback Speed Multiplier", list(slow_mo_options.keys()))
    speed_factor = slow_mo_options[selected_speed_label]

    if st.button("🤖 Run Calibration & Speed Analysis", type="primary"):
        with st.spinner("Processing video frames and running AI detection... Please wait."):
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
            CUSTOM_CLASS_ID = 0 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                results = model(frame, verbose=False)
                
                best_box = None
                highest_conf = 0.0

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls == CUSTOM_CLASS_ID and conf >= conf_threshold:
                            if conf > highest_conf:
                                highest_conf = conf
                                best_box = box.xyxy[0].cpu().numpy()

                if best_box is not None:
                    x1, y1, x2, y2 = map(int, best_box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    centers.append((frame_count, cx, cy))

                    # Draw bounding box and center dot
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Ball {highest_conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(frame)

            cap.release()
            out.release()

            st.success("Analysis Complete!")

            # Display Annotated Video Preview
            st.subheader("🎥 Annotated AI Detection Preview")
            with open(output_preview_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes, format="video/webm")

            if len(centers) < 5:
                st.error(f"Only {len(centers)} frames tracked. Try lowering the Confidence slider.")
            else:
                # Calculate max frame-to-frame displacement speed
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

                # Alternative scaling option: if you want to calibrate using the trajectory span itself
                start_pt = np.array([centers[0][1], centers[0][2]])
                end_pt = np.array([centers[-1][1], centers[-1][2]])
                straight_line_pixel_span = np.linalg.norm(end_pt - start_pt)
                
                if straight_line_pixel_span > 0:
                    # Use the reference line calibration scale
                    meters_per_pixel = known_ref_meters / straight_line_pixel_span
                    
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
                    st.write(f"- **Calibration Reference Length:** {known_ref_meters} meters")
                    st.write(f"- **Confidence Threshold Used:** {conf_threshold}")
                    st.write(f"- **Forced True Recording FPS:** {true_fps}")
                    st.write(f"- **Max Frame-to-Frame Displacement:** {max_pixel_speed:.2f} pixels/frame (between frames {best_segment[0]} and {best_segment[1]})")
                    st.write(f"- **Calculated Scale Factor:** {meters_per_pixel:.6f} meters/pixel")
                    st.write(f"- **Total Frames Processed:** {frame_count}")

else:
    st.info("👆 Upload a video file above to begin.")
