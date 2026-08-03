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
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
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

    if st.button("🤖 Run AI Detection & Show Preview", type="primary"):
        with st.spinner("Processing frames, running YOLO detection, and generating preview... Please wait."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps == 0:
                fps = 30.0

            # Setup temporary output video for annotated preview
            output_preview_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_preview_path, fourcc, fps, (width, height))

            centers = []
            frame_count = 0
            SPORTS_BALL_CLASS_ID = 32 
            preview_frames_captured = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                results = model(frame, verbose=False)
                
                detected_in_frame = False
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls == SPORTS_BALL_CLASS_ID and conf > 0.25:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            
                            centers.append((frame_count, cx, cy))
                            detected_in_frame = True

                            # Draw visual bounding box and center dot on frame for preview
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(frame, f"Ball Conf: {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Write annotated frame to preview video
                out.write(frame)

                # Keep a few sample frames to display static previews if needed
                if detected_in_frame and len(preview_frames_captured) < 3:
                    preview_frames_captured.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
            out.release()

            st.success("AI Preview Generation Complete!")

            # Display Annotated Video Preview
            st.subheader("🎥 Annotated YOLO Detection Preview")
            st.video(output_preview_path)

            if len(centers) < 2:
                st.error("Could not track the ball across enough consecutive frames. Try lowering confidence thresholds or using a clearer angle.")
            else:
                # Calculations
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
                    peak_mps = (max_pixel_speed * fps * meters_per_pixel) * speed_factor
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
                    st.write(f"- **Video Base FPS:** {fps}")
                    st.write(f"- **Slow-mo Multiplier Applied:** {speed_factor}x")
                    st.write(f"- **Total Pixel Span of Trajectory:** {total_pixel_span:.2f} px")
                    st.write(f"- **Max Frame-to-Frame Displacement:** {max_pixel_speed:.2f} pixels/frame (between frames {best_segment[0]} and {best_segment[1]})")
                    st.write(f"- **Estimated Scale Factor:** {meters_per_pixel:.6f} meters/pixel")
                    st.write(f"- **Total Frames Processed:** {frame_count}")

else:
    st.info("👆 Upload a video file above to generate the AI detection preview.")
