import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# Page layout configuration
st.set_page_config(page_title="Volleyball Speed Tracker - Manual Court Calibration", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Volleyball Speed Tracker with Court Calibration")

# Load your custom-trained YOLO model weights
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# File Uploader Section
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Extract the first frame to display for calibration reference
    cap_temp = cv2.VideoCapture(video_path)
    ret, first_frame = cap_temp.read()
    cap_temp.release()

    if ret:
        # Convert BGR to RGB for Streamlit display
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = first_frame.shape

        st.subheader("📐 Step 1: Court Line Calibration")
        st.info(f"Video Resolution: **{w} x {pixel_height if 'pixel_height' in locals() else h} pixels**. Use the reference line below to calibrate scale.")
        
        # Display the first frame so the user can see coordinate locations
        st.image(first_frame_rgb, caption="First Frame Reference (Use pixel coordinates or click points)", use_column_width=True)

        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            ref_line_type = st.selectbox(
                "Select Reference Line to Measure",
                [
                    "9-Meter Half-Court Line (Center to Baseline)",
                    "6-Meter Attack Line to Baseline",
                    "3-Meter Attack Line to Center Line",
                    "Custom Reference Line"
                ]
            )
        with col_cal2:
            if "9-Meter" in ref_line_type:
                default_meters = 9.0
            elif "6-Meter" in ref_line_type:
                default_meters = 6.0
            elif "3-Meter" in ref_line_type:
                default_meters = 3.0
            else:
                default_meters = 5.0
            known_meters = st.number_input("Real-World Length of this Line (meters)", value=default_meters, step=0.5)

        st.markdown("Enter the pixel coordinates `(x, y)` of the **two endpoints** of your chosen reference line (visible on the image above):")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Point 1 (Start of Line)**")
            x1_ref = st.number_input("X1 Pixel", value=int(w * 0.3), step=1)
            y1_ref = st.number_input("Y1 Pixel", value=int(h * 0.7), step=1)
        with col_p2:
            st.markdown("**Point 2 (End of Line)**")
            x2_ref = st.number_input("X2 Pixel", value=int(w * 0.7), step=1)
            y2_ref = st.number_input("Y2 Pixel", value=int(h * 0.7), step=1)

        # Calculate pixel length of the reference line
        ref_pixel_length = np.sqrt((x2_ref - x1_ref)**2 + (y2_ref - y1_ref)**2)
        if ref_pixel_length > 0:
            calibrated_meters_per_pixel = known_meters / ref_pixel_length
            st.success(f"Calibration successful! Scale factor locked at: **{calibrated_meters_per_pixel:.6f} meters/pixel** (Line length: {ref_pixel_length:.1f} pixels)")
        else:
            st.warning("Reference line pixel length is 0. Please adjust coordinates.")

    st.markdown("---")
    st.subheader("⚙️ Step 2: Detection & Timing Settings")
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

    if st.button("🤖 Run AI Detection & Calculate Speed", type="primary"):
        if ref_pixel_length <= 0:
            st.error("Please provide a valid reference line calibration before running analysis.")
        else:
            with st.spinner("Processing video frames with custom model and calibrated scale... Please wait."):
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
                    # Calculate max frame-to-frame displacement speed using calibrated scale
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

                    # Peak speed calculation using the manual calibrated meters_per_pixel scale factor
                    peak_mps = (max_pixel_speed * true_fps * calibrated_meters_per_pixel) * speed_factor
                    max_speed_kmh = peak_mps * 3.6

                    # Display Metrics and Raw Data Breakdown
                    st.markdown("---")
                    st.subheader("📊 Calculation Data Breakdown")
                    
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Calculated Peak Speed", f"{max_speed_kmh:.2f} km/h")
                    res_col2.metric("Total Frames Tracked", len(centers))

                    with st.expander("🔍 View Raw Tracking & Math Details"):
                        st.write(f"- **Calibration Reference Line:** {ref_line_type} ({known_meters}m)")
                        st.write(f"- **Reference Line Pixel Length:** {ref_pixel_length:.1f} px")
                        st.write(f"- **Calibrated Scale Factor:** {calibrated_meters_per_pixel:.6f} meters/pixel")
                        st.write(f"- **Forced True Recording FPS:** {true_fps}")
                        st.write(f"- **Max Frame-to-Frame Displacement:** {max_pixel_speed:.2f} pixels/frame (between frames {best_segment[0]} and {best_segment[1]})")
                        st.write(f"- **Total Frames Processed:** {frame_count}")

else:
    st.info("👆 Upload a video file above to begin calibration and analysis.")
