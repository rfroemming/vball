import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

# Page layout configuration
st.set_page_config(page_title="Volleyball Speed Tracker - Custom Calibration", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Volleyball Speed Tracker with Custom Calibration")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap_temp = cv2.VideoCapture(video_path)
    ret, first_frame = cap_temp.read()
    cap_temp.release()

    if ret:
        h, w, _ = first_frame.shape

        st.subheader("📐 Step 1: Custom Object / Distance Calibration")
        st.info(f"Video Resolution: **{w} x {h} pixels**. Click two points on any known object or distance reference (e.g., volleyball diameter = 0.2m, or a court line), then enter its real-world size below.")

        # Flexible calibration inputs
        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            reference_label = st.text_input("Description of Reference Object", value="Volleyball Diameter (0.2m)")
        with col_cal2:
            known_meters = st.number_input("Real-World Length / Distance (meters)", value=0.2, step=0.05, format="%.3f")

        # Initialize session state safely
        if "p1" not in st.session_state:
            st.session_state.p1 = (int(w * 0.4), int(h * 0.5))
        if "p2" not in st.session_state:
            st.session_state.p2 = (int(w * 0.45), int(h * 0.5))
        if "active_point" not in st.session_state:
            st.session_state.active_point = "P1"

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🔴 Set Next Click to: Point 1 (Start)", use_container_width=True):
                st.session_state.active_point = "P1"
        with col_act2:
            if st.button("🔵 Set Next Click to: Point 2 (End)", use_container_width=True):
                st.session_state.active_point = "P2"

        st.markdown(f"👉 **Currently targeting:** **{st.session_state.active_point}**. Click on the image below to update it.")
        
        # Draw calibration line and points onto a copy of the frame for visual feedback
        annotated_frame = first_frame.copy()
        p1 = st.session_state.p1
        p2 = st.session_state.p2

        # Draw line connecting P1 and P2
        cv2.line(annotated_frame, p1, p2, (0, 0, 255), 3)
        # Draw Point 1 (Red circle)
        cv2.circle(annotated_frame, p1, 10, (0, 0, 255), -1)
        cv2.putText(annotated_frame, "P1", (p1[0] - 15, p1[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Draw Point 2 (Blue circle)
        cv2.circle(annotated_frame, p2, 10, (255, 0, 0), -1)
        cv2.putText(annotated_frame, "P2", (p2[0] - 15, p2[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Interactive Image Display
        coords = streamlit_image_coordinates(annotated_frame_rgb, width=700, key="calib_image")

        if coords is not None:
            scale_percent = w / 700.0
            orig_x = int(coords["x"] * scale_percent)
            orig_y = int(coords["y"] * scale_percent)

            if st.session_state.active_point == "P1":
                if st.session_state.p1 != (orig_x, orig_y):
                    st.session_state.p1 = (orig_x, orig_y)
                    st.rerun()
            else:
                if st.session_state.p2 != (orig_x, orig_y):
                    st.session_state.p2 = (orig_x, orig_y)
                    st.rerun()

        x1_ref, y1_ref = st.session_state.p1
        x2_ref, y2_ref = st.session_state.p2

        ref_pixel_length = np.sqrt((x2_ref - x1_ref)**2 + (y2_ref - y1_ref)**2)
        if ref_pixel_length > 0:
            calibrated_meters_per_pixel = known_meters / ref_pixel_length
            st.success(f"Calibration successful! Scale factor locked at: **{calibrated_meters_per_pixel:.6f} meters/pixel** (Span length: {ref_pixel_length:.1f} pixels)")
        else:
            calibrated_meters_per_pixel = 0.0
            st.warning("Reference span pixel length is 0. Please select two distinct points.")

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
            st.error("Please click two distinct reference points on the image first.")
        else:
            with st.spinner("Processing video frames with custom model, trajectory rendering, and calibrated scale... Please wait."):
                cap = cv2.VideoCapture(video_path)
                detected_container_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if detected_container_fps == 0:
                    detected_container_fps = 30.0

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

                        # Draw current bounding box and center dot
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # Draw full trajectory trail connecting all tracked frame centers
                    if len(centers) > 1:
                        for i in range(1, len(centers)):
                            pt1 = (centers[i-1][1], centers[i-1][2])
                            pt2 = (centers[i][1], centers[i][2])
                            cv2.line(frame, pt1, pt2, (255, 255, 0), 3)

                    out.write(frame)

                cap.release()
                out.release()

                st.success("Analysis Complete!")

                st.subheader("🎥 Annotated AI Detection & Trajectory Preview")
                with open(output_preview_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, format="video/webm")

                if len(centers) < 5:
                    st.error(f"Only {len(centers)} frames tracked. Try lowering the Confidence slider.")
                else:
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

                    peak_mps = (max_pixel_speed * true_fps * calibrated_meters_per_pixel) * speed_factor
                    max_speed_kmh = peak_mps * 3.6

                    st.markdown("---")
                    st.subheader("📊 Calculation Data Breakdown")
                    
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Calculated Peak Speed", f"{max_speed_kmh:.2f} km/h")
                    res_col2.metric("Total Frames Tracked", len(centers))

                    with st.expander("🔍 View Raw Tracking & Math Details"):
                        st.write(f"- **Reference Object:** {reference_label} ({known_meters}m)")
                        st.write(f"- **Reference Span Pixel Length:** {ref_pixel_length:.1f} px")
                        st.write(f"- **Calibrated Scale Factor:** {calibrated_meters_per_pixel:.6f} meters/pixel")
                        st.write(f"- **Forced True Recording FPS:** {true_fps}")
                        st.write(f"- **Max Frame-to-Frame Displacement:** {max_pixel_speed:.2f} pixels/frame (between frames {best_segment[0]} and {best_segment[1]})")
                        st.write(f"- **Total Frames Processed:** {frame_count}")

else:
    st.info("👆 Upload a video file above to begin calibration and analysis.")
