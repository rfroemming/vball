import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

# Page layout configuration
st.set_page_config(page_title="Volleyball Speed Tracker", page_icon="🏐", layout="centered")

st.markdown("### 🏐 AI-Powered Volleyball Speed Tracker")

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
        st.info(f"Video Resolution: **{w} x {h} pixels**. Click two points on any known object or distance reference, then enter its real-world size below.")

        # Single calibration input for the known distance
        known_meters = st.number_input("Real-World Length / Distance of Reference (meters)", value=6.0, step=0.5, format="%.2f")

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
        
        annotated_frame = first_frame.copy()
        p1 = st.session_state.p1
        p2 = st.session_state.p2

        cv2.line(annotated_frame, p1, p2, (0, 0, 255), 3)
        cv2.circle(annotated_frame, p1, 10, (0, 0, 255), -1)
        cv2.putText(annotated_frame, "P1", (p1[0] - 15, p1[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(annotated_frame, p2, 10, (255, 0, 0), -1)
        cv2.putText(annotated_frame, "P2", (p2[0] - 15, p2[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

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
    st.subheader("⚙️ Step 2: Video Speed Settings")
    
    speed_mode = st.radio(
        "Video Recording Mode",
        ["Standard Speed (30 FPS)", "Slow Motion (0.25x / 120-240 FPS handled via multiplier)"],
        index=0
    )
    
    if "Slow Motion" in speed_mode:
        effective_fps = 120.0  
    else:
        effective_fps = 30.0

    if st.button("🤖 Run AI Detection & Calculate Speed", type="primary"):
        if ref_pixel_length <= 0:
            st.error("Please click two distinct reference points on the image first.")
        else:
            with st.spinner("Processing video frames and calculating trajectory... Please wait."):
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
                conf_threshold = 0.10

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

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

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
                    st.error("Only a few frames were tracked. Ensure the model detects the ball clearly.")
                else:
                    max_pixel_speed = 0.0
                    best_segment = (0, 0)
                    
                    # Lists to calculate horizontal-dominant movement vector averages
                    horizontal_speeds = []

                    for i in range(1, len(centers)):
                        f1, x1, y1 = centers[i-1]
                        f2, x2, y2 = centers[i]
                        frame_diff = f2 - f1
                        if frame_diff > 0:
                            pix_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            pix_speed_per_frame = pix_dist / frame_diff
                            
                            # Check if movement is primarily horizontal (dx > abs(dy) or minor vertical change)
                            dx = abs(x2 - x1)
                            dy = abs(y2 - y1)
                            if dx >= dy:  # Horizontal direction dominant phase
                                segment_mps = (pix_speed_per_frame * effective_fps) * calibrated_meters_per_pixel
                                horizontal_speeds.append(segment_mps)

                            if pix_speed_per_frame > max_pixel_speed:
                                max_pixel_speed = pix_speed_per_frame
                                best_segment = (f1, f2)

                    peak_mps = max_pixel_speed * effective_fps * calibrated_meters_per_pixel
                    max_speed_kmh = peak_mps * 3.6

                    # Compute average horizontal flight speed if available
                    avg_horizontal_kmh = (np.mean(horizontal_speeds) * 3.6) if horizontal_speeds else max_speed_kmh

                    st.markdown("---")
                    st.subheader("📊 Calculation Data Breakdown")
                    
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Calculated Peak Speed", f"{max_speed_kmh:.2f} km/h")
                    

                    with st.expander("🔍 View Raw Tracking & Math Details"):
                        st.write(f"- **Reference Object:** ({known_meters}m)")
                        st.write(f"- **Reference Span Pixel Length:** {ref_pixel_length:.1f} px")
                        st.write(f"- **Calibrated Scale Factor:** {calibrated_meters_per_pixel:.6f} meters/pixel")
                        st.write(f"- **Effective FPS Used:** {effective_fps}")
                        st.write(f"- **Max Frame-to-Frame Displacement:** {max_pixel_speed:.2f} pixels/frame (between frames {best_segment[0]} and {best_segment[1]})")
                        st.write(f"- **Total Frames Processed:** {frame_count}")

else:
    st.info("👆 Upload a video file above to begin calibration and analysis.")
