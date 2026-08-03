import streamlit as st
import tempfile
import cv2

# Page layout configuration
st.set_page_config(page_title="Video Analysis", page_icon="🏐", layout="centered")

st.markdown("### 🏐 Video Analysis Dashboard")

# Initialize Session State variables for timestamps
if "hit_time" not in st.session_state:
    st.session_state.hit_time = 0.0
if "landing_time" not in st.session_state:
    st.session_state.landing_time = 0.0
if "current_pos" not in st.session_state:
    st.session_state.current_pos = 0.0

# 1. File Uploader Section
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Open video to get properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    st.info("💡 **Tip:** Use the slider below to scrub through the video. The live position display will update instantly.")

    # Live Position Slider acting as the video timeline controller
    st.session_state.current_pos = st.slider(
        "🎥 Live Video Timeline Position", 
        min_value=0.0, 
        max_value=float(duration), 
        value=float(st.session_state.current_pos), 
        step=0.001,
        format="%.3f s"
    )

    # Live Position Display Box
    st.markdown(
        f"""
        <video id="vid" width="100%" controls style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div style="background-color: #161b22; padding: 12px 15px; border-radius: 6px; border: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="color: #8b949e; font-family: monospace; font-size: 13px; font-weight: bold;">LIVE POSITION DISPLAY:</span>
            <span style="color: #58a6ff; font-family: monospace; font-size: 18px; font-weight: bold;">{st.session_state.current_pos:.3f} s</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Native Python Capture Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📍 Capture as Mark Hit", use_container_width=True, type="primary"):
            st.session_state.hit_time = round(st.session_state.current_pos, 3)
            st.success(f"Captured Hit Time: {st.session_state.hit_time} s")
    with col_btn2:
        if st.button("📍 Capture as Mark Landing", use_container_width=True, type="primary"):
            st.session_state.landing_time = round(st.session_state.current_pos, 3)
            st.success(f"Captured Landing Time: {st.session_state.landing_time} s")

    st.markdown("---")

    # Timestamps inputs (reflecting captured or typed values)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.hit_time = st.number_input(
            "📍 Mark Hit (seconds)", 
            min_value=0.0, 
            value=float(st.session_state.hit_time), 
            step=0.001,
            format="%.3f"
        )
    with col2:
        st.session_state.landing_time = st.number_input(
            "📍 Mark Landing (seconds)", 
            min_value=0.0, 
            value=float(st.session_state.landing_time), 
            step=0.001,
            format="%.3f"
        )

    st.markdown("---")

    # Parameters Form
    col3, col4 = st.columns(2)
    with col3:
        distance = st.number_input("Distance (m)", min_value=0.1, value=15.0, step=0.5)
    with col4:
        hit_type = st.selectbox("Hit type", ["Serve", "Spike", "Pass", "Setter Dump"])

    # Video Playback Speed Selector (handling slow-motion footage)
    slow_mo_options = {
        "Normal (1x)": 1.0,
        "Slow-mo (1/2x)": 0.5,
        "Slow-mo (1/4x)": 0.25,
        "Slow-mo (1/8x)": 0.125
    }
    selected_speed_label = st.selectbox("Video playback speed", list(slow_mo_options.keys()))
    speed_factor = slow_mo_options[selected_speed_label]
    
    st.caption("Real time = video timestamp elapsed $\times$ this factor")

    player_name = st.text_input("Player (optional)", placeholder="Player name")

    st.markdown("---")

    # Calculation logic
    if st.button("🚀 Calculate Speed", type="primary"):
        raw_time_elapsed = st.session_state.landing_time - st.session_state.hit_time
        real_time_elapsed = raw_time_elapsed * speed_factor
        
        if real_time_elapsed > 0:
            speed_mps = distance / real_time_elapsed
            speed_kmh = speed_mps * 3.6
            speed_mph = speed_mps * 2.23694
            
            st.success("Analysis Complete!")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Calculated Speed (KM/H)", f"{speed_kmh:.2f} km/h")
            res_col2.metric("Calculated Speed (MPH)", f"{speed_mph:.2f} mph")
            
            if player_name:
                st.info(f"Recorded for player: **{player_name}** ({hit_type})")
        else:
            st.error("Error: 'Mark Landing' timestamp must occur *after* 'Mark Hit' timestamp.")

else:
    st.info("👆 Upload a video file above to start analyzing your clips.")
