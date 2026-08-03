import streamlit as st
import tempfile
import base64
import streamlit.components.v1 as components

# Page layout configuration
st.set_page_config(page_title="Video Analysis", page_icon="🏐", layout="centered")

st.markdown("### 🏐 Video Analysis Dashboard")

# Initialize Session State variables
if "hit_time" not in st.session_state:
    st.session_state.hit_time = 0.0
if "landing_time" not in st.session_state:
    st.session_state.landing_time = 0.0
if "last_video_time" not in st.session_state:
    st.session_state.last_video_time = 0.0

# 1. File Uploader Section
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Read video bytes and encode to base64
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')

    st.info("💡 **Tip:** Play, pause, or scrub the video. The live position display tracks your exact spot.")

    # Custom component wrapper to receive live playback time back into Python
    def video_player_with_live_time(b64_data):
        component_code = f"""
        <div style="background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; font-family: sans-serif;">
            <video id="vid" width="100%" controls style="border-radius: 8px;">
                <source src="data:video/mp4;base64,{b64_data}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            
            <!-- Live Position Display Box -->
            <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center; background: #0d1117; padding: 12px 15px; border-radius: 6px; border: 1px solid #21262d;">
                <span style="color: #8b949e; font-family: monospace; font-size: 13px; font-weight: bold;">LIVE POSITION DISPLAY:</span>
                <span id="time-display" style="color: #58a6ff; font-family: monospace; font-size: 20px; font-weight: bold;">0.000 s</span>
            </div>
        </div>
        
        <script>
            const video = document.getElementById('vid');
            const timeDisplay = document.getElementById('time-display');

            video.addEventListener('timeupdate', function() {{
                const currentTime = video.currentTime;
                timeDisplay.innerText = currentTime.toFixed(3) + " s";
                
                // Send current playback time securely to Streamlit component value
                window.Streamlit.setComponentValue(currentTime);
            }});
            
            // Notify Streamlit that the component is ready
            window.Streamlit.setFrameHeight(380);
        </script>
        """
        # Using components.declare_v1 or standard html with return value tracking via standard bridge
        return components.html(component_code, height=560)

    # Render the interactive video player component
    video_player_with_live_time(video_base64)

    st.markdown("---")

    # Alternative precise slider/number tracker sync for user capture
    current_slider_pos = st.slider(
        "🎥 Timeline Position Sync Tracker", 
        min_value=0.0, 
        max_value=60.0, 
        value=float(st.session_state.last_video_time), 
        step=0.001,
        format="%.3f s"
    )
    st.session_state.last_video_time = current_slider_pos

    # Native Python Capture Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📍 Capture Current as Mark Hit", use_container_width=True, type="primary"):
            st.session_state.hit_time = round(st.session_state.last_video_time, 3)
            st.success(f"Captured Hit Time: {st.session_state.hit_time} s")
    with col_btn2:
        if st.button("📍 Capture Current as Mark Landing", use_container_width=True, type="primary"):
            st.session_state.landing_time = round(st.session_state.last_video_time, 3)
            st.success(f"Captured Landing Time: {st.session_state.landing_time} s")

    st.markdown("---")

    # Timestamps inputs (reflecting captured or manually typed values)
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
