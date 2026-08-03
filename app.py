import streamlit as st
import tempfile
import base64
import streamlit.components.v1 as components

# Page layout configuration
st.set_page_config(page_title="Video Analysis", page_icon="🏐", layout="centered")

st.markdown("### 🏐 Video Analysis Dashboard")

# Initialize Session State variables for timestamps
if "hit_time" not in st.session_state:
    st.session_state.hit_time = 0.0
if "landing_time" not in st.session_state:
    st.session_state.landing_time = 0.0

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

    st.info("💡 **Tip:** Play, pause, or scrub the video. Use the capture buttons inside the player card to grab timestamps instantly.")

    # Combined Video Player + Live Display + Capture Buttons Component with Streamlit JS Bridge
    player_component_html = f"""
    <div style="background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; font-family: sans-serif;">
        <video id="vid" width="100%" controls style="border-radius: 8px;">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        
        <!-- Live Position Display Box -->
        <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center; background: #0d1117; padding: 12px 15px; border-radius: 6px; border: 1px solid #21262d;">
            <span style="color: #8b949e; font-family: monospace; font-size: 13px; font-weight: bold;">LIVE POSITION DISPLAY:</span>
            <span id="time-display" style="color: #58a6ff; font-family: monospace; font-size: 20px; font-weight: bold;">0.000 s</span>
        </div>

        <!-- Embedded Interactive Capture Buttons -->
        <div style="display: flex; gap: 10px; margin-top: 12px;">
            <button id="btn-hit" style="flex: 1; background-color: #1f6feb; color: white; border: none; padding: 10px; border-radius: 6px; font-weight: bold; cursor: pointer;">📍 Capture as Mark Hit</button>
            <button id="btn-landing" style="flex: 1; background-color: #1f6feb; color: white; border: none; padding: 10px; border-radius: 6px; font-weight: bold; cursor: pointer;">📍 Capture as Mark Landing</button>
        </div>
        <div id="status-msg" style="color: #3fb950; font-size: 12px; margin-top: 8px; text-align: center; font-family: monospace; min-height: 18px;"></div>
    </div>

    <!-- Official Streamlit Component Communication Script -->
    <script src="https://streamlit.com/components/streamlit-component-lib.js"></script>
    
    <script>
        const video = document.getElementById('vid');
        const timeDisplay = document.getElementById('time-display');
        const btnHit = document.getElementById('btn-hit');
        const btnLanding = document.getElementById('btn-landing');
        const statusMsg = document.getElementById('status-msg');

        let currentTime = 0.0;

        video.addEventListener('timeupdate', function() {{
            currentTime = video.currentTime;
            timeDisplay.innerText = currentTime.toFixed(3) + " s";
        }});

        function sendValueToPython(actionType, timeVal) {{
            const roundedTime = Number(timeVal.toFixed(3));
            if (actionType === 'hit') {{
                statusMsg.innerText = "Captured Hit Time: " + roundedTime + " s";
            }} else {{
                statusMsg.innerText = "Captured Landing Time: " + roundedTime + " s";
            }}
            
            // Send back using Streamlit's official component API
            window.Streamlit.setComponentValue({{ action: actionType, time: roundedTime }});
        }}

        btnHit.addEventListener('click', function() {{
            sendValueToPython('hit', currentTime);
        }});

        btnLanding.addEventListener('click', function() {{
            sendValueToPython('landing', currentTime);
        }});

        // Set initial frame height
        window.Streamlit.setFrameHeight(450);
    </script>
    """
    
    # Render component and catch action outputs
    component_output = components.html(player_component_html, height=560)

    # Handle data sent back from the component buttons
    if isinstance(component_output, dict):
        if component_output.get("action") == "hit":
            st.session_state.hit_time = component_output.get("time", st.session_state.hit_time)
        elif component_output.get("action") == "landing":
            st.session_state.landing_time = component_output.get("time", st.session_state.landing_time)

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
