import streamlit as st
import cv2
import tempfile
from core.engine import MatrixEngine
from core.analytics import AnalyticsManager
import config

st.set_page_config(page_title="Matrix LED Dashboard", layout="wide")
st.title("AI Matrix LED Simulator") 

# Sidebar
st.sidebar.header("Control Panel")
conf_val = st.sidebar.slider("Detection Confidence", 0.1, 1.0, config.DEFAULT_CONF)
light_val = st.sidebar.slider("High Beam Intensity", 0.0, 1.0, config.HIGH_BEAM_INTENSITY)
uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    engine = MatrixEngine()
    
    # Init analytics manager for video
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analytics = AnalyticsManager(height, width)

    col1, col2 = st.columns(2)
    view_orig = col1.empty()
    view_proc = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Processing frame 
        processed_frame, detections = engine.process_frame(frame, conf_val, light_val)
        
        # Heatmap update
        analytics.update_heatmap(detections)
        
        # Update UI
        view_orig.image(frame, channels="BGR", caption="Original Feed")
        view_proc.image(processed_frame, channels="BGR", caption="Matrix LED Active Output")

    cap.release()
    
    # Heatmap display
    st.markdown("---")
    st.subheader("📊 Traffic Density Analysis (Heatmap)")
    st.image(analytics.generate_heatmap(), channels="BGR", use_column_width=True)