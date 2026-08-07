import cv2
import tempfile
import os
import time
import pandas as pd
import streamlit as st
import supervision as sv
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="VisionAI | Enterprise Object Tracking & Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise UI / UX Styling with Glassmorphism, Google Fonts, and Micro-Animations
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap">

<style>
    /* Global Theme Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151D2A 0%, #0B0F17 100%);
        color: #E2E8F0;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 22, 35, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-top: 1rem;
    }

    /* Header Banner */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 1.75rem;
        background: rgba(18, 26, 42, 0.65);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
    }

    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #00F59B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 2px;
        font-weight: 500;
    }

    /* Live Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Metric Cards with Glowing Hover Effects */
    .metric-card {
        position: relative;
        background: rgba(18, 26, 42, 0.6);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .metric-card-in {
        border-left: 4px solid #10B981;
    }
    .metric-card-in:hover {
        box-shadow: 0 12px 30px -10px rgba(16, 185, 129, 0.35);
    }

    .metric-card-out {
        border-left: 4px solid #F43F5E;
    }
    .metric-card-out:hover {
        box-shadow: 0 12px 30px -10px rgba(244, 63, 94, 0.35);
    }

    .metric-card-total {
        border-left: 4px solid #3B82F6;
    }
    .metric-card-total:hover {
        box-shadow: 0 12px 30px -10px rgba(59, 130, 246, 0.35);
    }

    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metric-value-in {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }

    .metric-value-out {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F43F5E;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
        text-shadow: 0 0 20px rgba(244, 63, 94, 0.3);
    }

    .metric-value-total {
        font-size: 2.5rem;
        font-weight: 800;
        color: #60A5FA;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.3);
    }

    /* Video Frame Styling */
    .video-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }

    /* Custom Gradient Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Top Enterprise Banner Header
st.markdown("""
<div class="header-container">
    <div>
        <div class="brand-title">⚡ VisionAI Analytics Engine</div>
        <div class="brand-subtitle">Real-Time YOLOv8 Object Tracking, Trajectory Motion & Line-Zone Traffic Intelligence</div>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        System Active • GPU/CPU Ready
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration Controls
st.sidebar.markdown("## 🎛️ Video Stream Source")
video_option = st.sidebar.selectbox(
    "Select Input Media",
    ["highway_car.mp4", "highway_car2.mp4", "mall_counting.mp4", "Upload Custom Video"]
)

st.sidebar.markdown("## 🧠 AI Detection & Model")
model_option = st.sidebar.selectbox(
    "YOLO Architecture",
    ["yolov8s.pt", "yolov8m.pt"]
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)

target_class = st.sidebar.selectbox(
    "Target Object Filter",
    ["Cars / Vehicles (Class 2)", "Pedestrians / People (Class 0)", "All Categories"]
)

if target_class == "Cars / Vehicles (Class 2)":
    classes = [2]
elif target_class == "Pedestrians / People (Class 0)":
    classes = [0]
else:
    classes = None

st.sidebar.markdown("## 🎨 Visual Tracking FX")
show_traces = st.sidebar.checkbox("Enable Motion Trajectory Trails", value=True)
show_boxes = st.sidebar.checkbox("Enable Bounding Boxes & ID Labels", value=True)

# Handle Custom Video Upload
video_path = None
if video_option == "Upload Custom Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI File", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
else:
    video_path = video_option

if video_path and os.path.exists(video_path):
    @st.cache_resource
    def load_yolo_model(model_name):
        return YOLO(model_name)

    model = load_yolo_model(model_option)

    # Inspect Video Resolution
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    if ret:
        H, W = sample_frame.shape[:2]
        cap.release()

        st.sidebar.markdown("## 📏 Counting Line Position")
        line_y_percent = st.sidebar.slider("Line Vertical Position (% Height)", 10, 90, 50, 5)
        line_y = int(H * (line_y_percent / 100.0))

        # Initialize Supervision Components
        line_zone = sv.LineZone(
            start=sv.Point(0, line_y),
            end=sv.Point(W, line_y)
        )
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.8)
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        trace_annotator = sv.TraceAnnotator(trace_length=30)

        st.sidebar.markdown("---")
        start_btn = st.sidebar.button("▶️ Launch AI Inference Stream", use_container_width=True)

        # Layout Split: Main Video (Left) vs Real-Time Analytics Cards (Right)
        col_video, col_stats = st.columns([3.2, 1.2])

        with col_stats:
            st.markdown("### 📊 Real-Time Analytics")
            in_metric = st.empty()
            out_metric = st.empty()
            total_metric = st.empty()
            chart_placeholder = st.empty()

            # Render Initial Cards
            in_metric.markdown("""
            <div class="metric-card metric-card-in">
                <div class="metric-title">🟩 IN / ENTRY COUNT</div>
                <div class="metric-value-in">0</div>
            </div>
            """, unsafe_allow_html=True)

            out_metric.markdown("""
            <div class="metric-card metric-card-out" style="margin-top: 1rem;">
                <div class="metric-title">🟥 OUT / EXIT COUNT</div>
                <div class="metric-value-out">0</div>
            </div>
            """, unsafe_allow_html=True)

            total_metric.markdown("""
            <div class="metric-card metric-card-total" style="margin-top: 1rem;">
                <div class="metric-title">🟦 TOTAL TRAFFIC VOLUME</div>
                <div class="metric-value-total">0</div>
            </div>
            """, unsafe_allow_html=True)

        with col_video:
            st_frame = st.empty()

        if start_btn:
            tracking_generator = model.track(
                source=video_path,
                stream=True,
                classes=classes,
                conf=confidence_threshold,
                imgsz=640
            )

            history_data = []
            frame_counter = 0

            for result in tracking_generator:
                frame_counter += 1
                frame = result.orig_img.copy()
                detections = sv.Detections.from_ultralytics(result)

                # Draw Motion Trail Traces
                if show_traces and len(detections) > 0:
                    frame = trace_annotator.annotate(scene=frame, detections=detections)

                # Draw Bounding Boxes & Labels
                if show_boxes and len(detections) > 0:
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id] if detections.tracker_id is not None else []
                    if labels:
                        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                # Trigger Line Counter & Annotate Line
                line_zone.trigger(detections)
                line_annotator.annotate(frame, line_zone)

                # Convert to RGB & Render Stream
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

                # Live Counts
                in_count = line_zone.in_count
                out_count = line_zone.out_count
                total_count = in_count + out_count

                # Update Stat Cards
                in_metric.markdown(f"""
                <div class="metric-card metric-card-in">
                    <div class="metric-title">🟩 IN / ENTRY COUNT</div>
                    <div class="metric-value-in">{in_count}</div>
                </div>
                """, unsafe_allow_html=True)

                out_metric.markdown(f"""
                <div class="metric-card metric-card-out" style="margin-top: 1rem;">
                    <div class="metric-title">🟥 OUT / EXIT COUNT</div>
                    <div class="metric-value-out">{out_count}</div>
                </div>
                """, unsafe_allow_html=True)

                total_metric.markdown(f"""
                <div class="metric-card metric-card-total" style="margin-top: 1rem;">
                    <div class="metric-title">🟦 TOTAL TRAFFIC VOLUME</div>
                    <div class="metric-value-total">{total_count}</div>
                </div>
                """, unsafe_allow_html=True)

                # Collect trend history every 10 frames
                if frame_counter % 10 == 0:
                    history_data.append({"Frame": frame_counter, "IN": in_count, "OUT": out_count})
                    df_chart = pd.DataFrame(history_data).set_index("Frame")
                    chart_placeholder.area_chart(df_chart, height=180, color=["#10B981", "#F43F5E"])

    else:
        st.error(f"Unable to read video stream from: {video_path}")
else:
    st.info("👈 Select a video source or upload custom media from the sidebar control panel.")
