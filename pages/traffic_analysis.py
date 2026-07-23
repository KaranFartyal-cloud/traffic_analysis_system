import streamlit as st
import cv2
import time
import pandas as pd
from youtube.downloader import YouTubeDownloader
from pathlib import Path
import os
from detection.detector import VehicleDetector

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Traffic Analysis System", page_icon="🚦", layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🚦 AI Traffic Analysis System")

st.markdown(
    "Analyze live YouTube traffic videos using **YOLO**, **OpenCV**, and **Machine Learning**."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Settings")

confidence = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.40)

start = st.sidebar.button("▶ Start Analysis")

# -----------------------------
# URL
# -----------------------------


BASE_DIR = Path(__file__).resolve().parent.parent
video_folder = BASE_DIR / "assets" / "traffic"

if not video_folder.exists():
    st.error(f"Video folder not found:\n{video_folder}")
    st.stop()

videos = [
    f.name
    for f in video_folder.iterdir()
    if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
]

if not videos:
    st.error("No videos found in assets/videos")
    st.stop()

selected_video = st.selectbox("Select Video", videos)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([3, 1])

video_placeholder = left.empty()

with right:
    st.subheader("Live Statistics")

    cars_metric = st.empty()
    bikes_metric = st.empty()
    bus_metric = st.empty()
    truck_metric = st.empty()

    st.divider()

    total_metric = st.empty()

    st.divider()

    density_metric = st.empty()

    st.divider()

    prediction_metric = st.empty()

# -----------------------------
# Start
# -----------------------------
if start:

    video_path = str(video_folder / selected_video)

    st.success(f"Selected Video: {selected_video}")

    detector = VehicleDetector()

    # -------------------------
    # Process Frames
    # -------------------------
    graph_placeholder = st.empty()

    start_time = time.time()

    minute_data = []

    current_minute = 0

    for frame, detections in detector.process_video(video_path):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        # -------------------------
        # Count Vehicles
        # -------------------------
        cars = 0
        bikes = 0
        buses = 0
        trucks = 0

        for obj in detections:

            if obj["label"] == "Car":
                cars += 1

            elif obj["label"] == "Motorcycle":
                bikes += 1

            elif obj["label"] == "Bus":
                buses += 1

            elif obj["label"] == "Truck":
                trucks += 1

        total = len(detections)

        elapsed = time.time() - start_time

        minute = int(elapsed // 60)

        if minute > current_minute:
            minute_data.append(
                {
                    "Minute": current_minute,
                    "Cars": cars,
                    "Motorcycles": bikes,
                    "Buses": buses,
                    "Trucks": trucks,
                    "Total": total,
                }
            )

            current_minute = minute

            df = pd.DataFrame(minute_data)

            graph_placeholder.line_chart(df.set_index("Minute"))

        # -------------------------
        # Density
        # -------------------------
        if total < 10:
            density = "🟢 LOW"

        elif total < 25:
            density = "🟡 MEDIUM"

        else:
            density = "🔴 HIGH"

        # -------------------------
        # Update Dashboard
        # -------------------------
        cars_metric.metric("Cars", cars)

        bikes_metric.metric("Motorcycles", bikes)

        bus_metric.metric("Bus", buses)

        truck_metric.metric("Truck", trucks)

        total_metric.metric("Total Vehicles", total)

        density_metric.metric("Traffic Density", density)
