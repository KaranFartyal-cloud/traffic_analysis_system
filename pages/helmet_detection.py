import streamlit as st
import cv2

from youtube.downloader import YouTubeDownloader
from detection.helmet_detector import HelmetDetector

st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Helmet Detection")

confidence = st.sidebar.slider(
    "Confidence",
    0.10,
    1.00,
    0.40
)

start = st.sidebar.button("▶ Start Detection")

import os

video_folder = "assets/videos"

videos = [
    f for f in os.listdir(video_folder)
    if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
]

selected_video = st.selectbox(
    "Select Video",
    videos
)

left,right = st.columns([3,1])

video_placeholder = left.empty()

with right:

    st.subheader("Live Statistics")

    helmet_metric = st.empty()

    nohelmet_metric = st.empty()

    total_metric = st.empty()

    compliance_metric = st.empty()

if start:

    video_path = os.path.join(video_folder, selected_video)

    detector = HelmetDetector(
        confidence=confidence
    )

    for frame, detections in detector.process_video(video_path):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

        with_helmet = 0
        without_helmet = 0

        for obj in detections:

            if obj["label"] == "With Helmet":
                with_helmet += 1
            else:
                without_helmet += 1

        total = with_helmet + without_helmet

        compliance = (
            round((with_helmet / total) * 100, 2)
            if total > 0 else 0
        )

        helmet_metric.metric("With Helmet", with_helmet)
        nohelmet_metric.metric("Without Helmet", without_helmet)
        total_metric.metric("Total Riders", total)
        compliance_metric.metric("Helmet Compliance", f"{compliance}%")