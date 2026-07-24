"""
detector.py

Loads the YOLO model and performs vehicle detection.
"""

import cv2
from ultralytics import YOLO
from pathlib import Path


class VehicleDetector:

    def __init__(self):

        from ultralytics import YOLO

        self.model = YOLO("yolov8n.pt")

        # Vehicle classes (COCO IDs)
        self.vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        detections = []

        annotated_frame = frame.copy()

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls = int(box.cls[0])

                if cls not in self.vehicle_classes:
                    continue

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = self.vehicle_classes[cls]

                detections.append(
                    {"label": label, "confidence": conf, "bbox": (x1, y1, x2, y2)}
                )

                color = (0, 255, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    annotated_frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        return annotated_frame, detections

    def process_video(self, stream_url):

        print(stream_url)

        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            raise Exception("Unable to open stream.")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame, detections = self.detect(frame)

            yield frame, detections

        cap.release()
