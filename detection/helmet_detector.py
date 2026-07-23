import cv2
import math
from ultralytics import YOLO
import cvzone


class HelmetDetector:

    def __init__(self, model_path="best.pt", confidence=0.4):

        self.model = YOLO(model_path)
        self.confidence = confidence

        self.class_names = [
            "With Helmet",
            "Without Helmet"
        ]

    def process_video(self, stream_url):

        cap = cv2.VideoCapture(stream_url)

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = self.model(frame)

            detections = []

            for r in results:

                for box in r.boxes:

                    conf = float(box.conf[0])

                    if conf < self.confidence:
                        continue

                    cls = int(box.cls[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    label = self.class_names[cls]

                    color = (0,255,0)

                    if label == "Without Helmet":
                        color = (0,0,255)

                    cvzone.cornerRect(
                        frame,
                        (x1,y1,x2-x1,y2-y1),
                        colorR=color
                    )

                    cvzone.putTextRect(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1,y1-10),
                        scale=0.8,
                        thickness=1,
                        colorR=color
                    )

                    detections.append({
                        "label":label,
                        "confidence":conf
                    })

            yield frame, detections

        cap.release()