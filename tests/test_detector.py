from youtube.downloader import YouTubeDownloader
from detection.detector import VehicleDetector

import cv2

url = input("Enter YouTube URL : ")

yt = YouTubeDownloader()

video = yt.get_stream_url(url)
detector = VehicleDetector()

for frame, detections in detector.process_video(video["stream_url"]):

    print(detections)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()