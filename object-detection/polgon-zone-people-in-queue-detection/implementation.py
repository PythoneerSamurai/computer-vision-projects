# imports
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# initializing the model and the VideoCapture object
MODEL = YOLO('/kaggle/input/person-detection-model-dataset/runs/detect/train/weights/last.pt')
capture = cv2.VideoCapture('/kaggle/input/video-data/cr.mp4')

# defining the polygon regions
polygons = [
    np.array([
        [686, 78],[786, 70],[936, 624],[728, 636],[686, 78]
    ]),
    np.array([
        [948, 23],[1100, 15],[1216, 127],[1268, 375],[1196, 467],[948, 23]
    ])
]

# initializing the BoxAnnotator, PolygonZones, PolygonZoneAnnotator objects
box_annotator = sv.BoxAnnotator()
zone_one = sv.PolygonZone(polygon=polygons[0], triggering_anchors=[sv.Position.CENTER])
zone_two = sv.PolygonZone(polygon=polygons[1], triggering_anchors=[sv.Position.TOP_CENTER])
zones = [zone_one, zone_two]
zone_annotators = [sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.from_rgb_tuple((255, 0, 0)), thickness=4) for zone
                   in zones]

# initializing the video codec and the VideoWriter object
video_codec = cv2.VideoWriter.fourcc(*'mp4v')
video = cv2.VideoWriter('/kaggle/working/processed_video.mp4', video_codec, 20.0, (1280, 720))

# defining a function to process individual frames
def process_frame(frame, result):
    detections = sv.Detections.from_ultralytics(result)

    for zone, zone_annotator in zip(zones, zone_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
        frame = zone_annotator.annotate(scene=frame)

    return frame

# a loop to break the video into frames, and also to write the output video with processed frames
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    results = MODEL(frame, save=False)
    for result in results:
        processed_frame = process_frame(frame, result)
        video.write(processed_frame)
