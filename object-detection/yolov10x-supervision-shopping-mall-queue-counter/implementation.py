# imports
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# initializing the model object (using the yolov10x model I trained), and the VideoCapture object to read the test video
MODEL = YOLO('/kaggle/input/scut-head-output/runs/detect/train/weights/last.pt')
capture = cv2.VideoCapture('/kaggle/input/test-video/cr.mp4')

# defining the polygons to count the objects enclosed by them
polygons = [
    np.array([
    [773, 41],[682, 49],[728, 636],[732, 636],[936, 624],[773, 41]
    ]),
    np.array([
    [956, 96],[948, 12],[1100, 16],[1224, 124],[1248, 408],[1196, 472],[956, 96]
    ])
]

# initializing the TriangleAnnotator, PolygonZones, PolygonZoneAnnotator objects
triangle_annotator = sv.TriangleAnnotator()
zones = [sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.BOTTOM_CENTER]) for polygon in polygons]
zone_annotators = [sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.from_rgb_tuple((255, 0, 0)), thickness=4) for zone
                   in zones]

# initializng the video codec object and the VideoWriter object to write the output video
video_codec = cv2.VideoWriter.fourcc(*'mp4v')
video = cv2.VideoWriter('/kaggle/working/processed_video.mp4', video_codec, 25.0, (1280, 720))

# defining a function to filter the detections, selecting only those that are enclosed by the polygons, and annotating them.
def process_frame(frame, result):
    # converting detections from yolov10x (ultralytics) format to supervision format
    detections = sv.Detections.from_ultralytics(result)
    
    for zone, zone_annotator in zip(zones, zone_annotators):
        # using the PolygonZone objects to identify the objects enclosed by the polygons
        mask = zone.trigger(detections=detections)
        # filtering the detections to select only those which are enclosed by the polygons
        detections_filtered = detections[mask]
        # annotating the filtered detections with an overhead triangle marker using the TriangleAnnotator object
        frame = triangle_annotator.annotate(scene=frame, detections=detections_filtered)
        # annotating the frame with the polygons that were initialized before.
        frame = zone_annotator.annotate(scene=frame)
    
    return frame

# main loop
while capture.isOpened():
    # breaking the test (input) video into frames.
    ret, frame = capture.read()
    # if no more frames left break the loop
    if not ret:
        break
    # carrying inference on the frames using my model
    results = MODEL(frame, save=False, imgsz=(1280, 720), conf=0.4)
    # looping through all the detections
    for result in results:
        processed_frame = process_frame(frame, result)
        # writing the output video
        video.write(processed_frame)
