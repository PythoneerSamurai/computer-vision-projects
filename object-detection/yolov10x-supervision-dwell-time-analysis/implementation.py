import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# a class that counts the time an object spends in the defined polygon zones
class FPSBasedTimer():
    def __init__(self, fps=30):
        self.fps = fps  # defining the video fps
        self.frame_id = 0  # the frame at which an object enters the polygon zones
        self.tracker_id2frame_id = {}  # a dictionary that maps each tracker to the frame it entered the polygon zones

    # a function that counts the time an object spends in the polygon zones, the frames of the videos keep on getting incremented, and the first time
    # an object enters the polygon zone is saved, when it leaves a simple formula calulates the time it spent in the polygon zone.
    def tick(self, detections):
        self.frame_id += 1
        times = []
        for tracker_id in detections.tracker_id:
            self.tracker_id2frame_id.setdefault(tracker_id, self.frame_id)
            start_frame_id = self.tracker_id2frame_id[tracker_id]
            time_duration = (self.frame_id - start_frame_id)/self.fps  # formula
            times.append(time_duration)
            
        return np.array(times)

# initializing the model object using my yolov10x model that detects human heads, and the VideoCapture object to read individual frames
MODEL = YOLO('/kaggle/input/yolov10x-supervision-queue-count-model/pytorch/default/1/last.pt')
CAPTURE = cv2.VideoCapture('/kaggle/input/test-video/cr.mp4')

# defining the polygon zones
POLYGONS = [
    np.array([
    [773, 41],[682, 49],[728, 636],[732, 636],[936, 624],[773, 41]
    ]),
    np.array([
    [956, 96],[948, 12],[1100, 16],[1224, 124],[1248, 408],[1196, 472],[956, 96]
    ])
]

# initializing the BoxAnnotator, PolygonZones, PolygonZoneAnnotator, ByteTrack, LabelAnnotator objects.
LABEL_ANNOTATOR = sv.LabelAnnotator()  # for annotating the detections with the time they spend in the polygon zones.
BOX_ANNOTATOR = sv.BoxAnnotator()  # for annotating the human heads present in the polygon zones with bounding boxes.
TRACKER = sv.ByteTrack()  # for tracking detections throughout frames, in order to calculate the time at which they entered and left the zones.
ZONES = [sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.BOTTOM_CENTER]) for polygon in POLYGONS]  # initializing PolygonZones.
TIMERS = [FPSBasedTimer() for _ in ZONES]  # initializing two FPSBasedTimer objects for the two polygon zones defined above.
ZONE_ANNOTATORS = [sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.from_rgb_tuple((255, 0, 0)), thickness=4) for zone
                   in ZONES]  # initializing two PolygonZoneAnnotator objects for annotating the frames with the polygons defined above.
VIDEO_CODEC = cv2.VideoWriter.fourcc(*'mp4v')  # specifying the video codec for the output video.
VIDEO = cv2.VideoWriter('/kaggle/working/processed_video.mp4', VIDEO_CODEC, 25.0, (1280, 720))  # initializing the video writer object for generating the output frame.

# defining a function that processes frames 
def process_frame(frame, result):
    detections = sv.Detections.from_ultralytics(result)  # converting detections from yolov10x (ultralytics) format to supervision format.
    detections = TRACKER.update_with_detections(detections=detections)  # updating the tracker with detections present in each frame.
    
    for (idx, zone), zone_annotator in zip(enumerate(ZONES), ZONE_ANNOTATORS):
        mask = zone.trigger(detections=detections)  # getting the detections in the polygon zones by triggering the zones on the detections.
        detections_filtered = detections[mask]  # filtering out the detections present in the polygon zones from the total detections.
        time_in_zone = TIMERS[idx].tick(detections_filtered)  # calculating the time each object spends in the polygon zones.
        labels = [f'time: {time:6.1f}s' for time in time_in_zone]  # defining the labels to annotate each filtered detection with.
        frame = BOX_ANNOTATOR.annotate(scene=frame, detections=detections_filtered)  # annotating the frames with bounding boxes drawn on each filtered detection's head
        frame = LABEL_ANNOTATOR.annotate(scene=frame, detections=detections_filtered, labels=labels)  # annotating the bounding boxes with the time they spent in the zones
        frame = ZONE_ANNOTATOR.annotate(scene=frame)  # annotating the frames with the polygon zones. 
    
    return frame

# main loop
while CAPTURE.isOpened():
    ret, frame = CAPTURE.read()
    if not ret:
        break
    results = MODEL(frame, save=False, imgsz=(1280, 720), conf=0.4)
    for result in results:
        processed_frame = process_frame(frame, result)
        VIDEO.write(processed_frame)
