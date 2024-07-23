# imports
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# initializing the model object (using my own trained yolov10x model), and the VideoCapture object (to read the frames of the input video)
MODEL = YOLO('/kaggle/input/temporary-dataset/last.pt/last.pt')
CAPTURE = cv2.VideoCapture('/kaggle/input/temporary-dataset/round_about_video.mp4')

# defining the polygons for the entry zones towards the intersections
ENTRY_POLYGONS = [
    np.array([
        [613, 285], [813, 289], [816, 101], [709, 97], [697, 154], [670, 212], [639, 258], [613, 285]
    ]),
    np.array([
        [1194, 393], [1204, 525], [1459, 528], [1436, 393], [1194, 393]
    ]),
    np.array([
        [1037, 895], [1231, 901], [1219, 1080], [1034, 1080], [1037, 895]
    ]),
    np.array([
        [435, 705], [457, 571], [669, 618], [644, 752], [435, 705]
    ])
]

# defining the polygons for the exit zones towards the intersections
EXIT_POLYGONS = [
    np.array([
        [1076, 140], [1144, 140], [1140, 293], [1065, 290], [1076, 140]
    ]),
    np.array([
        [1260, 584], [1279, 734], [1447, 677], [1432, 552], [1260, 584]
    ]),
    np.array([
        [713, 868], [782, 868], [761, 1040], [698, 1040], [713, 868]
    ]),
    np.array([
        [476, 539], [549, 408], [607, 405], [655, 393], [676, 375], [676, 548], [476, 539]
    ])
]

# four entry zones, four exit zones = four colors.
COLOR_HEX_LIST = sv.ColorPalette.from_hex(
    ['#ff0000', '#00ff00', '#0000ff', '#5d8aa8']
)
TRACKER = sv.ByteTrack() # tracker object to keep track of detections throughout the frames
ENTRY_ZONES = [sv.PolygonZone(polygon=polygon) for polygon in ENTRY_POLYGONS] # initializing the PolygonZone objects for the four entry polygons
EXIT_ZONES = [sv.PolygonZone(polygon=polygon) for polygon in EXIT_POLYGONS] # initializing the PolygonZone objects for the four exit polygons
ENTRY_ZONE_ANNOTATORS = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=COLOR_HEX_LIST.by_idx(idx),
        thickness=4
    )
    for (idx, zone)
    in enumerate(ENTRY_ZONES)
] # intializing the PolygonZoneAnnotator objects for annotating the frames with the four entry polygon zones
EXIT_ZONE_ANNOTATORS = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=COLOR_HEX_LIST.by_idx(idx),
        thickness=4
    )
    for (idx, zone)
    in enumerate(EXIT_ZONES)
] # intializing the PolygonZoneAnnotator objects for annotating the frames with the four exit polygon zones
LABEL_ANNOTATORS = [
    sv.LabelAnnotator(color=COLOR_HEX_LIST.by_idx(idx))
    for (idx, _)
    in enumerate(ENTRY_ZONES)
] # LabelAnnotator objects to annotate the filtered detections (those present in the polygon zones)
BOX_ANNOTATORS = [
    sv.BoxAnnotator(
        color=COLOR_HEX_LIST.by_idx(idx)
    )
    for (idx, _)
    in enumerate(ENTRY_ZONES)
] # BoxAnnotator objects for annotating the filtered detections with bounding boxes
TRACE_ANNOTATORS = [
    sv.TraceAnnotator(
        color=COLOR_HEX_LIST.by_idx(idx),
        thickness=2,
        trace_length=100
    )
    for (idx, _)
    in enumerate(ENTRY_ZONES)
] # TraceAnnotator objects for annotating the detections with a trace mark (in order to analyze their flow)

VIDEO_CODEC = cv2.VideoWriter.fourcc(*'mp4v') # video codec for the output video
OUTPUT_VIDEO = cv2.VideoWriter('/kaggle/working/output-video.mp4', VIDEO_CODEC, 30.0, (1920, 1080)) # VideoWriter object to merge the frames into an output video

# a function that processes individual frames
def frame_processor(frame, inferences):
    
    entry_vehicle_count = 0
    exit_vehicle_count = 0
    
    detections = sv.Detections.from_ultralytics(inferences) # converting detections from yolov10x format to supervision format
    detections = TRACKER.update_with_detections(detections=detections) # updating the tracker to track the detections

    # loop for annotating and processing the entry polygon zones
    for zone, zone_annotator, box_annotator, trace_annotator, label_annotator in zip(ENTRY_ZONES, ENTRY_ZONE_ANNOTATORS,
                                                                                     BOX_ANNOTATORS, TRACE_ANNOTATORS,
                                                                                     LABEL_ANNOTATORS):
        filterate = zone.trigger(detections=detections) # getting boolean values for the detections that are present in the entry polygon zones
        detections_filtered = detections[filterate] # using the filterate to filter only those detections that are present in the entry polygon zones
        entry_vehicle_count += len(detections_filtered) # counting the number of vehicles entering the intersection
        frame = box_annotator.annotate(detections=detections_filtered, scene=frame) # annotating the filtered detections with bounding boxes
        labels = [f'IN#{tracker_id}' for tracker_id in detections_filtered.tracker_id] # defining the labels for the label annotator
        frame = trace_annotator.annotate(detections=detections, scene=frame) # tracing the detections
        frame = label_annotator.annotate(detections=detections_filtered, scene=frame, labels=labels) # labeling the filtered detections with their tracker ids
        frame = zone_annotator.annotate(scene=frame) # annotating the frames with the defined polygon zones
        
    cv2.putText(frame, f'Vehicles entering intersection: {entry_vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3) # putting the number of vehicles entering the intersection on the frame

    # loop for annotating and processing the exit polygon zones (in-loop description is same as the above, however entry=exit)
    for zone, zone_annotator, box_annotator, trace_annotator, label_annotator in zip(EXIT_ZONES, EXIT_ZONE_ANNOTATORS,
                                                                                     BOX_ANNOTATORS, TRACE_ANNOTATORS,
                                                                                     LABEL_ANNOTATORS):
        filterate = zone.trigger(detections=detections)
        detections_filtered = detections[filterate]
        exit_vehicle_count += len(detections_filtered)
        frame = box_annotator.annotate(detections=detections_filtered, scene=frame)
        labels = [f'OUT#{tracker_id}' for tracker_id in detections_filtered.tracker_id]
        frame = trace_annotator.annotate(detections=detections, scene=frame)
        frame = label_annotator.annotate(detections=detections_filtered, scene=frame, labels=labels)
        frame = zone_annotator.annotate(scene=frame)

    cv2.putText(frame, f'Vehicles exiting intersection: {exit_vehicle_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    return frame # returning the frame to merge into the output video

# main loop
while CAPTURE.isOpened():
    success, frame = CAPTURE.read()
    if not success: # if failure to load the frames (or the frames end), break the loop
        break
    else:
        inferences = MODEL(frame, imgsz=(1920, 1088), save=False, conf=0.15) # using my model to detect the vehicles present in the zones
        for inference in inferences: # looping through all the detections
            processed_frame = frame_processor(frame, inference) # getting the processed frame
            OUTPUT_VIDEO.write(processed_frame) # writing the output video
