# imports
from ultralytics import YOLO
from collections import defaultdict, deque
import supervision as sv
import cv2
import numpy as np

# initializing the model object using my own trained yolov10x model
MODEL = YOLO("/kaggle/input/vehicle-detection-last-model/pytorch/default/1/last.pt")

# initializing the VideoCapture object (to split the input video into frames), and the VideoWriter object (to join the processed frames into the output video)
CAPTURE = cv2.VideoCapture("/kaggle/input/highway-video/stabilized-highway.mp4")
WIDTH, HEIGHT, FPS = (int(CAPTURE.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))  # getting the width, height, and fps of the video
WRITER = cv2.VideoWriter("/kaggle/working/output-video.mp4", cv2.VideoWriter.fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

'''
Defining the source and target matrix for perspective transformation.
When trying to estimate an object's speed using a single camera, it becomes very inaccurate if we use the pixel values that the object travels in order
to calculate it's speed, that is if the camera is placed at the front (not a top-down view). Taking the example of a highway, in an image the farther part of
the highway is going to be smaller, and the pixel values travelled by the objects are going to be less, even though in reality the highway is equally
distanced in real life. This is why we have to transform the highway from the perspective of single camera, to a top down view of the highway, the latter 
having width and height same as that of the highway in reality. The implementation is quite simple, we just have to define the source matrix (having the
polygon coordinates of the section of the highway whose perspective we want to transform), and the target matrix (having the polygon coordinates of the
actual highway section in real life), afterwards we can just call the cv2.getPerspectiveTransform() function to transform the source perspective to the
target perpective. Now all that's left is to convert the coordinates travelled by the objects from the source perspective to the target perspective, for
the calculation of the distance, using the target pixel values. This is easily accomplished by calling the cv2.perspectiveTransform() function, passing
the source points and the transformed matrix (output of cv2.getPerspectiveTranform() function) as arguments. Lastly we can use these transformed points
to calculate the distance travelled by the objects. Note that we need to know the actual width and height of the target highway section
to accomplish this with high accuracy. In this implementation, I wasn't quite sure of the width and height, still I read an article and calculated the
width by adding the standard lane widths, and shoulder widths. However I had to make a raw guess of the height of the section, which I assumed to be
60 meters long, this gave me acceptable vehicle speeds.
'''
SOURCE_MATRIX = np.array([
    [448, 141],[832, 137],[1055, 222],[198, 222]
])
TARGET_MATRIX = np.array([
    [0, 0], [39, 0], [39, 60], [0, 60]
])
TRANSFORMED_MATRIX = cv2.getPerspectiveTransform(SOURCE_MATRIX.astype(np.float32), TARGET_MATRIX.astype(np.float32))

# a function that transforms the coordinates that an object travelled on the source image to the target perspective
def transform_points(points):
    if points.size == 0:
        return points
    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped_points, TRANSFORMED_MATRIX)
    return transformed_points.reshape(-1, 2)


# defining the polygon for annotation on the frames, and for filtering the detections (only the detections present in the zone will be kept).
POLYGON = np.array([
    [444, 140],[829, 136],[1281, 303],[1281, 688],[1251, 718],[36, 718],[36, 718],[3, 684],[3, 292],[444, 140]
])

# initializing the PolygonZone object for detection filtering
POLYGON_ZONE = sv.PolygonZone(polygon=POLYGON, triggering_anchors=[sv.Position.BOTTOM_CENTER])
# initializing the PolygonZoneAnnotator object to annotate individual frames with the polygon zone defined above.
POLYGON_ZONE_ANNOTATOR = sv.PolygonZoneAnnotator(zone=POLYGON_ZONE, thickness=3, color=sv.Color.RED)

# initializing the RoundBoxAnnotator object for annotating the filtered detections with a round-edged bounding box.
BOX_ANNOTATOR = sv.RoundBoxAnnotator(thickness=3, color=sv.Color.RED)
# intializing the TraceAnnotator object for annotating individual frames with colored lines following the coordinates travelled by objects.
TRACE_ANNOTATOR = sv.TraceAnnotator(trace_length=60, thickness=3)
# initializing the LabelAnnotator for labelling each object with it's speed and tracker id.
LABEL_ANNOTATOR = sv.LabelAnnotator(color=sv.Color.BLACK)
# initializing the ByteTrack object to keep track of the filtered detections through out the frames, and to get their tracker ids for labelling.
TRACKER = sv.ByteTrack(frame_rate=FPS)
'''
initializing a defaultdict object for storing the source pixel coordinates travelled by the objects (for conversion into target points).
deque (doubly ended queue) is used here because it allows quicker appending and popping of elements on both ends of the container.
later the starting y-coordinate of the detected objects and the ending y-coordinates of the object will be used to calculate the distance, and
the starting y-coordinate is present at the end of the deque, and the ending y-coordinate is present at the beginning.
'''
coordinates = defaultdict(lambda: deque(maxlen=FPS))

# a function that processes individual frames
def frame_processor(frame, inference):
    detections = sv.Detections.from_ultralytics(inference)  # converting detections from yolov10x format to supervision format.
    detections = detections[detections.class_id == 2]  # initial filtering to remove unwanted objects being detected.
    detections = TRACKER.update_with_detections(detections)  # tracking the detected objects
    detections = detections[POLYGON_ZONE.trigger(detections)]  # final filtering to keep  only those objects that are present in the polygon zone.

    points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)  # getting the points travelled by the objects on the souce image
    points = transform_points(points).astype(int)  # transforming those points to the target perspective
    
    labels = []
    '''
    looping through tracker ids, saving the points they travelled, calculating the distance when the number of frames they travelled is more than half
    of the FPS.
    note that we are only using the y-coordinate of the detected object's bounding box bottom center, this is because in reality the highway section
    being used is straight, thus the x-axis is not changing, only the y-axis is changing.
    '''
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)
        if len(coordinates[tracker_id]) < FPS/2:
            labels.append(f"#{tracker_id}")
        else:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start-coordinate_end)
            time = len(coordinates[tracker_id]) / FPS
            speed = (distance/time)*3.6
            labels.append(f"#{tracker_id} {int(speed)} km/h")
          
    # annotating the frame.
    frame = BOX_ANNOTATOR.annotate(frame, detections)
    frame = TRACE_ANNOTATOR.annotate(frame, detections)
    frame = LABEL_ANNOTATOR.annotate(frame, detections, labels)
    frame = POLYGON_ZONE_ANNOTATOR.annotate(frame)

    WRITER.write(frame)

# looping through frames of the input video.
while CAPTURE.isOpened():
    success, frame = CAPTURE.read()
    if not success:
        break
    else:
        results = MODEL(frame, imgsz=(WIDTH, 1088), save=False, conf=0.25)  # using my model to carry inference on frames.
        for inference in results:
            frame_processor(frame, inference)
