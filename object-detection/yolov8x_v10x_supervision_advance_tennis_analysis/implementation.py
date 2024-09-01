# Imports
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Specifying paths to the models, input and output videos, and the radar image to be annotated later.
TENNIS_BALL_DETECTION_MODEL_PATH = "/kaggle/input/yolov8x_v10x_tennis_analysis_models/pytorch/tennis_ball_detection_model/1/best.pt"
PLAYER_DETECTION_MODEL_PATH = "/kaggle/input/yolov8x_v10x_tennis_analysis_models/pytorch/person_detection_model/1/last.pt"
INPUT_VIDEO_PATH = "/kaggle/input/input-video/input_video.mp4"
OUTPUT_VIDEO_PATH = "/kaggle/working/output_video.mp4"
RADAR_COURT_IMAGE_PATH = "/kaggle/input/radar-court/tennis_court_top_down.jpg"

# Loading the radar view tennis court image into memory for carrying annotations later on.
RADAR_COURT_IMAGE = cv2.imread(RADAR_COURT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

# Reading the input video and fetching the width and height of the video frames and the FPS of the video.
INPUT_VIDEO_READER = cv2.VideoCapture(INPUT_VIDEO_PATH)
WIDTH, HEIGHT, FPS = (
    int(INPUT_VIDEO_READER.get(x))
    for x
    in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Specifying various parameters used to define the properties of the output video
FOUR_CC = cv2.VideoWriter.fourcc(*"mp4v")
OUTPUT_VIDEO_FPS = 20.0
VIDEO_RESOLUTION = (2372, HEIGHT)

# Initializing a cv2.VideoWriter object for writing the output video.
OUTPUT_VIDEO_WRITER = cv2.VideoWriter(
    filename=OUTPUT_VIDEO_PATH,
    fourcc=FOUR_CC,
    fps=OUTPUT_VIDEO_FPS,
    frameSize=VIDEO_RESOLUTION,
)

"""
Defining the matrices for perspective transformations. In order to accurately estimate the speed of the ball and
it's location on the court in real life, we must remove the visual distortion caused by the single camera recording
the match. This is because in the video the farther part of the court occupies less pixels in width as compared to the 
closer part of the court, the same goes for the length of the court which cannot be correctly estimated using the 
pixel values of the input video frames. Therefore, we must transform the pixel values occupied by the court in the
video frames to a court having pixel width and height same as the court's in real life. In this way we can use a single
pixel of the transformed court as 1 meter in length. Ultimately, we can use the transformed court to correctly estimate
the speeds of the ball and it's location on the court.
We define three matrices for perspective transformations.
1) The source matrix has the coordinates of each corner of the court in the video.
2) The target real view matrix has the coordinates of each corner of the court in real life (where 1 meter = 1 pixel).
3) The target radar view matrix has the coordinates of each corner of the radar court image loaded above, this image
   will be used to show a top-down view of the court along with the positions the ball hit on the court.
Afterwards, the cv2.getPerspectiveTransform function is used to transform the source matrix to corresponding output
matrices.
Later we define functions to transform the coordinates travelled by the ball on the source matrix to the output 
matrices.
"""
SOURCE_MATRIX = np.array([
    [364, 860], [576, 304], [1332, 300], [1564, 860]
])
TARGET_REAL_VIEW_MATRIX = np.array([
    [0, 77], [0, 0], [35, 0], [35, 77]
])
TARGET_RADAR_VIEW_MATRIX = np.array([
    [0, 563], [0, 0], [352, 0], [352, 563]
])

REAL_VIEW_TRANSFORMED_MATRIX = cv2.getPerspectiveTransform(SOURCE_MATRIX.astype(np.float32),
                                                           TARGET_REAL_VIEW_MATRIX.astype(np.float32))
RADAR_VIEW_TRANSFORMED_MATRIX = cv2.getPerspectiveTransform(SOURCE_MATRIX.astype(np.float32),
                                                            TARGET_RADAR_VIEW_MATRIX.astype(np.float32))

"""
Defining the polygon zones that the court will be divided into for the purpose of filtering detections.
The court polygons will be used to define the regions the ball can be present in the court.
The player polygons are used to define the regions the players move in, this is to filter the players from other
people in the video.
"""
COURT_POLYGONS = [
    np.array([
        [518, 856], [567, 680], [1355, 677], [1412, 856]
    ]), np.array([
        [648, 395], [673, 304], [1239, 304], [1264, 392]
    ]), np.array([
        [364, 859], [579, 304], [673, 304], [518, 856]
    ]), np.array([
        [1564, 857], [1412, 854], [1239, 302], [1333, 302]
    ]), np.array([
        [570, 678], [648, 393], [958, 393], [961, 678]
    ]), np.array([
        [958, 393], [961, 678], [1355, 675], [1267, 393]
    ])
]
PLAYER_POLYGONS = [
    np.array([
        [579, 295], [609, 201], [1300, 198], [1333, 298]
    ]), np.array([
        [364, 865], [1564, 862], [1648, 1056], [300, 1053]
    ])
]

"""
Initializing PolygonZone objects to convert the polygon coordinates defined above into supervision format, and to
specify the triggering anchors (the xy-coordinates) that will decide if an object is inside the zones or not.
Player one polygon zone is the region in the video in which player one can move in, same goes for the player two
polygon zone. Along with the player polygon zones, the court polygon zones are also initialized, these zones will
be used to decide the xy-coordinates of the ball in the court.
"""
PLAYER_ONE_POLYGON_ZONE = sv.PolygonZone(
    polygon=PLAYER_POLYGONS[0],
    triggering_anchors=[sv.Position.CENTER]
)
PLAYER_TWO_POLYGON_ZONE = sv.PolygonZone(
    polygon=PLAYER_POLYGONS[1],
    triggering_anchors=[sv.Position.BOTTOM_CENTER]
)
POLYGON_ZONES = [
    sv.PolygonZone(
        polygon=polygon,
        triggering_anchors=[sv.Position.BOTTOM_CENTER]
    )
    for polygon
    in COURT_POLYGONS
]
POLYGON_ZONES.append(PLAYER_ONE_POLYGON_ZONE)
POLYGON_ZONES.append(PLAYER_TWO_POLYGON_ZONE)

# Initializing a PolygonZoneAnnotator object for annotating the video frames with the polygon zones defined above.
POLYGON_ZONE_ANNOTATORS = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.WHITE,
        thickness=2,
    )
    for _, zone
    in enumerate(POLYGON_ZONES)
]

"""
Initializing supervision annotators for annotating the input video frames with the desired information.
1) LabelAnnotator is used to annotate the frames with the player names (in this case "Djokovic" and "Sonego")
2) RoundBoxAnnotator is used to annotate the frames with boxes having rounded corners around the ball and the players.
"""
LABEL_ANNOTATORS = [
    sv.LabelAnnotator(
        color=sv.Color.RED,
        text_scale=1.0,
        text_thickness=2,
    )
    for _
    in POLYGON_ZONES
]
ROUND_BOX_ANNOTATORS = [
    sv.RoundBoxAnnotator(
        color=sv.Color.RED,
    )
    for _
    in POLYGON_ZONES
]

"""
Initializing trackers to track the ball and the players throughout the video. Tracker ids received by the ball
tracker are used to estimate the speeds of the ball, whereas the information from player tracker is used to keep
track of the players for correct name annotation.
"""
BALL_TRACKER = sv.ByteTrack(frame_rate=FPS)
PLAYER_TRACKER = sv.ByteTrack(frame_rate=FPS)

"""
Instantiating two of my own trained models, a yolov8x model for ball detection and a yolov10x model for player 
detections.
"""
BALL_DETECTION_MODEL = YOLO(TENNIS_BALL_DETECTION_MODEL_PATH)
PLAYER_DETECTION_MODEL = YOLO(PLAYER_DETECTION_MODEL_PATH)

# Specifying the minimum confidence threshold for registering inferences as correct detections.
CONFIDENCE_THRESHOLD = 0.2

"""
Specifying parameters for drawing circles on the radar image, corresponding to the xy-coordinates of the points on the 
court in real life hit by the ball.
"""
RADIUS = 4
COLOR = (0, 36, 255)
THICKNESS = 3

"""
Specifying parameters for annotating the image with the textual information representing various analysis factors of 
the tennis match.
"""
HEADING_TEXT_ORIGIN = (2035, 620)
P1_TEXT_ORIGIN = (1960, 740)
P1_AVG_TEXT_ORIGIN = (1960, 860)
P2_TEXT_ORIGIN = (1960, 800)
P2_AVG_TEXT_ORIGIN = (1960, 920)

HEADING_FONT_THICKNESS = 2
STATS_FONT_THICKNESS = 1
FONT_SCALE = 1.0
FONT_FACE = cv2.FONT_HERSHEY_TRIPLEX
TEXT_COLOR = (255, 255, 255)

"""
Initializing three defaultdict objects for storing and processing information. Dictionaries are not used because
they give a key error if a specific key is not present in the dictionary, however defaultdict fixes this by adding
a non-present key into the dictionary and assigning it an empty value of the variable type.
Another point to be noted is that in the "coordinates" defaultdict, deque is used in place of a normal list. This is 
because deque (doubly ended queue) allows faster access to the first and last index, which will serve beneficial
later on in speed estimation.
"""
coordinates = defaultdict(lambda: deque(maxlen=int(FPS)))
tracker_id_to_coordinates = defaultdict(list)
zone_id_to_ball_speed = defaultdict(list)


# A function that transforms the points travelled by the ball on the source matrix to the real life court matrix.
def transform_real_points(points):
    if points.size == 0:
        return points
    else:
        """
        An additional dimensions is added to the points because the cv2,perspectiveTransform function expects a 3d
        input, whereas xy-coordinates are on the 2d plane, later on this third dimension is removed.
        """
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, REAL_VIEW_TRANSFORMED_MATRIX)
        return transformed_points.reshape(-1, 2)  # removing third dimension


# A function that transforms the points travelled by the ball on the source matrix to the radar court matrix.
def transform_radar_points(points):
    if points.size == 0:
        return points
    else:
        """
        An additional dimensions is added to the points because the cv2,perspectiveTransform function expects a 3d
        input, whereas xy-coordinates are on the 2d plane, later on this third dimension is removed.
        """
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, RADAR_VIEW_TRANSFORMED_MATRIX)
        return transformed_points.reshape(-1, 2)  # removing third dimension


"""
Defining a function used to draw the points on the xy-coordinates hit by the ball on the input video court after
being transformed to the radar court image.
"""


def draw_points(tracker_id, image):
    points_list = []
    y_coordinate_list = []
    x_coordinate_list = []
    max_len = len(list(tracker_id_to_coordinates[tracker_id]))
    for index, value in enumerate(tracker_id_to_coordinates[tracker_id]):
        zone_id, coordinates = value
        x_coordinate, y_coordinate = coordinates
        y_coordinate_list.append(y_coordinate)
        x_coordinate_list.append(x_coordinate)
        if index == max_len - 1:
            """
            We only draw the farthest point travelled by the ball on each side of the court, this is because this
            point is most likely to be the one at which the ball hits the court (as you can see in the output video
            present on my IDrive, link on the readme of this repository).
            zone_id 1 corresponds the farthest zone on the court from the camera's perspective, therefore the
            farthest point travelled by the ball will be the one having the minimum y-coordinate value (because
            this zone is closer to the top part of the frame), and thus it is stored, the opposite goes for the
            zone_id 0 which is closer to the bottom part of the frame, thus the farthest distance travelled by the 
            ball will have the maximum y-coordinate value.
            """
            if zone_id == 1:
                index_min = y_coordinate_list.index(min(y_coordinate_list))
                points_list.append([int(x_coordinate_list[index_min]), int(min(y_coordinate_list))])
            elif zone_id == 0:
                index_max = y_coordinate_list.index(max(y_coordinate_list))
                points_list.append([int(x_coordinate_list[index_max]), int(max(y_coordinate_list))])

    # Drawing the circles.
    cv2.circle(
        img=image,
        center=points_list[0],
        radius=RADIUS,
        color=COLOR,
        thickness=THICKNESS,
    )


# Defining a function that annotates the frames with the speed analytics of the ball.
def annotate_speeds(image):
    cv2.putText(
        img=image,
        text="Speed Stats",
        org=HEADING_TEXT_ORIGIN,
        fontFace=FONT_FACE,
        fontScale=FONT_SCALE,
        color=TEXT_COLOR,
        thickness=HEADING_FONT_THICKNESS,
    )
    cv2.putText(
        img=image,
        text="P1",
        org=P1_TEXT_ORIGIN,
        fontFace=FONT_FACE,
        fontScale=FONT_SCALE,
        color=TEXT_COLOR,
        thickness=STATS_FONT_THICKNESS,
    )
    cv2.putText(
        img=image,
        text="P1 Avg.",
        org=P1_AVG_TEXT_ORIGIN,
        fontFace=FONT_FACE,
        fontScale=FONT_SCALE,
        color=TEXT_COLOR,
        thickness=STATS_FONT_THICKNESS,
    )
    cv2.putText(
        img=image,
        text="P2",
        org=P2_TEXT_ORIGIN,
        fontFace=FONT_FACE,
        fontScale=FONT_SCALE,
        color=TEXT_COLOR,
        thickness=STATS_FONT_THICKNESS,
    )
    cv2.putText(
        img=image,
        text="P2 Avg.",
        org=P2_AVG_TEXT_ORIGIN,
        fontFace=FONT_FACE,
        fontScale=FONT_SCALE,
        color=TEXT_COLOR,
        thickness=STATS_FONT_THICKNESS,
    )

    zone_ids = list(zone_id_to_ball_speed.keys())
    for zone in zone_ids:
        speeds = zone_id_to_ball_speed[zone]
        if len(speeds) != 0:
            max_speed = max(speeds)
            average_speed = sum(speeds) / len(speeds)
            """
            After being hit by player one, who is close to zone 0, the ball will enter zone one, thus the speed
            of the ball while it is travelling to the zone 1 and before it is hit by the player two, is the speed
            with which player one hit the ball, same goes for the opposite.
            """
            if zone == 1:
                cv2.putText(
                    img=image,
                    text=f"P1       {max_speed:4.1f} km/h",
                    org=P1_TEXT_ORIGIN,
                    fontFace=FONT_FACE,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=STATS_FONT_THICKNESS,
                )
                cv2.putText(
                    img=image,
                    text=f"P1 Avg.  {average_speed:4.1f} km/h",
                    org=P1_AVG_TEXT_ORIGIN,
                    fontFace=FONT_FACE,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=STATS_FONT_THICKNESS,
                )
            else:
                cv2.putText(
                    img=image,
                    text=f"P2       {max_speed:4.1f} km/h",
                    org=P2_TEXT_ORIGIN,
                    fontFace=FONT_FACE,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=STATS_FONT_THICKNESS,
                )
                cv2.putText(
                    img=image,
                    text=f"P2 Avg.  {average_speed:4.1f} km/h",
                    org=P2_AVG_TEXT_ORIGIN,
                    fontFace=FONT_FACE,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=STATS_FONT_THICKNESS,
                )

        else:
            pass


# Defining a function that carries out almost all frame processing.
def frame_processor(frame, ball_detection_model_inference, player_detection_model_inference):
    # Converting detections from yolo format to supervision format.
    ball_detections = sv.Detections.from_ultralytics(ball_detection_model_inference)
    player_detections = sv.Detections.from_ultralytics(player_detection_model_inference)
    # Updating the trackers with the detections from the models.
    ball_detections = BALL_TRACKER.update_with_detections(ball_detections)
    player_detections = PLAYER_TRACKER.update_with_detections(player_detections)

    for (idx, zone), zone_annotator, round_box_annotator, label_annotator in zip(enumerate(POLYGON_ZONES),
                                                                                 POLYGON_ZONE_ANNOTATORS,
                                                                                 ROUND_BOX_ANNOTATORS,
                                                                                 LABEL_ANNOTATORS):

        # Filtering the detections from models to keep only those present in the defined zones.
        player_detections_mask = zone.trigger(player_detections)
        player_detections_filtered = player_detections[player_detections_mask]
        # Zone 6 and 7 are outside the court, and are the zones where the players can go.
        if idx not in [6, 7]:
            ball_detections_mask = zone.trigger(ball_detections)
            ball_detections_filtered = ball_detections[ball_detections_mask]

        # Getting the xy-coordinates of the pixel values travelled by the ball and transforming them.
        points = ball_detections_filtered.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        real_view_transformed_points = transform_real_points(points)
        radar_view_transformed_points = transform_radar_points(points)

        # Assigning names to each player using the tracker_ids of the players.
        labels = []
        for _ in player_detections_filtered.tracker_id:
            if idx in [1, 6]:
                labels.append("P1: Sonego")
            else:
                labels.append("P2: Djokovic")

        for tracker_id, [_, y1], [x2, y2] in zip(ball_detections_filtered.tracker_id, real_view_transformed_points,
                                                 radar_view_transformed_points):
            zone_coordinate_list = []

            """
            This loop handles the coordinates to be appended to the tracker_id_to_coordinates defaultdict for drawing 
            points on the radar court image.
            """
            if tracker_id is not None:
                if idx in [0, 1, 2, 3]:
                    zone_coordinate_list.append(idx)
                    zone_coordinate_list.append([x2, y2])
                    tracker_id_to_coordinates[tracker_id].append(zone_coordinate_list)

            """
            Only the y-coordinate is used for speed estimation, this is because in real life the ball travels much
            more y-axis distance as compared to the x-axis distance.
            """
            coordinates[tracker_id].append(y1)
            if len(coordinates[tracker_id]) > FPS / 3:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = coordinate_end - coordinate_start
                time = len(coordinates[tracker_id]) / FPS
                speed = abs((distance / time) * 3.6)
                zone_id_to_ball_speed[idx].append(speed)

        """
        This loop handles the drawing of points on the radar image. Only when a tracker_id is lost, it's stored
        coordinates are used to draw the points, this is because as long as a tracker_id exists in the detections
        it can move further thus shadowing the farthest point travelled by the ball.
        """
        tracker_ids = list(tracker_id_to_coordinates.keys())
        if len(tracker_ids) != 0:
            for tracker_id in tracker_ids:
                if tracker_id not in ball_detections.tracker_id:
                    draw_points(tracker_id, RADAR_COURT_IMAGE)
                    del tracker_id_to_coordinates[tracker_id]
                else:
                    pass

        # Annotating the frame with various annotations.
        round_box_annotator.annotate(
            scene=frame,
            detections=player_detections_filtered,
        )
        round_box_annotator.annotate(
            scene=frame,
            detections=ball_detections_filtered,
        )
        label_annotator.annotate(
            scene=frame,
            detections=player_detections_filtered,
            labels=labels,
        )
        zone_annotator.annotate(scene=frame)

        # Making an empty mask image to attach the radar court image to the actual frames (watch the output video).
        mask = np.zeros((1080, 1920 + 452, 3), dtype=np.uint8)
        mask[:1080, :1920, :3] = frame
        mask[:563, 1970:1970 + 352, :3] = RADAR_COURT_IMAGE

        # Annotating the mask image with the speed stats.
        annotate_speeds(mask)

    # Appending the mask to the output video.
    OUTPUT_VIDEO_WRITER.write(mask)


# Main loop.
while INPUT_VIDEO_READER.isOpened():
    status, frame = INPUT_VIDEO_READER.read()
    if not status:
        print("Out of frames!")
        break
    else:
        # Getting inferences from models and passing to the frame_processor function.
        ball_detection_model_inference = BALL_DETECTION_MODEL.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False,
        )[0]
        player_detection_model_inference = PLAYER_DETECTION_MODEL.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False,
        )[0]
        frame_processor(
            frame=frame,
            ball_detection_model_inference=ball_detection_model_inference,
            player_detection_model_inference=player_detection_model_inference,
        )

        # Saving a separate copy of the annotated radar image (not necessary).
        cv2.imwrite("annotated_radar_view_court_image.jpg", RADAR_COURT_IMAGE)
