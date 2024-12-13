# imports
import cv2  # importing open-cv library for image and video processing.
import numpy as np  # importing numpy for the utilization of ndarrays.
import supervision as sv  # importing supervision for handling and using the models' inference + for annotations.
# importing gray2rgb for converting an image to rgb while visually keeping it in grayscale.
from skimage.color import gray2rgb
from ultralytics import YOLO  # importing the ultralytics YOLO class for loading and using the trained YOLOv8 models.

# specifying various paths for proper functioning of the code.
INPUT_VIDEO_PATH = "/kaggle/input/testing-video/clipped_video.mp4"
OUTPUT_COURT_VIDEO_PATH = "/kaggle/working/processed_video.avi"
OUTPUT_RADAR_VIDEO_PATH = "/kaggle/working/processed_court.avi"
RADAR_IMAGE_PATH = "/kaggle/input/court-image/court.jpg"
BALL_DETECTION_MODEL_PATH = """
    /kaggle/input/yolov8x_volleyball_analysis_models/pytorch/default/1/ball_detection_model.pt
"""
PLAYERS_DETECTION_MODEL_PATH = """
    /kaggle/input/yolov8x_volleyball_analysis_models/pytorch/default/1/players_referee_detection_model.pt
"""
KEY_POINTS_REGRESSION_MODEL_PATH = """
    /kaggle/input/yolov8x_volleyball_analysis_models/pytorch/default/1/key_points_regression_model.pt
"""

# loading the radar volleyball court image into memory and fetching its attributes.
RADAR_IMAGE = cv2.imread(RADAR_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
RADAR_IMAGE_HEIGHT, RADAR_IMAGE_WIDTH, _ = RADAR_IMAGE.shape

# loading the trained YOLOv8x models into memory.
BALL_DETECTION_MODEL = YOLO(BALL_DETECTION_MODEL_PATH)  # trained from scratch (yolov8x.yaml).
PLAYERS_DETECTION_MODEL = YOLO(PLAYERS_DETECTION_MODEL_PATH)  # trained using pretrained weights (yolov8x.pt).
KEY_POINTS_REGRESSION_MODEL = YOLO(KEY_POINTS_REGRESSION_MODEL_PATH)  # trained from scratch (yolov8x-pose.yaml).

# loading the input video into memory.
INPUT_VIDEO = cv2.VideoCapture(INPUT_VIDEO_PATH)

# fetching the attributes of input video frames for later use.
INPUT_VIDEO_FRAME_WIDTH, INPUT_VIDEO_FRAME_HEIGHT, INPUT_VIDEO_FPS = (
    int(INPUT_VIDEO.get(attribute))
    for attribute
    in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS,
)
)

# defining parameters for the cv2.VideoWriter class.
FOURCC = cv2.VideoWriter.fourcc(*"mp4v")  # four-character-code.
INPUT_VIDEO_FRAME_SIZE = (INPUT_VIDEO_FRAME_WIDTH, INPUT_VIDEO_FRAME_HEIGHT)

# initializing a cv2.VideoWriter object for merging the annotated court frames into an output video.
OUTPUT_COURT_VIDEO_WRITER = cv2.VideoWriter(
    filename=OUTPUT_COURT_VIDEO_PATH,
    fourcc=FOURCC,
    frameSize=INPUT_VIDEO_FRAME_SIZE,
    fps=INPUT_VIDEO_FPS,
)

# initializing a cv2.VideoWriter object for merging the annotated radar images into an output video.
OUTPUT_RADAR_VIDEO_WRITER = cv2.VideoWriter(
    filename=OUTPUT_RADAR_VIDEO_PATH,
    fourcc=FOURCC,
    frameSize=(RADAR_IMAGE_WIDTH, RADAR_IMAGE_HEIGHT),
    fps=INPUT_VIDEO_FPS,
)

"""
Defining the selected key points present on the radar image being used. These key points will be used to transform the
perspective of the court in the input video frames to the radar court image via homography. This is necessary for
accurately estimating the locations of the players on the court in real-life.
Later we use the key points regressed by the key points regression model as the source matrix and the list below as
the target matrix for homography calculation.
"""
RADAR_IMAGE_PITCH_KEY_POINTS = [
    (739, 229), (1218, 229), (1218, 866), (739, 866)
]

# initializing a supervision ColorPalette object defining the colors for annotating team one, team two, and referee.
COLOR_PALETTE = sv.ColorPalette.from_hex(["#FFFFFF", "#ff0000", "#00ff7f"])

# initializing various supervision annotators for annotating the input video frames with model inferences.
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.CLASS,
    thickness=2,
)
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(
    color=sv.Color.WHITE,
    thickness=1,
)
VERTEX_ANNOTATOR = sv.VertexAnnotator(
    color=sv.Color.BLACK,
    radius=2,
)

"""
Defining a class for calculating the homography between the input video court and the radar court image using the
key points regressed by the key points regression model as the source matrix and the radar court key points defined
above as the target matrix. 
This class also provides the functionality to transform any cartesian coordinates from the input video court to the 
radar court image, which will be used to estimate the position of the players on the radar court image, using the 
points predicted by the player detection model on the input video court.
"""
class ViewTransformer:
    def __init__(self, source_matrix, target_matrix):
        source_matrix = source_matrix.astype(np.float32)
        target_matrix = target_matrix.astype(np.float32)
        self.transformed_matrix, _ = cv2.findHomography(source_matrix, target_matrix)

    def transform_points(self, points):
        if points.size == 0:
            return points
        points = points.reshape(-1, 1, 2).astype(np.float32)  # 3D matrix is required for perspective transformation.
        points = cv2.perspectiveTransform(points, self.transformed_matrix)
        return points.reshape(-1, 2).astype(np.float32)  # removing the additional axis added before.


# defining a function to draw points on the radar court image based upon the transformed cartesian coordinates.
def draw_points(points, image, color):
    x, y = int(points[0]), int(points[1])
    annotated_image = cv2.circle(
        img=image,
        center=(x, y),
        radius=8,
        color=color,
        thickness=-1,
    )
    return annotated_image


# defining the main function that deals with all inference handling, data annotation, and video writing.
def frame_processor(frame, ball_detection_inference, players_detection_inference, key_points_inference):
    try:
        # converting inferences from YOLO format to supervision format.
        ball_detections = sv.Detections.from_ultralytics(ball_detection_inference)
        players_detections = sv.Detections.from_ultralytics(players_detection_inference)
        key_points_regressions = sv.KeyPoints.from_ultralytics(key_points_inference)

        # fetching the input video frame court's regressed key points and the radar image court key points.
        frame_ref_points = key_points_regressions.xy[0]
        frame_ref_key_points = sv.KeyPoints(xy=frame_ref_points[np.newaxis, ...])
        pitch_ref_points = np.array(RADAR_IMAGE_PITCH_KEY_POINTS)

        # calculating homography.
        view_transformer = ViewTransformer(
            source_matrix=frame_ref_points,
            target_matrix=pitch_ref_points
        )

        # filtering player detections.
        team_one_detections = players_detections[players_detections.class_id == 1]
        team_two_detections = players_detections[players_detections.class_id == 2]

        # fetching predicted players' cartesian coordinates.
        frame_team_one_coordinates = team_one_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        frame_team_two_coordinates = team_two_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        # converting the players' cartesian coordinates from input video frame courts to the radar court image.
        pitch_team_one_coordinates = view_transformer.transform_points(frame_team_one_coordinates)
        pitch_team_two_coordinates = view_transformer.transform_points(frame_team_two_coordinates)

        court = RADAR_IMAGE.copy()

        # drawing points on the radar court image.
        for points in pitch_team_one_coordinates:
            court = draw_points(points, court, (0, 0, 255))
        for points in pitch_team_two_coordinates:
            court = draw_points(points, court, (127, 255, 0))

        # annotating the input video frames with inference data.
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(frame, ball_detections)
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(frame, players_detections)
        annotated_frame = VERTEX_ANNOTATOR.annotate(frame, frame_ref_key_points)
        annotated_frame = cv2.resize(annotated_frame, (1920, 1080))

        # writing the output videos
        OUTPUT_RADAR_VIDEO_WRITER.write(court)
        OUTPUT_COURT_VIDEO_WRITER.write(annotated_frame)
    except:
        pass


# main loop
while INPUT_VIDEO.isOpened():
    status, original_frame = INPUT_VIDEO.read()
    if not status:
        print("Out of Frames!")
        break
    else:
        resized_original_frame = cv2.resize(original_frame, (640, 640))
        ball_detection_inference = BALL_DETECTION_MODEL.predict(
            resized_original_frame,
            save=False,
            verbose=False,
        )[0]
        players_detection_inference = PLAYERS_DETECTION_MODEL.predict(
            resized_original_frame,
            save=False,
            verbose=False,
        )[0]

        # the yolov8x-pose key point regression model was trained on grayscale images for better accuracy.
        grayscale_resized_frame = cv2.cvtColor(
            resized_original_frame,
            cv2.COLOR_RGB2GRAY
        )
        grayscale_resized_frame = gray2rgb(grayscale_resized_frame)
        key_points_inference = KEY_POINTS_REGRESSION_MODEL.predict(
            grayscale_resized_frame,
            conf=0.1,
            save=False,
            verbose=False,
        )[0]
        frame_processor(resized_original_frame, ball_detection_inference, players_detection_inference,
                        key_points_inference)
