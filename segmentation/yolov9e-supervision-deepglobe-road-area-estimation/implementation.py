# imports
from ultralytics import YOLO
import supervision as sv
import cv2
import os

# Initializing parameters for adding text to processed images using cv2.putText() function
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255, 255, 255)
TEXT_SCALE = 1
TEXT_THICKNESS = 4
ROAD_COUNT_TEXT_ORIGIN = (75, 50)
AREA_RATIO_TEXT_ORIGIN = (75, 80)
TOTAL_PIXELS_TEXT_ORIGIN = (75, 110)

# Initializing parameters for configuring and saving the output video using cv2.VideoWriter() class
OUTPUT_VIDEO_PATH = "/kaggle/working/yolov9e-supervision-road-area-ration-estimation.avi"
FOUR_CC = cv2.VideoWriter.fourcc(*"mp4v")
FPS = 1
IMAGE_SIZE = (1024, 1024)

# Instantiating a YOLOv9e model object using my own trained model (trained on the deepglobe road extraction dataset)
MODEL = YOLO("/kaggle/input/yolov9e-deepglobe-road-segmentation/pytorch/default/1/last.pt")

# Initializing a VideoWriter object for processing and writing the output video
VIDEO_WRITER = cv2.VideoWriter(
    filename=OUTPUT_VIDEO_PATH,
    fourcc=FOUR_CC,
    fps=FPS,
    frameSize=IMAGE_SIZE,
)

# Initializning a MaskAnnotator() object for annotating the images with the segmentation masks.
MASK_ANNOTATOR = sv.MaskAnnotator()

# Input images directory
IMAGE_DIR = "/kaggle/input/deepglobe-road-extraction-dataset/test"

# Looping through the images
for image_name in os.listdir(IMAGE_DIR):

    try:
        IMAGE_PATH = os.path.join(IMAGE_DIR, image_name)  # getting the image paths
        IMAGE = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)  # reading the images into memory

        inference = MODEL.predict(IMAGE_PATH, imgsz=IMAGE_SIZE, conf=0.8)[0]  # carrying inference with my model
        segmentations = sv.Detections.from_ultralytics(inference)  # converting inference to supervision format
        numpy_segmentations = inference.masks.data.cpu().numpy()  # converting inference to numpy arrays

        '''
        calculating the area occupied by segmentations in the masks, the masks are binary, the segmentations have 1 
        pixel value and the rest of the mask has 0 pixel value, therefore we can use the cv2.countNonZero() function to 
        get the area occupied by the segmentations in pixels. later we estimate the area ratio of the roads to the 
        complete image.
        '''
        non_zero_pixels = cv2.countNonZero(numpy_segmentations)
        image_area = IMAGE.shape[0] * IMAGE.shape[1]
        area_ratio = ((non_zero_pixels / image_area) * 100)

        MASK_ANNOTATOR.annotate(detections=segmentations, scene=IMAGE)  # annotating the images with segmentations

        # annotating the images with text representing the road counts, road to image ratio, and road pixel area.
        cv2.putText(
            IMAGE,
            f"No. of Road(s) detected: {len(numpy_segmentations)}",
            ROAD_COUNT_TEXT_ORIGIN,
            FONT_STYLE,
            TEXT_SCALE,
            FONT_COLOR,
            TEXT_THICKNESS,
        )
        cv2.putText(
            IMAGE,
            f"Road to Image ratio: {area_ratio:4.3f}%",
            AREA_RATIO_TEXT_ORIGIN,
            FONT_STYLE,
            TEXT_SCALE,
            FONT_COLOR,
            TEXT_THICKNESS,
        )
        cv2.putText(
            IMAGE,
            f"Pixel Area of Road(s): {non_zero_pixels} pixels",
            TOTAL_PIXELS_TEXT_ORIGIN,
            FONT_STYLE,
            TEXT_SCALE,
            FONT_COLOR,
            TEXT_THICKNESS,
        )
        VIDEO_WRITER.write(IMAGE)  # writing the output video

    except AttributeError:  # AttributeError is thrown if the model doesn't detect any roads.
        pass
