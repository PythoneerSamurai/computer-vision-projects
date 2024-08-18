# imports
from ultralytics import YOLO
import supervision as sv
import cv2
import os

# Initializing parameters for adding text to processed images using cv2.putText() function
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255, 255, 255)
TEXT_SCALE = 0.75
TEXT_THICKNESS = 2
CANCER_COUNT_TEXT_ORIGIN = (15, 50)
CENTER_OF_MASS_TEXT_ORIGIN = (15, 80)
AREA_RATIO_TEXT_ORIGIN = (15, 110)
TOTAL_PIXELS_TEXT_ORIGIN = (15, 140)

# Initializing parameters for configuring and saving the output video using cv2.VideoWriter() class
OUTPUT_VIDEO_PATH = "/kaggle/working/yolov8x-supervision-skin-cancer-analysis.avi"
FOUR_CC = cv2.VideoWriter.fourcc(*"mp4v")
FPS = 1
IMAGE_SIZE = (600, 450)

# Instantiating a YOLOv8x model object using my own trained model (trained on the HAM10000 skin cancer dataset)
MODEL = YOLO("/kaggle/input/yolov8x-ham10000-segmentation/pytorch/default/1/ham10000.pt")

# Initializing a VideoWriter object for processing and writing the output video
VIDEO_WRITER = cv2.VideoWriter(
    filename=OUTPUT_VIDEO_PATH,
    fourcc=FOUR_CC,
    fps=FPS,
    frameSize=IMAGE_SIZE,
)

# Initializning a MaskAnnotator() object for annotating the images with the segmentation masks.
MASK_ANNOTATOR = sv.MaskAnnotator(color=sv.Color.YELLOW)
# Input images directory
IMAGE_DIR = "/kaggle/input/ham1000-segmentation-and-classification/images"

# Looping through the images
for index, image_name in enumerate(os.listdir(IMAGE_DIR)):
    # Dataset has 10000 images, carrying inference on 200 images only to keep the output video size small.
    if index == 200:
        break
    else:
        try:
            IMAGE_PATH = os.path.join(IMAGE_DIR, image_name)  # getting the image paths
            IMAGE = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)  # reading the images into memory

            inference = MODEL.predict(
                IMAGE_PATH,
                imgsz=IMAGE_SIZE,
                conf=0.8,
                save=False,
                verbose=False,
            )[0]  # carrying inference with my model
            segmentations = sv.Detections.from_ultralytics(inference)  # converting inference to supervision format
            numpy_segmentations = inference.masks.data.cpu().numpy()  # converting detected masks to numpy arrays
            center_of_mass = segmentations.get_anchors_coordinates(sv.Position.CENTER_OF_MASS)[0]  # getting the x, y coordinates of the center of mass of the detected cancers.

            '''
            calculating the area occupied by segmentations in the masks, the masks are binary, the segmentations have 1 
            pixel value and the rest of the mask has 0 pixel value, therefore we can use the cv2.countNonZero() function to 
            get the area occupied by the segmentations in pixels. later we estimate the area ratio of the cancer(s) to the 
            complete image.
            '''
            non_zero_pixels = cv2.countNonZero(numpy_segmentations[0])
            image_area = IMAGE.shape[0] * IMAGE.shape[1]
            area_ratio = ((non_zero_pixels / image_area) * 100)

            MASK_ANNOTATOR.annotate(detections=segmentations, scene=IMAGE)  # annotating the images with segmentations

            # annotating the images with text representing the cancer analysis.
            cv2.putText(
                IMAGE,
                f"No. of Cancer(s) detected: {len(numpy_segmentations)}",
                CANCER_COUNT_TEXT_ORIGIN,
                FONT_STYLE,
                TEXT_SCALE,
                FONT_COLOR,
                TEXT_THICKNESS,
            )

            cv2.putText(
                IMAGE,
                f"Center of Mass: {center_of_mass}",
                CENTER_OF_MASS_TEXT_ORIGIN,
                FONT_STYLE,
                TEXT_SCALE,
                FONT_COLOR,
                TEXT_THICKNESS,
            )
            cv2.putText(
                IMAGE,
                f"Cancer to Image ratio: {area_ratio:4.3f}%",
                AREA_RATIO_TEXT_ORIGIN,
                FONT_STYLE,
                TEXT_SCALE,
                FONT_COLOR,
                TEXT_THICKNESS,
            )
            cv2.putText(
                IMAGE,
                f"Pixel Area of Cancer(s): {non_zero_pixels} pixels",
                TOTAL_PIXELS_TEXT_ORIGIN,
                FONT_STYLE,
                TEXT_SCALE,
                FONT_COLOR,
                TEXT_THICKNESS,
            )
            VIDEO_WRITER.write(IMAGE)

        except AttributeError:
            pass
