'''
This project aims to accurately estimate the damage done to constructions.
Using my own yolov8x-seg model trained on the Massachussets Buildings Dataset, the total number of
buildings before and after destruction are detected and are then used to estimate the damage done in terms of the number
of buildings lost and the decrease in the area of occupancy.
'''

# imports
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Defining SAHI (Slicing Aided Hyper Inference) hyperparameters.
CONFIDENCE_THRESHOLD = 0.2  # The minimum confidence score for acceptable detections made by the model.
DEVICE = "cpu"  # The device used to carry inference (cpu or cuda:0).
SLICE_HEIGHT, SLICE_WIDTH = 250, 250  # The slice height and width used to divide the image into small batches.

# Defining the model path using my own trained yolov8x-seg model
MODEL_PATH = "/kaggle/input/yolov8x-sahi-construction-damage-estimation/pytorch/default/1/best.pt"

# Specifying paths to the images and output folders
BEFORE_DESTRUCTION_IMAGE_PATH = "/kaggle/input/before_destruction.png"
AFTER_DESTRUCTION_IMAGE_PATH = "/kaggle/input/after_destruction.png"
BEFORE_DESTRUCTION_ANNOTATED_IMAGE_EXPORT_DIRECTORY = "/kaggle/working/before"
AFTER_DESTRUCTION_ANNOTATED_IMAGE_EXPORT_DIRECTORY = "/kaggle/working/after"

IMAGE = cv2.imread(
    BEFORE_DESTRUCTION_IMAGE_PATH,  # Reading an image to get it's width and height.
    cv2.IMREAD_UNCHANGED
)
HEIGHT, WIDTH, _ = IMAGE.shape
PIXEL_AREA_OF_IMAGE = WIDTH * HEIGHT

MODEL = AutoDetectionModel.from_pretrained(  # Initializing an AutoDetectionModel object.
    model_type="yolov8",
    model_path=MODEL_PATH,
    device=DEVICE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
)

#  Getting the predictions on the images before and after the destruction
results = [
    get_sliced_prediction(
        image=BEFORE_DESTRUCTION_IMAGE_PATH,
        detection_model=MODEL,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
    ),
    get_sliced_prediction(
        image=AFTER_DESTRUCTION_IMAGE_PATH,
        detection_model=MODEL,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
    )
]

# Saving the processed and annotated images.
results[0].export_visuals(
    export_dir=BEFORE_DESTRUCTION_ANNOTATED_IMAGE_EXPORT_DIRECTORY,
    hide_conf=True,
    hide_labels=True,
)
results[1].export_visuals(
    export_dir=AFTER_DESTRUCTION_ANNOTATED_IMAGE_EXPORT_DIRECTORY,
    hide_conf=True,
    hide_labels=True,
)

# Converting the results to COCO predictions for easy handling of data.
before_destruction_object_prediction_list = results[0].to_coco_predictions()
after_destruction_object_prediction_list = results[1].to_coco_predictions()

# Getting the total number of detected buildings before and after destruction.
buildings_before_destruction = len(before_destruction_object_prediction_list)
buildings_after_destruction = len(after_destruction_object_prediction_list)

# Calculating the total area occupied before and after destruction.
occupied_area_before_destruction = 0
occupied_area_after_destruction = 0

for building in before_destruction_object_prediction_list:
    occupied_area_before_destruction += building['area']

for building in after_destruction_object_prediction_list:
    occupied_area_after_destruction += building['area']

# Estimating the damage.
buildings_lost = buildings_before_destruction - buildings_after_destruction
previous_ratio_of_occupancy = (buildings_before_destruction / PIXEL_AREA_OF_IMAGE) * 100
current_ratio_of_occupancy = (buildings_after_destruction / PIXEL_AREA_OF_IMAGE) * 100

# Printing the output.
print(f"Number of buildings detected before destruction: {buildings_before_destruction}")
print(f"Number of buildings detected after destruction: {buildings_after_destruction}")
print(f"Occupied area of buildings before destruction (in pixels): {occupied_area_before_destruction}")
print(f"Occupied area of buildings after destruction (in pixels): {occupied_area_after_destruction}")
print(f"Ratio of occupancy before destruction: {previous_ratio_of_occupancy:3.3f}")
print(f"Ratio of occupancy after destruction: {current_ratio_of_occupancy:3.3f}")
print(f"Buildings lost due to destruction: {buildings_lost}")
