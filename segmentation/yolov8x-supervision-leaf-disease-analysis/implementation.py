# imports
import cv2
import supervision as sv
from ultralytics import YOLO

# specifiying paths.
IMAGE_PATH = "/kaggle/input/leaf-disease-segmentation-dataset/data/data/images/00000.jpg"
MODEL_PATH = "/kaggle/input/leaf-disease-first-run/runs/segment/train/weights/best.pt"
OUTPUT_PATH = "/kaggle/working"

# instantiating a supervision MaskAnnotator object for annotating images with segmentation masks.
MASK_ANNOTATOR = sv.MaskAnnotator()

# instantiating a YOLOv8x model object using my own trained model (trained on the leaf disease segmentation dataset available on kaggle).
MODEL = YOLO(MODEL_PATH)

# reading the input image
IMAGE = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)

# carrying inference on the input image using my model.
inference = MODEL.predict(
    IMAGE_PATH,
    conf=0.6,
    save=False,
    verbose=False,
)[0]

# defining a function that takes an image and yolov8x inference (carried on the image) as inputs and processes the image.
def image_processor(image, inference):
    
    segmentations = sv.Detections.from_ultralytics(inference)  # converting inference to supervision format.
    numpy_segmentations = inference.masks.data.cpu().numpy()  # converting inference to an numpy ndarray.
            
    non_zero_pixels = cv2.countNonZero(numpy_segmentations[0])  # estimating the total pixel area occupied by the disease(s).
    image_area = image.shape[0] * image.shape[1]  # calculating the total image area.
    area_ratio = ((non_zero_pixels / image_area) * 100)  # estimating the area in ratio of disease(s) to the full image area.

    MASK_ANNOTATOR.annotate(detections=segmentations, scene=image)  # annotating the input image with segmentation masks.

    # printing information regarding the detected disease(s).
    print(f"Number of disease(s) detected: {len(numpy_segmentations)}")
    print(f"Pixel area of disease(s): {non_zero_pixels}")
    print(f"Disease(s) to Image ratio: {area_ratio:4.1f}")

    # writing the output image.
    cv2.imwrite(f"{OUTPUT_PATH}/processed_image.jpg", image)

# calling the function defined above.
image_processor(IMAGE, inference)
