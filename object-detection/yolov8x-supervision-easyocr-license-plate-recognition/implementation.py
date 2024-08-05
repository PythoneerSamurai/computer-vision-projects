# imports
import cv2
import easyocr
import supervision as sv
from ultralytics import YOLO

# initializing the YOLO model object using my yolov8x model.
MODEL = YOLO("/kaggle/input/yolov8-license-plate-detection/pytorch/overall-best-and-last-epoch-models/1/last.pt")

'''
initializing the VideoCapture object (to read the input video and iterate over it frame by frame) and the VideoWriter 
object (to write the output video).
width, height, and fps will be used later to increase the accuracy of detections and tracking.
'''

VIDEO_CAPTURE = cv2.VideoCapture("/kaggle/input/car-video/demo.mp4")
WIDTH, HEIGHT, FPS = (
    int(VIDEO_CAPTURE.get(prop)) for prop in
    (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
VIDEO_WRITER = cv2.VideoWriter(
    "/kaggle/working/output_video.avi",
    cv2.VideoWriter.fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT)
)

# initializing BoxCornerAnnotator and LabelAnnotator objects for annotating the detections.
BOX_CORNER_ANNOTATOR = sv.BoxCornerAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# initializing a ByteTrack object for tracking detections throughout the frames.
TRACKER = sv.ByteTrack(frame_rate=FPS)

# initializing a Reader object to read text in license plates. 
READER = easyocr.Reader(["en"], gpu=True)

# frame count will be used for calculating the time each vehicle spends in the video.
frame_count = 1

# dictionaries that assign values to tracker ids for ocr text annotation and time calculation.
TRACKER_ID_TO_TEXT = {}
TRACKER_ID_TO_FRAME_ID = {}
TRACKER_ID_TO_TIME = {}


# a function that is used to estimate the time each detection spends in the video.
def time_estimator(frame_index, detections):
    """ 
    Looping through tracker ids, if the tracker id is not in the specified dictionary, add it as a key and assign the 
    frame it was detected in twice in a list as values. The first index of the value list remains same throughout the 
    video because it is the entry frame of the detection, whereas the second index keeps getting incremented as long as 
    the detection is in the video. The time each detection spends in a video is calculated everytime the function is 
    called and labelled upon the detection. The formula for time calculation is very simple, just subtract the entry 
    frame from the current frame and divide it with the FPS of the video, this gives us accurate time in seconds.
    """
    for tracker_id in detections.tracker_id:
        if tracker_id not in TRACKER_ID_TO_FRAME_ID.keys():
            TRACKER_ID_TO_FRAME_ID[tracker_id] = [frame_index, frame_index]
        else:
            TRACKER_ID_TO_FRAME_ID[tracker_id][1] = frame_index

    # loop for time estimation
    for tracker_id in TRACKER_ID_TO_FRAME_ID.keys():
        entry_frame = TRACKER_ID_TO_FRAME_ID[tracker_id][0]
        exit_frame = TRACKER_ID_TO_FRAME_ID[tracker_id][1]
        time_spent_in_video = (exit_frame - entry_frame) / FPS
        TRACKER_ID_TO_TIME[tracker_id] = time_spent_in_video


# a function that processes individual frames.
def frame_processor(frame, frame_index, inference):
    # converting detections from YOLO format to supervision format.
    detections = sv.Detections.from_ultralytics(inference)
    # using the tracker to track detections with unique tracker ids.
    detections = TRACKER.update_with_detections(detections=detections)
    # annotating the frame with box corners placed around each detection.
    BOX_CORNER_ANNOTATOR.annotate(scene=frame, detections=detections)  
    
    """
    adding each tracker id to the specified dictionary, and assigning it an initial value of "Bad Recognition", 
    because the OCR hasn't run yet.
    """
    for tracker_id in detections.tracker_id:
        if tracker_id not in TRACKER_ID_TO_TEXT.keys():
            TRACKER_ID_TO_TEXT[tracker_id] = "Bad Recognition"
        else:
            pass

    # looping through each tracker id along with its bounding box coordinates.
    for bounding_box_coordinates, tracker_id in zip(detections.xyxy.tolist(), detections.tracker_id):
        """
        if the tracker id doesn't have a value of "Bad Recognition", it means that it has already been assigned an OCR 
        text value.
        """
        if TRACKER_ID_TO_TEXT[tracker_id] != "Bad Recognition":
            pass
        else:
            bounding_box_coordinates = [int(pixel_value) for pixel_value in bounding_box_coordinates]
            left, top, right, bottom = bounding_box_coordinates
            cropped_frame = frame[top:bottom, left:right]
            """
            converting the cropped frame to gray scale and inverse thresholding it seems to improve easyocr's 
            detections.
            """
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            """
            any pixel with intensity greater than 140 will be set to 0 (black), and lower that 140 will be set to 255 
            (white).
            """
            _, threshold_cropped_frame = cv2.threshold(cropped_frame, 140, 255, cv2.THRESH_BINARY_INV)
            try:
                text = READER.readtext(threshold_cropped_frame)[0][
                    1]  # using easyocr to get the text from license plates.
                text = text.upper()  # converting text to uppercase.
                """
                the license plates I'm recognizing have a length of 8 characters, so we'll only keep those texts with 
                length 8.
                """
                if len(text) != 8:
                    pass
                else:
                    """
                    there are two types of plates, one with a "-" at the specified index and the other with a hyphen at 
                    index 2.
                    """
                    if text[1] == "-":
                        '''
                        this specific type of license plate doesn't have a number as the first character, so we skip 
                        those OCR detections which have a number as the first character.
                        '''
                        if text[0].isdigit():
                            pass
                        else:
                            character_list = list(text)  # converting the text to a list
                            '''
                            this type of license plate has only numbers at index 2 to 5, so we convert the alphabets, if 
                            detected, to their corresponding numbers.
                            '''
                            substring_list = character_list[2:5]
                            for index, character in enumerate(substring_list):
                                if character in ["I", "L", "l"]:
                                    substring_list[index] = "1"
                                elif character == "S":
                                    substring_list[index] = "5"
                                elif character == "B":
                                    substring_list[index] = "8"
                                elif character == "C":
                                    substring_list[index] = "6"
                            character_list[2:5] = substring_list
                            reformated_text = ""
                            for character in character_list:
                                reformated_text += character
                            TRACKER_ID_TO_TEXT[tracker_id] = reformated_text
                    elif text[2] == "-" and text[5] == "-":  # another type of license plate in the video.
                        TRACKER_ID_TO_TEXT[tracker_id] = text
            except IndexError:
                pass

    time_estimator(frame_index, detections)

    labels = []
    for tracker_id in detections.tracker_id:
        labels.append(f"{TRACKER_ID_TO_TEXT[tracker_id]}, {TRACKER_ID_TO_TIME[tracker_id]:4.1f}s")

    LABEL_ANNOTATOR.annotate(scene=frame, detections=detections,
                             labels=labels)  # annotating each license plate with its text and time spend in the video.
    VIDEO_WRITER.write(frame)  # appending the frame to the output video.


# loop for iterating over the video frame by frame.
while VIDEO_CAPTURE.isOpened():
    success, frame = VIDEO_CAPTURE.read()
    if not success:  # means no more frames left in the video.
        break
    else:
        inference = MODEL(frame, imgsz=(WIDTH, HEIGHT))[0]  # using my model to carry inference on the frames.
        frame_processor(frame, frame_count, inference)
        frame_count += 1  # incrementing frame count for time calculation.
