import os
from tkinter import filedialog as fd

import customtkinter as ctk
import cv2
from ultralytics import YOLO

SOURCE = None
OUTPUT = None
CUSTOM_BACKGROUND = None
confidence = 0.5

IMAGE_SUFFIXES = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.webp']
VIDEO_SUFFIXES = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm']
FOLDERS = []

directory = os.getcwd()
if '\\' in directory:
    directory.replace('\\', '/')

for folder in os.listdir(directory):
    FOLDERS.append(folder)

if 'temp' not in FOLDERS:
    os.mkdir(f'{directory}/temp')

MODEL_PATH = f"{directory}/best.pt"
MODEL = YOLO(MODEL_PATH)


def video_background_remover(SOURCE):
    for index, src in enumerate(SOURCE):
        cap = cv2.VideoCapture(src)
        i = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(f'{directory}/temp/frame{i}{index}.png', frame)
            img = cv2.imread(f'{directory}/temp/frame{i}{index}.png')
            height, width, _ = img.shape
            results = MODEL(source=f'{directory}/temp/frame{i}{index}.png', conf=confidence)

            try:

                for x in range(0, len(results[0].masks.data)):
                    cv2.imwrite(
                        f"{directory}/temp/output{x}{index}.png",
                        (results[0].masks.data[x].numpy() * 255).astype("uint8"))

                for x in range(0, len(results[0].masks.data) - 1):
                    mask_one = cv2.imread(
                        f"{directory}/temp/output{x}{index}.png")
                    mask_two = cv2.imread(
                        f"{directory}/temp/output{x + 1}{index}.png")
                    combined = cv2.bitwise_or(mask_one, mask_two)
                    cv2.imwrite(
                        f"{directory}/temp/output{x + 1}{index}.png",
                        combined)

                real = cv2.imread(f"{directory}/temp/frame{i}{index}.png")
                mask = cv2.imread(f"{directory}/temp/output{len(results[0].masks.data) - 1}{index}.png")
                mask = cv2.resize(mask, (width, height))
                if CUSTOM_BACKGROUND is None:
                    mask = cv2.bitwise_not(mask)
                    combined = cv2.bitwise_or(real, mask)
                    cv2.imwrite(
                        f"{directory}/temp/frame{i}{index}.png",
                        combined)
                else:
                    black_combined = cv2.bitwise_and(mask, real)
                    mask = cv2.bitwise_not(mask)
                    custom_bg = cv2.imread(CUSTOM_BACKGROUND)
                    custom_bg = cv2.resize(custom_bg, (width, height))
                    masked_bg = cv2.bitwise_and(custom_bg, mask)
                    masked_bg = cv2.bitwise_or(masked_bg, black_combined)
                    cv2.imwrite(
                        f"{directory}/temp/frame{i}{index}.png",
                        masked_bg)

                for files in os.listdir(f'{directory}/temp'):
                    if 'output' in files:
                        os.remove(os.path.join(f'{directory}/temp', files))

                i += 1
            except:
                i += 1

            codec = cv2.VideoWriter.fourcc(*'mp4v')
            video = cv2.VideoWriter(f'{OUTPUT}/processed_video.mp4', codec, 30.0, (width, height))

            for x in range(0, i):
                img = cv2.imread(f'{directory}/temp/frame{x}{index}.png')
                video.write(img)

        for files in os.listdir(f'{directory}/temp'):
            if 'frame' in files:
                os.remove(os.path.join(f'{directory}/temp', files))

        cv2.destroyAllWindows()
        cap.release()


def image_background_remover(SOURCE):
    for index, src in enumerate(SOURCE):
        try:
            results = MODEL.predict(source=src, conf=confidence)
            for mask in range(0, len(results[0].masks.data)):
                cv2.imwrite(f'{OUTPUT}/mask{index}.jpeg',
                            (results[0].masks.data[mask].numpy() * 255).astype('uint8'))

            real = cv2.imread(src)
            height, width, _ = real.shape
            mask = cv2.imread(f'{OUTPUT}/mask{index}.jpeg')
            mask = cv2.resize(mask, (width, height))

            if CUSTOM_BACKGROUND is None:
                mask = cv2.bitwise_not(mask)
            else:
                black_combined = cv2.bitwise_and(mask, real)
                mask = cv2.bitwise_not(mask)
                custom_bg = cv2.imread(CUSTOM_BACKGROUND)
                custom_bg = cv2.resize(custom_bg, (width, height))
                masked_bg = cv2.bitwise_and(custom_bg, mask)
                masked_bg = cv2.bitwise_or(masked_bg, black_combined)
                cv2.imwrite(f'{OUTPUT}/masked_background{index}.jpeg', masked_bg)

            cv2.imwrite(f'{OUTPUT}/background_removed{index}.jpeg', cv2.bitwise_or(mask, real))
        except:
            if index == len(SOURCE) - 1:
                STATUS_LABEL.configure(text='No Detections')
                START_BUTTON.configure(state='normal')
            else:
                pass


##################################################### GUI CODE #########################################################


def input_button_function(event):
    global SOURCE
    SOURCE = fd.askopenfilenames()
    if SOURCE is not None:
        STATUS_LABEL.configure(text='Input Selected')


def output_button_function(event):
    global OUTPUT
    OUTPUT = fd.askdirectory()
    if '\\' in OUTPUT:
        OUTPUT.replace('\\', '/')
    if SOURCE is not None and OUTPUT is not None:
        START_BUTTON.configure(state='normal')
    if OUTPUT is not None:
        STATUS_LABEL.configure(text='Output Folder Selected')


def custom_background_button_function(event):
    global CUSTOM_BACKGROUND
    CUSTOM_BACKGROUND = fd.askopenfilename()
    if '\\' in CUSTOM_BACKGROUND:
        OUTPUT.replace('\\', '/')
    if CUSTOM_BACKGROUND is not None:
        STATUS_LABEL.configure(text='Custom Background Selected')


def start_button_func(event):
    if SOURCE is not None and OUTPUT is not None:
        START_BUTTON.configure(state='disabled')
        for path in SOURCE:
            for suffix in IMAGE_SUFFIXES:
                if suffix in path:
                    image_background_remover(SOURCE)
                    break
            else:
                video_background_remover(SOURCE)
        STATUS_LABEL.configure(text='Finished')
        START_BUTTON.configure(state='normal')
    elif SOURCE is None and OUTPUT is not None:
        STATUS_LABEL.configure(text='Choose Input Image(s)/video(s)')
    elif OUTPUT is None and SOURCE is not None:
        STATUS_LABEL.configure(text='Choose an Output Folder')
    else:
        STATUS_LABEL.configure(text='Choose Input Image(s)/Video(s) and Output Folder')


def slider_func(value):
    global confidence
    confidence = f'{value:.1f}'
    confidence = float(confidence)
    CONFIDENCE_LABEL.configure(text=f'conf={confidence}')


ctk.set_appearance_mode('dark')
ROOT = ctk.CTk()
ROOT.title('AI Background Remover')
ROOT.geometry('500x400')
ROOT.resizable(False, False)

INPUT_BUTTON = ctk.CTkButton(
    master=ROOT,
    width=150,
    height=50,
    text='Input Images / Videos'
)
INPUT_BUTTON.place(x=75, y=50)
INPUT_BUTTON.bind('<Button-1>', input_button_function)

OUTPUT_BUTTON = ctk.CTkButton(
    master=ROOT,
    width=150,
    height=50,
    text='Output Folder'
)
OUTPUT_BUTTON.place(x=275, y=50)
OUTPUT_BUTTON.bind('<Button-1>', output_button_function)

CUSTOM_BACKGROUND_BUTTON = ctk.CTkButton(
    master=ROOT,
    width=150,
    height=50,
    text='Custom Background\n(Default White)'
)
CUSTOM_BACKGROUND_BUTTON.place(x=75, y=150)
CUSTOM_BACKGROUND_BUTTON.bind('<Button-1>', custom_background_button_function)

START_BUTTON = ctk.CTkButton(
    master=ROOT,
    width=150,
    height=50,
    text='Start',
    fg_color='purple',
    state='disabled'
)
START_BUTTON.place(x=275, y=150)
START_BUTTON.bind('<Button-1>', start_button_func)

CONFIDENCE_SLIDER = ctk.CTkSlider(
    master=ROOT,
    width=350,
    height=20,
    from_=0,
    to=1,
    number_of_steps=10,
    command=slider_func
)
CONFIDENCE_SLIDER.place(x=75, y=250)

CONFIDENCE_LABEL = ctk.CTkLabel(
    master=ROOT,
    width=40,
    height=40,
    text='conf=0.5',
    font=('Nunito', 12, 'bold')
)
CONFIDENCE_LABEL.place(x=370, y=270)

STATUS_LABEL = ctk.CTkLabel(master=ROOT, width=200, height=40, text='', font=('Nunito', 12, 'bold', 'italic'))
STATUS_LABEL.place(x=20, y=350)

ROOT.mainloop()
