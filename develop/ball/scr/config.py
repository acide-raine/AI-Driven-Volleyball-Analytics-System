import cv2

#PATHS
VIDEO_PTAH = "/Users/pasha/Data_stat/short_Severyanka_korabelka.mp4"

MODEL_PATH = "/Users/pasha/Data_stat/develop/ball/model/VballNetV1_seq9_grayscale_148_h288_w512.onnx"

#VIDEO_PARAMS
cap = cv2.VideoCapture(VIDEO_PTAH)

FPS = cap.get(cv2.CAP_PROP_FPS)
VIDEO_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
VIDEO_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#MODEL_PARAMS
INPUT_W = 512
INPUT_H = 288
HEATMAP_TRASH = 0.5
SEQ_LEN = 9