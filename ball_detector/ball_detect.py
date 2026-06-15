import numpy as np
import cv2
import json

from ultralytics import YOLO

import time

model_path = "/content/YOLO_ball_detector.pt"
video_path = "/content/test_video.mp4"

def ball_detector(model_path, video_path):

    '''Функция детекции мяча

    Принимает:
    model_path – путь к модели
    video_path – путь к видео, на котором нужной найти мяч

    Возвращает:
    ball_cords – dict вида {[frame_id] = {"x": float(x),
                                          "y": float(y),
                                          "w": float(w),
                                          "h": float(h),
                                          "scores": float(scores[best_box])} – насколько модель уверена,
                                                                               что найденный объект это мяч.
    '''


    skiped_frame = 0
    valid_frame = 0
    all_frame = 0

    model = YOLO(model_path)

    ball_cords = {}

    start = time.time()

    results = model.predict(source = video_path,
                            stream = True)

    for frame_id, result in enumerate(results):
        all_frame += 1

        if len(result.boxes) == 0:
          skiped_frame += 1
          continue

        boxes = result.boxes.xywh.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        best_box = np.argmax(scores)

        x,y,w,h = boxes[best_box]

        end = time.time()

        ball_cords[frame_id] = {"x": float(x),
                                "y": float(y),
                                "w": float(w),
                                "h": float(h),
                                "scores": float(scores[best_box])}
        valid_frame += 1

    time_ = end - start
    timing = (all_frame / 30)

    print(f"Всего фреймов:{all_frame}")
    print(f"Успешно обработанных фреймов:{valid_frame}")
    print(f"Пропущено фреймов:{skiped_frame}")
    print(f"На обраотку {timing} секунду видео затрачено {time_} секунд")

    return ball_cords

ball_cords = ball_detector(model_path, video_path)

with open("ball_cords.json", "w") as f:
    json.dump(ball_cords, f)

print(f"Фактическая длина ball_cords: {len(ball_cords)}")