import cv2
import numpy as np
import ultralytics
import time

from video_pr.video_preprocess import sliding_window


def gen_maker(video_path, window_size, overlap):
  '''
  Функция для рестарта генератора с начала видео.
  '''
  return sliding_window(video_path, window_size, overlap)


def corners_sort(corners: np.ndarray) -> np.ndarray:

    '''
    Сначала была идея пойти от центроид и построить сортировку углов на if, elif, else, но такой подход будет слабым при экстремальной перспективе.
    Если камера будет стоять под углом на площадку, то все может поломаться, так как Y верхнего правого угла может стать меньше Y правого нижнего.
    Напрмиер при проекции трапецевидного паралеллипипеда, а не простой трапеции.

    Поэтому решил считать в радианах через arctan2. После раскладывать углы "по кругу" и собирать начиная с Top Left (TL)
    '''

    center = np.mean(corners, axis=0)

    angles = np.arctan2(corners[:,1] - center[1],
                        corners[:,0] - center[0])

    sorted_idx = np.argsort(angles)

    corners = corners[sorted_idx]

    sums = corners[:,0] + corners[:,1]
    start_idx = np.argmin(sums)

    corners = np.roll(corners, -start_idx, axis=0)

    return corners

def court_zone(model: ultralytics.YOLO,
               video_path: str,
               window_size: int = 30,
               overlap: int = 15) -> np.ndarray:

    '''Функция детекции  площадки

    Принцип работы:
    Обрабатывает кадры видео пока кол-в валидных кадров (valid_frames) < необходимого кол-ва кадров (total_frames = 450)
    Забирает из генератора словарь, в нем по ключу ["data"] забирает фреймы. Ищет и определяет на нем углы площадки.

    Если кол-во углов не равно 4 (if len(corners) != 4) пропускает кадр.

    В ином слуачае отправляет углы в corners_sort для поддержания порядка углов.

    Углы после сортировки сохраняет в corners_court (вид [[],[],[],...[]]
    и считается медианна между всеми углами каждой зоны (top left и тд)

    возвращает median_corners нампай массив с 8 точками координат углов (x, y каждого угла)
    '''

    processed_frames = 0
    valid_frames = 0

    total_frames = 450

    corners_court = []

    frame_gen = gen_maker(video_path, window_size, overlap) #Пересоздаем генератор при каждом запуске, чтобы запрашивать каждый раз первые 450 кадров

    start = time.time()

    break_outer = False

    while valid_frames < total_frames:
      frames = next(frame_gen)

      for frame in frames["data"]:
        processed_frames += 1
        print(f"Обрабатывается кадр №{processed_frames}")

        court_detect = model(frame)[0]

        if court_detect.masks is None:
          continue

        polygon_points = court_detect.masks.xy[0]
        contour = polygon_points.astype(np.int32)

        perimeter = cv2.arcLength(contour, True)

        e = 0.02 * perimeter
        approx_cornes = cv2.approxPolyDP(contour, e, True)

        corners = approx_cornes.reshape(-1, 2)

        if len(corners) != 4:
          continue

        corners = corners_sort(corners)

        corners_court.append(corners)
        valid_frames += 1

        if valid_frames >= total_frames:
          break_outer = True
          break

      if break_outer:
        break

    end = time.time()

    print(f"Обработано кадров: {processed_frames}")
    print(f"Валидных кадров: {valid_frames}")
    print(f"Время: {round(end - start, 1)} сек")

    corners_court = np.array(corners_court)

    median_corners = np.median(corners_court, axis=0)

    return median_corners

def build_homography(court_cords, physical_cords):

    '''Перевод площадки из системы координат площадки в физическую систему координат

    Размер "мира" 1900 : 2800 (worldld_size = (1900, 2800) где 1 пиксель равен 1 сантиметру реального мира

    перевод выполняется за счет гомографии (cv2.findHomography)'''

    H, mask = cv2.findHomography(court_cords,
                                 physical_cords,
                                 method=cv2.RANSAC)

    return H