import numpy as np
from typing import Tuple, Dict, List
from develop.ball.scr.MyTypes import Track, Detection
from develop.ball.scr import config


def track_size(last_frame: int, first_frame: int, fps :float = config.FPS ) -> Tuple[float, float]:

    '''
    Функция расчета длительности треков в фреймах и секундах.

    :param last_frame: int – последний кадр трека
           first_frame: int – первый кадр трека
           fps: float – кол-во кадров/секунда видео

    :return: size_frame: float – размер трека в фреймах
             size_sec: float – размер трека в секундах
    '''

    size_frame = float(last_frame - first_frame)

    size_sec = size_frame / fps if fps > 0 else 0.0

    return size_frame, size_sec

def track_distance(x: list, y: list) -> Tuple[float, float]:

    '''
    Функция рассчета дистанции, которую прошел объект по осям X, Y

    :param x: list – список всех Х координат
           y: list – список всех Y координат

    :return: x_dist: float – дистанция по оси Х, которую проделал объект за весь трек
             y_dist: float – дистанция по оси Y, которую проделал объект за весь трек
    '''

    x_dist = float(max(x) - min(x))
    y_dist = float(max(y) - min(y))

    return x_dist, y_dist

def pred_calculate(x: List[float], y: List[float]) -> Tuple[float, float]:
    '''

    ПЕРЕПИСАТЬ

    Функция прогноза положения объекта на основе скорсоти исходя из последних детектированных положений.

    Проходим циклом по массиву class_cords ищем последнее значение "pred", сохраняем в список found.
    Для расчета нужно 2 индекса, допустимая погрешность 5 кадров (0.16 секунды) – аргумент limit.

    Если за последнии 5 кадров мы встретили 2 "pred", то рассчитываем следующее положение объекта на оснвое скорости.

    В каждом массиве детектируемых координат берем элементы по индексам в массиве found и вычитаем. Получаем скорость.
    Далее к последней детектированной координате прибавляем скорость – это прогнозируемое положение объекта.

    :param x: list – список всех Х координат
           y: list – список всех Y координат
           class_cords: list – список всех классов координат ["detect", "pred"]
           limit: int – кол-во элементов, которые мы переберем, чтоб найти 2 "pred"

    :return: pred_x: float – предсказанная координата X
             pred_y: float – предсказанная координата Y

             Если за limit не смогли найти 2 "pred", тоЖ
             -1, -1
    '''

    if x[-1] >= 0:

        x_speed = x[-1] - x[-2]
        y_speed = y[-1] - y[-2]

        pred_x = x[-1] + x_speed
        pred_y = y[-1] + y_speed

    else:

        real_values_x = [v for v in x if v >= 0]
        real_values_y = [v for v in y if v >= 0]

        if len(real_values_x) > 2:

            mean_speed_x = np.mean(np.diff(real_values_x)[-10:])
            mean_speed_y = np.mean(np.diff(real_values_y)[-10:])

            pred_x = x[-2] + mean_speed_x
            pred_y = y[-2] + mean_speed_y

        else:
            return real_values_x[-1], real_values_y[-1]

    return pred_x, pred_y

def create_track(detection: Detection ) -> Track:

    '''
    Функция создания трека из детектируемых координат.

    :param detection: dict [key:str, value: any] – словарь с следующими значениями: frame: int – номер фрейма
                                                                                    visible: int – видимость мяча на фрейме
                                                                                    x: float – массив координат Х
                                                                                    y: float – массив координат Y

    :return: track: dict [key: str, value: any] – словарь с следующими значениями: frame_idx: list – массив фреймов входящих в трек
                                                                                   class_cords: list – массив классов координат
                                                                                   x: list – массив детектированных координат Х
                                                                                   y: list – массив детектированных координат Y
                                                                                   pred_x: list – массив прогнозированных координат Х
                                                                                   pred_y: list – массив прогнозированных координат Y
                                                                                   size_frame: float – размер трека в фреймах
                                                                                   size_sec: float – размер трека в секундах
                                                                                   dist_x: float – дистанция движения объекта по Х
                                                                                   dist_y: float – дистанция движения объекта по Y
                                                                                   first_frame: int – первый фрейм трека
                                                                                   last_frame: int – последний фрейм трека
    '''

    x = detection["x"]
    y = detection["y"]
    frame_idx = detection["frame"]

    track = {"frame_idx": [frame_idx],
             "x": [x],
             "y": [y],
             "pred_x": [x],
             "pred_y": [y],
             "size_frame": None,
             "size_sec": None,
             "dist_x": None,
             "dist_y": None,
             "first_frame": frame_idx,
             "last_frame": frame_idx}

    return track

def update_track(track: Track, detection: Detection) -> Track:
    '''
    Функция обновления трека. Структуру входящих и исходящих словарей можно посмотреть в описании create_track.
    Ниже изложен принцип работы.

    frame_idx – формируется из frame в detection

    class_cords – формируется из visible в detection. Если visible == 0, то class_cords == "pred".
    Если visible == 1, то class_cords == "detect". Все "pred" в будущем будут обработаны,
    а соответствующие им Х и Y спрогнозированы

    x, y – формируются из x, y из словаря detection.
    Если кол-во элементво в массиве больше двух, то их же мы записываем в pred_x, pred_y.
    Иначе вызываем find_last_real, считаем pred_x, pred_y и записываем в ключ словаря.
    При visible == 0 в х, у записывается -1, -1. Мы заменяем эти значения на pred_x, pred_y и записываем в track их

    first_frame – это значение в frame_idx по индексу 0

    last_frame – это значение в frame_idx по индексу -1

    size_frame, size_sec – считаем с помощью track_size

    dist_x, dist_y – считаем с помощью track_distance

    В return мы отдаем обновленный track
    '''

    track["frame_idx"].append(detection["frame"])

    track["x"].append(detection["x"])
    track["y"].append(detection["y"])

    if len(track["x"]) < 2:
        pred_x, pred_y = track["x"][-1], track["y"][-1]
    else:
        pred_x, pred_y = pred_calculate(track["x"], track["y"])

    track["pred_x"].append(pred_x)
    track["pred_y"].append(pred_y)

    first_frame = track["frame_idx"][0]
    last_frame = track["frame_idx"][-1]

    track["first_frame"] = first_frame
    track["last_frame"] = last_frame

    track_sz_fr, track_sz_sec = track_size(last_frame, first_frame)
    dist_x, dist_y = track_distance(track["x"], track["y"])

    track["size_frame"] = track_sz_fr
    track["size_sec"] = track_sz_sec

    track["dist_x"] = dist_x
    track["dist_y"] = dist_y

    return track

def match_detection(all_tracks: Dict[int, Track], detections: Detection) -> Dict[int, Detection]:

    '''

    ПЕРЕПИСАТЬ

    Функция сопоставления детекции с треком.
    При видимости мяча ("visible" == 1) – мы ищем дельты растояний между предсказанными и детектируемыми координатами И
    рассчитываем эвклида между ними. Если эвклид больше max_dist, то это или новый трек, или шум.
    Если эвклид меньше, то запоминаем id трека и привязываем к нему детекцию. Пока эвклид < max_dist.

    max_dist – максимально возможная погрешность между pred и detect

    :param all_tracks: dict[int, Track] – словарь всех треков. Структуру Track можно посмотреть в */scr/MyTypes.py
           detections: Detection – словарь с данными детекции. Структуру Detection можно посмотреть в */scr/MyTypes.py

    :return:Dict[int, Detection] – словарь с лучшими id треков и детекциями
    '''

    max_dist = 200.0
    matched = {}

    best_track_id = None
    best_dist = float('inf')

    if detections["visible"] == 0:
        return matched

    for track_id, track in all_tracks.items():

        dx = track["pred_x"][-1] - detections["x"]
        dy = track["pred_y"][-1] - detections["y"]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < best_dist:
            best_dist = dist
            best_track_id = track_id

    if best_track_id is not None and best_dist < max_dist:
        matched[best_track_id] = detections

    return matched


def gap_cleaner(all_track: Dict[int, Track], detection: Detection) -> Dict[int, Track]:

    '''
    Функция чистки трекво по  разрыву.
    Сравниваем разницу между последним фреймом детекции и трека, если больше 30 (gap), то удаляем.
    Сравниваем длинну x если меньше 15 (min_lenght), то удаляем.

    :param all_tracks: dict[int, Track] – словарь всех треков. Структуру Track можно посмотреть в */scr/MyTypes.py
           detections: Detection – словарь с данными детекции. Структуру Detection можно посмотреть в */scr/MyTypes.py

    :return: to_short: list[int] – cписок коротких треков
             big_gap: list[int] – список трекво с большим разрывом
             all_track: dict[int, Track] – мутировавший all_track без треков из to_short и big_gap
    '''

    frame = detection["frame"]

    gap = 40

    for track_id in list(all_track.keys()):
        track = all_track[track_id]

        if frame - track["last_frame"] >= gap:
            del(all_track[track_id])

    return all_track

def stability_track_analyzer (track: Track) -> float:

    '''
    Функция анализа стабильности трека.
    Проходит по всем детекциям трека (class_cords == "detect"). И считает скорости между соседними координатами.
    Далее считаем дисперсию – чем она плавнее, тем более вероятно, что это мяч (движется плавно, иногда ускоряется/замедляется)

    :param track: Track – словарь с данными о треке

    :return:score: float – очки стабильности трека
    '''

    speeds = []

    all_x = track["x"]
    all_y = track["y"]
    all_frame = track["frame_idx"]

    for index, _ in enumerate(all_x[1:], start=1):

        dx = all_x[index] - all_x[index - 1]
        dy = all_y[index] - all_y[index - 1]
        dt = all_frame[index] - all_frame[index - 1]

        speed_x = dx / dt
        speed_y = dy / dt

        speeds.append([speed_x, speed_y])

    if len(speeds) >= 2:
        var_x = np.var([s[0] for s in speeds])
        var_y = np.var([s[1] for s in speeds])
        total_var = var_x + var_y

        mean_x = np.mean([s[0] for s in speeds])
        mean_y = np.mean([s[1] for s in speeds])
        mean_speed = (mean_x**2 + mean_y**2)**0.5

        stability = mean_speed/ (1 + total_var)

        lenght_weight = np.log(len(speeds) + 1)

        # distance_weight = (track["dist_x"] + track["dist_y"]) + 1

        score = stability * lenght_weight

    else:
        score = 0

    return score

def select_main_ball (all_tracks: Dict[int, Track]) -> int:

    '''
    Функция определения главного мяча.
    Проходит по трекам и выбирает с наивысшим score.

    :param all_tracks: dict[int: Track] – словарь всех треков.

    :return: best_track_id: int – id трека с наибольшим score
    '''
    best_track_id = None
    best_score = 0

    for track_id, track in all_tracks.items():

        score = stability_track_analyzer(track)

        if score > best_score:
            best_score = score
            best_track_id = track_id
        else:
            continue

    return best_track_id