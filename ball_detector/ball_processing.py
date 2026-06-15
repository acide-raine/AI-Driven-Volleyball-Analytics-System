import json
import numpy as np

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from ultralytics import YOLO

from court_detector.court_detect import court_zone

TEST_VIDEO_PATH = "/Users/pasha/Data_stat/test_video.avi"

COURT_MODEL_PATH = "/Users/pasha/Data_stat/court_detector/model/YOLO_court_detector.pt"
COURT_MODEL = YOLO(COURT_MODEL_PATH)
court_cords = court_zone(COURT_MODEL, TEST_VIDEO_PATH)

ball_cords = json.loads(open('example_ball_cords_sorted.json').read()) #кординаты мяча

def collect_flights(ball_cords:dict) -> dict:
    
    '''Разделяем все детекции мяча на "полеты".
    Полет – это непрерывное движение мяча n кадров подряд. В случае если между кадрами 
    с детекцтированным мячем больше MAX_GAP кадров, то следующее окно становится новым "полетом"
    
    Функция принимает:
    cords – словарь координат детектируемого мяча вида{
                                                        "0": {
                                                            "x": 752.2496337890625,
                                                            "y": 303.49700927734375,
                                                            "w": 18.454345703125,
                                                            "h": 17.9747314453125,
                                                            "scores": 0.5802273154258728
                                                        },
                                                        "1": {
                                                            "x": 745.4769287109375,
                                                            "y": 303.8711242675781,
                                                            "w": 18.6982421875,
                                                            "h": 18.3812255859375,
                                                            "scores": 0.4544442296028137
                                                        },
    
    Внутри держит:
    flights – dict копилка для полетов. Вид {frame : (x, y) 
    current_flight – list для хранения полета, который сейчас обрабатывается
    MAX_GAP – int размер разрыва между детекциями
    flights_count – счетчик полетов

    Возвращает:
    flights – dict вид {frame : (x, y)
    '''

    flights = {}
    current_flight = []
    MAX_GAP = 15
    flights_count = 0

    sorted_frames = sorted(ball_cords.keys(), key=int)

    for frame_str in sorted_frames:
        frame = int(frame_str)

        x = ball_cords[frame_str]["x"]
        y = ball_cords[frame_str]["y"]

        point = [frame, x, y]

        if len(current_flight) == 0:
            current_flight.append(point)
        else:
            last_frame = current_flight[-1][0]
            gap = frame - last_frame

            if gap <= MAX_GAP:
                current_flight.append(point)
            else:
                flights_count += 1
                flights[flights_count] = current_flight

                current_flight = [point]

    if len(current_flight) > 0:
        flights_count += 1
        flights[flights_count] = current_flight

    return flights

flights = collect_flights(ball_cords)

def restored_trajectory(flights):

    restored_cord = {}
    buffer = []
    restored_count = 0

    for flight in flights.values():

        flight_t = [val[0] for val in flight]
        flight_x = [val[1] for val in flight]
        flight_y = [val[2] for val in flight]

        flight_t_np = np.array(flight_t).reshape(-1, 1)

        if len(flight_t_np) < 4:
            continue

        model_x = Pipeline([
                ("poly", PolynomialFeatures(2)),
                ("model", RANSACRegressor())
        ])
        model_x.fit(flight_t_np, flight_x)

        model_y = Pipeline([
                ("poly", PolynomialFeatures(2)),
                ("model", RANSACRegressor())
        ])
        model_y.fit(flight_t_np, flight_y)

        restored_t = np.arange(flight_t_np.min(), flight_t_np.max() + 1).reshape(-1, 1)

        pred_x = model_x.predict(restored_t)
        pred_y = model_y.predict(restored_t)

        for i, frame in enumerate(restored_t):
            buffer.append([frame.item(), pred_x[i].item(), pred_y[i].item()])

        restored_cord[restored_count] = buffer
        restored_count += 1
        buffer = []

    return restored_cord
