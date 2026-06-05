import cv2
import time

video_path = "/Stock_video/your_video.mp4"
window_size = 30
overlap = 15


def sliding_window(v_p, w_s, ovrlp):
    '''
    Функция генератор для чтения видео и сбора его кадров в массивы длинной w_s с перекрытием ovrlp

    На вход принимает:
    v_p – путь к видео
    w_s – размер окна
    ovrlp – размер перекрытия

    на выходе словарь (frame_array) вида:

    ["data"] – массивы кадров в окне (buffer)
    ["frames"] – список кадров, которые попали в окно (вид [1, 30])
    ["window_id"] – id окна
    ["time_taken"] – время на обработку w_s кадров

    так, при:
    w_s = 30
    ovrlp = 15

    Запустив функцию 2 раза вы получите 2 массива длинной 30.
    В массивах будут кадры [1, 30] и [16, 45]

    Сбор массива реализован через буфер (buffer).
    В первом запуске, он записывает в себя 30 кадров.
    В последующих удаляет step первых кадров и записываем новые пока len(buffer) < w_s
    '''

    cap = cv2.VideoCapture(v_p)
    step = w_s - ovrlp

    frame_counter = 0
    window_id = 0

    buffer = []

    while cap.isOpened():

        start = time.time()

        frame_array = {}

        ret, frame = cap.read()
        frame_counter += 1

        if not ret:
            break

        else:
            start = time.time()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(frame)

            if len(buffer) == w_s:

                window_id += 1

                frame_array["data"] = buffer
                frame_array["frames"] = [(frame_counter - w_s +1), frame_counter]
                frame_array["window_id"] = window_id

                end = time.time()
                time_taken = round((end - start),4)
                frame_array["time_taken"] = time_taken

                yield frame_array

                buffer = buffer[step:]

    cap.release()

# for window in sliding_window(video_path, 30, 15):
#     process(window) # тут потом поисываю логику что мне делать с каждым окном











