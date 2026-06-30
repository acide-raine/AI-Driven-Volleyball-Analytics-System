import cv2
import json

from develop.ball.scr import detection, tracker, config


cap = cv2.VideoCapture(config.VIDEO_PTAH)

orig_h, orig_w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)

MODEL = detection.load_model(config.MODEL_PATH)

input_name, output_name, output_seq, input_shape, output_shape = detection.model_params(MODEL)

frame_id_counter = 0
track_id_counter = 0

buffer_frame = []
detect = {}

all_track = {}

best_track = {}

main_id = None

test_count = 0

json_database = {}
json_file_path = "tracking_results.json"

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    else:
        frame_id_counter += 1
        preprocessed_frame = detection.preprocess_for_onnx(frame, config.INPUT_W, config.INPUT_H)
        buffer_frame.append(preprocessed_frame)

        if len(buffer_frame) == config.SEQ_LEN:

            #Detection part

            INPUT = detection.input_tensor(buffer_frame)
            buffer_frame.pop(0)

            OUTPUT = detection.inference(MODEL, input_name, output_name, INPUT)

            visible, decode_x, decode_y = detection.heatmap_decoder(OUTPUT, config.HEATMAP_TRASH)

            x, y = detection.original_dimension(decode_x, decode_y,
                                                config.INPUT_W, config.INPUT_H,
                                                config.VIDEO_WIDTH, config.VIDEO_HEIGHT)

            detect = detection.create_dict(frame_id_counter, visible, x, y)
            test_count += 1

            if test_count % 60 == 0:
                print(f"детекция на кадре {test_count}")
                print(detect)

            #Make and preprocess track

            matching = tracker.match_detection(all_track, detect)

            if matching:
                matching_id = list(matching.keys())[-1]
                current_track = tracker.update_track(all_track[matching_id], detect)
                if test_count % 60 == 0:
                    print()
                    print("текущий трек")
                    print(current_track)

            elif detect["visible"] == 1:
                track_id_counter += 1

                all_track[track_id_counter] = tracker.create_track(detect)

            all_track = tracker.gap_cleaner(all_track, detect)

            main_id = tracker.select_main_ball(all_track)


            if main_id:
                best_track = {main_id: all_track[main_id]}
                if test_count % 60 == 0:
                    print()
                    print("Лучший трек")
                    print(best_track)

                track_key = str(main_id)
                json_database[track_key] = all_track[main_id]

            else:
                continue

        else:
            continue


def convert_numpy(obj):
    if hasattr(obj, "dtype"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj

clean_json_database = convert_numpy(json_database)

with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(clean_json_database, f, ensure_ascii=False, indent=4)

print(f"Данные успешно сохранены в {json_file_path}")











