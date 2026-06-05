import os
import shutil
from sklearn.model_selection import train_test_split

images_dir = "/court_detector/dataset/images"
labels_dir = "/court_detector/dataset/labels"
output_dir = "/court_detector/Yolo_court_dataset"

image_extensions = (".jpg", ".jpeg", ".png")

all_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(images_dir)
    if f.lower().endswith(image_extensions)
]

train_names, val_names = train_test_split(all_names,
                                          test_size=0.2,
                                          random_state=42)

def move_files(name_list, split_name):

    '''Копирует изображения и соответствующие файлы разметки
    в структуру датасета YOLO.

    Параметры:
        name_list  — список имен файлов без расширения.
        split_name  — название выборки ("train" или "val").

    Логика работы:
        1. Находит изображение (.jpg, .jpeg или .png).
        2. Проверяет наличие соответствующего файла разметки .txt.
        3. Копирует изображение в images/<split_name>/.
        4. Копирует разметку в labels/<split_name>/.'''


    for name in name_list:

        img_name = None
        for ext in image_extensions:
            path = os.path.join(images_dir, name + ext)
            if os.path.exists(path):
                img_name = name + ext
                break

        txt_name = name + ".txt"

        img_path = os.path.join(images_dir, img_name) if img_name else None
        txt_path = os.path.join(labels_dir, txt_name)

        if img_path and os.path.exists(txt_path):

            shutil.copy(img_path,
                        os.path.join(output_dir, "images", split_name, img_name))

            shutil.copy(txt_path,
                        os.path.join(output_dir, "labels", split_name, txt_name))

os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

for split, names in [("train", train_names), ("val", val_names)]:
    move_files(names, split)

print(f"Готово! train={len(train_names)}, val={len(val_names)}")



