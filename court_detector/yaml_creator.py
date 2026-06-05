from pathlib import Path
import yaml

'''Работал с собственным датасетом и эксперементировал в гугл коллабе.
Чтоб каждый раз не заливать .yaml сделал функцию которая сама его строи и сохраняет куда надо'''

dataset_path = Path("/detect_court/Yolo_dataset")

data = {
    "path": str(dataset_path),
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "court"
    }
}

with open(dataset_path / "data.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)
