import cv2
import albumentations as A
from pathlib import Path
import shutil

IMG_PATHS = Path("/develop/detect_ball/Ball_Dataset/train/images")

LABEL_PATHS = Path("/develop/detect_ball/Ball_Dataset/train/labels")

augment = A.Compose([A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=0.2,
                    p=0.2),

                    A.MotionBlur(
                    blur_limit=(7, 12),
                    p=0.2),

                    A.GaussNoise(
                    std_range=(0.05, 0.15),
                    p=0.2),

                    A.ImageCompression(
                    quality_range=(50, 90),
                    p=0.2),

                    A.Downscale(
                    scale_range=(0.5, 0.9),
                    p=0.2),

                    A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=40,
                    val_shift_limit=20,
                    p=0.2),

                    A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=0.2)])


def augmentation(img_path, label_path, augment):

    img = cv2.imread(str(img_path))

    if img is None:
        return

    aug_img = augment(image=img)["image"]

    new_img_name = f"aug_{img_path.name}"
    new_label_name = f"aug_{label_path.name}"

    cv2.imwrite(str(IMG_PATHS / new_img_name), aug_img)
    shutil.copy(label_path, LABEL_PATHS / new_label_name)

aug_count = 0

for img_path in IMG_PATHS.glob("*.jpg"):
    label_path = LABEL_PATHS/ f"{img_path.stem}.txt"

    if not label_path.exists():
      continue

    augmentation(img_path, label_path, augment)
    aug_count += 1

print(f"aug_count: {aug_count}")

