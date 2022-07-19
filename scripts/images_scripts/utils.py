import cv2
import os
from PIL import Image


def resize_images(
        input_dir: str,
        target_dir: str,
        target_shape: tuple = (256, 256),
        interpolation=cv2.INTER_AREA
):
    os.makedirs(target_dir, exist_ok=True)

    folders = os.listdir(input_dir)

    for folder in folders:
        images = os.listdir(os.path.join(input_dir, folder))

        for image in images:
            loaded_image = cv2.imread(os.path.join(input_dir, folder, image))
            resized_image = cv2.resize(loaded_image, target_shape, interpolation=interpolation)
            cv2.imwrite(os.path.join(target_dir, image), resized_image)


def open_image_RGB(path_to_open):
    image = Image.open(path_to_open)
    return image


if __name__ == "__main__":

    resize_images(
        input_dir='/home/juravlik/Downloads/hnm_images',
        target_dir='/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/resized_images',
        target_shape=(256, 256),
        interpolation=cv2.INTER_AREA
    )
