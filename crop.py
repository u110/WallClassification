import os
from PIL.Image import Image


def crop_center(pil_img: Image, crop_width: int, crop_height: int) -> Image:
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img: Image) -> Image:
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def scriptPath():
    return os.path.dirname(os.path.abspath(__file__))
