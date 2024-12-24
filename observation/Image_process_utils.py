from PIL import Image
import numpy as np

from observation.image_crop import process_image


def process_image_npy_i(npy, o_size, left, top, right, bottom, re_size):
    try:
        img = Image.fromarray(npy)
        assert o_size == img.size
        img = process_image(img, left, top, right, bottom, re_size)
        return np.array(img)
    except Exception as e:
        print(f"发生错误: {e}")


def process_image_npy(npy, type, image_params):
    params = image_params

    left, top, right, bottom = params[type]["crop"]

    return process_image_npy_i(
        npy, params[type]["size"], left, top, right, bottom, params[type]["resize"]
    )
