import cv2.cv2 as cv2  # and not just `import cv2`, to make VSCode happier
import numpy as np

from PIL import Image


PIL_GRAYSCALE = "L"


def read_grayscale(filename: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


def pad_and_resize(image: Image.Image, desired_size: int = 48) -> Image.Image:
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new(PIL_GRAYSCALE, (desired_size, desired_size), 255)
    x_offset = (desired_size - new_size[0]) // 2
    y_offset = (desired_size - new_size[1]) // 2
    new_image.paste(resized_image, (x_offset, y_offset))

    return new_image


if __name__ == '__main__':
    import sys
    np.set_printoptions(edgeitems=30, linewidth=200, precision=1)
    print(read_grayscale(sys.argv[1]) / 255.0)
