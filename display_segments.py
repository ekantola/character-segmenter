'''
Tool for visualising the segmented letters for given image files, to help tune the segmenting algorithm.

If image files to segment have names ending with the actual solved text, separated by an underscore and until the extension, then use that to additionally color-code the segmented character images based on whether the number of segments match the lenght of the actual solution or not. This might be useful for detecting issues in segmentation.

Example: a file "myimage_textwithin.jpg" would suggest the actual solution is "textwithin" (length 10). If number of detected segments is 10, then border color will be green, otherwise red.
'''

import sample_image
import re
import sys

from segmenter import get_letter_images
from glob import iglob
from itertools import chain
from PIL import Image
from PIL.ImageColor import getrgb
from typing import List

BORDER_SIZE = 3

COLOR_OK = getrgb('green')
COLOR_NEUTRAL = getrgb('grey')
COLOR_ERROR = getrgb('red')


def combine_segmented_letter_images(filename: str, expected_length: int = None) -> Image.Image:
    letter_imgs = list(map(sample_image.pad_and_resize, get_letter_images(
        sample_image.read_grayscale(filename),
        expected_min_boxes=expected_length,
        expected_max_boxes=expected_length,
        contour_threshold=80)))
    widths, heights = zip(*(i.size for i in letter_imgs))

    if expected_length == None:
        border_color = COLOR_NEUTRAL
    elif len(letter_imgs) == expected_length:
        border_color = COLOR_OK
    else:
        border_color = COLOR_ERROR

    max_height = max(heights) + 2 * BORDER_SIZE
    combined_width_with_borders = sum(
        widths) + (len(widths) + 1) * BORDER_SIZE
    combined_img = Image.new(
        'RGB', (combined_width_with_borders, max_height), border_color)

    x_offset = BORDER_SIZE
    for i in letter_imgs:
        combined_img.paste(i, (x_offset, BORDER_SIZE))
        x_offset += img_width(i) + BORDER_SIZE

    return combined_img


def combine_all_segmentation_results(filenames: List[str], expected_lengths: List[int]):
    images = []
    for filename, expected_length in zip(filenames, expected_lengths):
        try:
            images.append(combine_segmented_letter_images(
                filename, expected_length))
        except Exception as e:
            print(f'Skipping unprocessable file {filename}: {e}')

    max_width = max(map(img_width, images)) + 2 * BORDER_SIZE
    combined_height_with_borders = sum(
        map(img_height, images)) + (len(images) + 1) * BORDER_SIZE

    combined_img = Image.new('RGB', (max_width, combined_height_with_borders))

    y_offset = BORDER_SIZE
    for i in images:
        combined_img.paste(i, (BORDER_SIZE, y_offset))
        y_offset += img_height(i) + BORDER_SIZE

    return combined_img


def img_width(img: Image.Image) -> int:
    return img.size[0]


def img_height(img: Image.Image) -> int:
    return img.size[1]


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: {sys.argv[0]} image-filename-pattern...')
        exit(1)

    filenames = list(chain.from_iterable(map(iglob, sys.argv[1:])))
    print('Input files:\n' + '\n'.join(filenames))

    expected_lengths = []
    fnre = re.compile('^.*_([a-z]+)\\.[^.]+$')
    for filename in filenames:
        match = fnre.match(filename)
        expected_lengths.append(len(match[1]) if match else None)

    img = combine_all_segmentation_results(filenames, expected_lengths)
    img.show()
    img.save('out/display_segments.png')
