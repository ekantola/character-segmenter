import cv2.cv2 as cv2
import sample_image
import numpy as np
import sys

from PIL import Image
from functools import reduce
from operator import itemgetter
from typing import Iterable, List


class SegmenterException(Exception):
    pass


def get_letter_images(raw_image: np.ndarray, **kwargs) -> Iterable[Image.Image]:
    boxes = get_letter_bounding_boxes(raw_image, **kwargs)
    img = Image.fromarray(raw_image)

    def cropped_letter_img(box):
        return img.crop(box=(box[3][0] + 1, box[3][1] + 1, box[1][0], box[1][1]))

    return map(cropped_letter_img, boxes)


def get_letter_bounding_boxes(
        image: np.ndarray,
        expected_min_boxes: int = None,
        expected_max_boxes: int = None,
        contour_threshold: int = 80):
    '''
    Looks like the contour threshold of around 80 performs best for my test data, so choosing that one as a default. The case might be very different for some other kind of images.
    '''
    _, thresh = cv2.threshold(image, contour_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    letter_boxes = merge_vertical_overlaps(
        bounding_boxes(flatten_innermost(contours[1:])))

    if expected_max_boxes:
        while len(letter_boxes) > expected_max_boxes:
            letter_boxes = merge_two_boxes_with_smallest_areas(letter_boxes)

    if expected_min_boxes and len(letter_boxes) < expected_min_boxes:
        # Try once more harder with a lower threshold
        letter_boxes = get_letter_bounding_boxes(
            image, expected_min_boxes, expected_max_boxes, contour_threshold / 2)
        # If that still fails, then no can do
        if len(letter_boxes) < expected_min_boxes:
            raise SegmenterException(
                'Could not split letters enough to meet expected minimum ' +
                'of {expected_min_boxes} (now got: {len(letter_boxes)})')

    return letter_boxes


def flatten_innermost(contours: Iterable) -> Iterable[list]:
    return map(lambda contour: list(map(first_elem, contour)), contours)


def bounding_box(contour: list):
    x_coords = list(map(first_elem, contour))
    y_coords = list(map(second_elem, contour))
    x_min, y_min, x_max, y_max = min(x_coords), min(
        y_coords), max(x_coords), max(y_coords)

    return [[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]


def bounding_boxes(contours: Iterable[list]) -> Iterable[list]:
    return sorted(filter(box_area, map(bounding_box, contours)), key=by_x_min)


def by_x_min(box: list) -> int:
    return box[0][0]


def merge_vertical_overlaps(bounding_boxes: Iterable[list]) -> list:
    return list(reduce(merge_to_previous_if_overlap, bounding_boxes, []))


def merge_to_previous_if_overlap(merged: list, current_box: list) -> list:
    if len(merged) and has_vertical_overlap(merged[-1], current_box):
        merged[-1] = bounding_box(merged[-1] + current_box)
    else:
        merged.append(current_box)
    return merged


def has_vertical_overlap(bounding_box1: list, bounding_box2: list) -> bool:
    x_min1 = bounding_box1[0][0]
    x_max1 = bounding_box1[1][0]
    x_min2 = bounding_box2[0][0]
    x_max2 = bounding_box2[1][0]
    return max(x_min1, x_min2) < min(x_max1, x_max2)


def merge_two_boxes_with_smallest_areas(boxes: List[list]) -> List[list]:
    areas = list(map(box_area, boxes))
    adjacent_sums = [x + y for x, y in zip(areas, areas[1:] + [0])][:-1]
    first_index = min(enumerate(adjacent_sums), key=itemgetter(1))[0]
    return boxes[:first_index] + [bounding_box(boxes[first_index] + boxes[first_index+1])] + boxes[first_index+2:]


def box_area(box: List[list]) -> int:
    return box_width(box) * box_height(box)


def box_width(box: List[list]) -> int:
    return abs(box[1][0] - box[0][0])


def box_height(box: List[list]) -> int:
    return abs(box[1][1] - box[2][1])


def first_elem(x: Iterable):
    return x[0]


def second_elem(x: Iterable):
    return x[1]


if __name__ == '__main__':
    import os
    import re
    import sys
    import traceback
    from glob import iglob
    from itertools import chain

    if len(sys.argv) < 2:
        print('Usage: {sys.argv[0]} outdir image-filename-pattern...')
        exit(1)

    _, outdir, *file_patterns = sys.argv

    filenames = chain.from_iterable(map(iglob, file_patterns))
    filename_re = re.compile('^.*_([a-z]+)\\.[^.]+$')

    print('Segmenting images: ', end='', flush=True)
    input_count = 0
    success_count = 0
    success_letter_count = 0
    error_count = 0
    for filename in filenames:
        input_count += 1
        try:
            match = filename_re.match(filename)
            if match:
                solved_text = match[1]
                expected_length = len(solved_text)
            else:
                solved_text = None
                expected_length = None

            letter_imgs = list(map(sample_image.pad_and_resize, get_letter_images(
                sample_image.read_grayscale(filename),
                expected_min_boxes=expected_length,
                expected_max_boxes=expected_length,
                contour_threshold=80)))

            basename, _ = os.path.basename(filename).split(".")

            if not solved_text:
                solved_text = '_' * len(letter_imgs)

            for i, letter_img in enumerate(letter_imgs):
                letter = solved_text[i]
                letter_outdir = f'{outdir}/{letter}'
                os.makedirs(letter_outdir, exist_ok=True)
                letter_img.save(f'{letter_outdir}/{basename}_{i}.bmp')
                success_letter_count += 1

            success_count += 1

        except Exception as e:
            print(f'\nCould not process {filename}: {traceback.format_exc()}')
            error_count += 1

        if input_count % 1000 == 0:
            print(str(input_count) + '...', end='', flush=True)

    print('done.')
    print(
        f'Total {input_count} input files, {success_count} processed successfully ' +
        f'({success_letter_count} total letters segmented), with {error_count} errors')
