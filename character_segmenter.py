import cv2.cv2 as cv2
import sample_image
import numpy as np
import sys

from PIL import Image
from functools import reduce
from operator import itemgetter
from typing import Iterable


def get_letter_images(raw_image: np.ndarray, expected_max_length: int = 7, **kwargs) -> Iterable[Image.Image]:
    boxes = get_letter_bounding_boxes(raw_image, expected_max_length, **kwargs)
    img = Image.fromarray(raw_image)

    def crop_letter(box):
        return img.crop(box=(box[3][0] + 1, box[3][1] + 1, box[1][0], box[1][1]))

    return map(sample_image.pad_and_resize, map(crop_letter, boxes))


def get_letter_bounding_boxes(image: np.ndarray, expected_max_length: int, contour_threshold: int = 80):
    '''
    Looks like the contour threshold of around 80 performs best for my test data, so choosing that one as a default. The case might be very different for some other kind of images.
    '''
    _, thresh = cv2.threshold(image, contour_threshold, 255, cv2.THRESH_BINARY)
    [contours, _] = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    merged = merge_vertical_overlaps(
        bounding_boxes(flatten_innermost(contours[1:])))

    while len(merged) > expected_max_length:
        merged = merge_two_thinnest_adjacent_boxes(merged)

    return merged


def flatten_innermost(contours: Iterable) -> Iterable[list]:
    return map(lambda contour: list(map(first_elem, contour)), contours)


def bounding_box(contour: list):
    x_coords = list(map(first_elem, contour))
    y_coords = list(map(second_elem, contour))
    x_min, y_min, x_max, y_max = min(x_coords), min(
        y_coords), max(x_coords), max(y_coords)

    return [[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]


def bounding_boxes(contours: Iterable[list]) -> Iterable[list]:
    return sorted(map(bounding_box, contours), key=by_x_min)


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


def merge_two_thinnest_adjacent_boxes(boxes: list) -> list:
    widths = list(map(box_width, boxes))
    adjacent_sums = [x + y for x, y in zip(widths, widths[1:] + [0])][:-1]
    first_index = min(enumerate(adjacent_sums), key=itemgetter(1))[0]
    return boxes[:first_index] + [bounding_box(boxes[first_index] + boxes[first_index+1])] + boxes[first_index+2:]


def box_width(box):
    return box[1][0] - box[0][0]


def first_elem(x: Iterable):
    return x[0]


def second_elem(x: Iterable):
    return x[1]
