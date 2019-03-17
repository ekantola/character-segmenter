import sample_image
import unittest

from segmenter import get_letter_images


class CharacterSegmenterTest(unittest.TestCase):
    def test_get_letter_images_for_word_skiers_with_defaults(self):
        sample = sample_image.read_grayscale('test_data/word_skiers.jpg')
        letters = get_letter_images(sample)

        self.assertEqual(
            [(19, 24), (17, 32), (6, 28), (16, 19), (15, 19), (15, 22)],
            list(map(image_size, letters))
        )

    def test_segment_word_heater_with_max_length(self):
        sample = sample_image.read_grayscale('test_data/word_heater.tiff')
        letters = get_letter_images(sample, expected_max_boxes=6)

        self.assertEqual(
            [(14, 30), (21, 25), (17, 22), (10, 23), (17, 19), (15, 19)],
            list(map(image_size, letters))
        )

    def test_get_letter_images_for_word_tonally_with_min_length(self):
        sample = sample_image.read_grayscale('test_data/word_tonally.png')
        letters = get_letter_images(sample, expected_min_boxes=7)

        self.assertEqual(
            [(11, 24), (19, 18), (20, 21), (17, 23), (5, 33), (4, 31), (22, 29)],
            list(map(image_size, letters))
        )


def image_size(img):
    return img.size
