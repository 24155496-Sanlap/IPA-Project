"""Unit tests for object-detection validation that do not require model weights."""

import unittest
import numpy as np

from object_detection import parse_class_prompts, validate_rgb_image


class ObjectDetectionHelperTests(unittest.TestCase):
    def test_prompts_split_deduplicate_and_normalize(self):
        self.assertEqual(
            parse_class_prompts(" red backpack, wooden chair\nred backpack; helmet "),
            ["red backpack", "wooden chair", "helmet"],
        )

    def test_empty_prompts_rejected(self):
        with self.assertRaises(ValueError):
            parse_class_prompts(" , ; \n ")

    def test_prompt_limit(self):
        with self.assertRaises(ValueError):
            parse_class_prompts(",".join(f"object{i}" for i in range(81)))

    def test_rgb_contract(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        result = validate_rgb_image(image)
        self.assertEqual(result.shape, (4, 5, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_non_rgb_or_wrong_dtype_rejected(self):
        with self.assertRaises(ValueError):
            validate_rgb_image(np.zeros((4, 5), dtype=np.uint8))
        with self.assertRaises(TypeError):
            validate_rgb_image(np.zeros((4, 5, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
