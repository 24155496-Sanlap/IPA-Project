"""Regression tests for filter input/output contracts."""

import unittest

import numpy as np

from filters import name_function_map, validate_image, blend_images


class FilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = np.random.default_rng(7).integers(
            0, 256, size=(32, 48, 3), dtype=np.uint8
        )

    def test_all_default_filters_return_rgb_uint8(self):
        import json
        from pathlib import Path
        def build_kwargs(name, values):
            values = dict(values)
            if name == "vignette":
                values["center"] = (values.pop("center_x"), values.pop("center_y"))
            elif name == "monochrome":
                values["tint"] = tuple(values.pop(f"tint_{channel}") for channel in "rgb")
            elif name == "cinematic":
                values["shadow_tint"] = tuple(values.pop(f"shadow_tint_{channel}") for channel in "rgb")
                values["highlight_tint"] = tuple(values.pop(f"highlight_tint_{channel}") for channel in "rgb")
            return values

        config_path = Path(__file__).resolve().parents[1] / "parameters.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))["filters"]
        for name, function in name_function_map.items():
            with self.subTest(filter=name):
                values = {
                    raw_name.lower().replace(" ", "_"): spec["default"]
                    for raw_name, spec in config[name].items()
                    if isinstance(spec, dict) and "type" in spec
                }
                result = function(self.image, **build_kwargs(name, values))
                self.assertEqual(result.shape, self.image.shape)
                self.assertEqual(result.dtype, np.uint8)

    def test_filters_do_not_mutate_source(self):
        source = self.image.copy()
        for name, function in name_function_map.items():
            with self.subTest(filter=name):
                function(source)
                np.testing.assert_array_equal(source, self.image)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(TypeError):
            validate_image([[1, 2, 3]])
        with self.assertRaises(ValueError):
            validate_image(np.zeros((10, 10), dtype=np.uint8))
        with self.assertRaises(TypeError):
            validate_image(np.zeros((10, 10, 3), dtype=np.float32))

    def test_blend_endpoints(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        filtered = np.full((2, 2, 3), 200, dtype=np.uint8)
        np.testing.assert_array_equal(blend_images(source, filtered, 0), source)
        np.testing.assert_array_equal(blend_images(source, filtered, 1), filtered)


if __name__ == "__main__":
    unittest.main()
