"""Regression checks without model downloads or private card source files."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elestrals_mask as workflow


class WorkflowTests(unittest.TestCase):
    def test_models_never_receive_stat_bar_or_lower_pixels(self):
        seen = []

        def remove(picture, **kwargs):
            seen.append(picture.size)
            return Image.new('L', picture.size, 255)

        fake = types.SimpleNamespace(new_session=lambda *args, **kwargs: object(), remove=remove)
        with patch.dict(sys.modules, {'rembg': fake}):
            mask = workflow.model_masks({'card': np.zeros((20, 12, 3), np.uint8)},
                                        ['birefnet-general'], 13)['card', 'birefnet-general']
        self.assertEqual(seen, [(12, 13)])
        self.assertEqual(mask.shape, (20, 12))
        self.assertTrue((mask[:13] == 255).all())
        self.assertFalse(mask[13:].any())

    def test_lower_layout_cannot_be_overwritten_by_any_model(self):
        rgb = np.full((2048, 1468, 3), [80, 100, 120], np.uint8)
        bar = np.zeros(rgb.shape[:2], bool)
        bar[1540:1950, 60:1400] = True
        layout = workflow.detected_layout(rgb, bar)
        self.assertEqual(workflow.stat_bar_top(bar), 1540)
        self.assertFalse((layout[1540:] == 128).any())
        self.assertTrue((layout[bar] == 255).all())
        for value in (0, 128, 255):
            model = np.full(layout.shape, value, np.uint8)
            final = np.where(layout == 128, model, layout)
            np.testing.assert_array_equal(final[1540:], layout[1540:])

    def test_empty_frame_cannot_create_an_unbounded_model_region(self):
        with self.assertRaises(ValueError):
            workflow.stat_bar_top(np.zeros((20, 12), bool))

    def test_footer_preserves_credits_but_not_detached_black_art(self):
        rgb = np.full((2048, 1468, 3), [70, 80, 100], np.uint8)
        boundary = (1930 + 20 * np.sin(np.linspace(0, 6, 1468))).astype(int)
        for column, row in enumerate(boundary):
            rgb[row:, column] = 0
        rgb[1980:1990, 350:700] = 255
        rgb[1840:1850, 400:420] = 0
        footer = workflow.footer_mask(rgb)
        self.assertTrue(footer[1980:1990, 350:700].all())
        self.assertFalse(footer[1840:1850, 400:420].any())
        self.assertTrue(footer[1951:].all())
        self.assertFalse(footer[1900:1908, 100:1300].any())

    def test_black_connected_art_does_not_become_border(self):
        rgb = np.full((300, 200, 3), [100, 140, 180], np.uint8)
        rgb[:, :3] = 0
        rgb[100:120, :100] = 0
        _, border, _ = workflow.frame_masks(rgb)
        self.assertTrue(border[110, 1])
        self.assertFalse(border[110, 80])

    def test_subject_corrections_require_matching_artwork(self):
        blank = np.full((178, 128, 3), 100, np.uint8)
        self.assertEqual(workflow.subject_correction(blank), (None, None))
        for entry in ('trifernal', 'emphyrix', 'stratomoth'):
            with self.subTest(artwork=entry):
                reference = Path(workflow.ROOT) / 'corrections' / (entry + '-reference.png')
                rgb = np.array(Image.open(reference).convert('RGB'))
                mask, name = workflow.subject_correction(rgb)
                self.assertIsNotNone(name)
                self.assertEqual(mask.shape, rgb.shape[:2])


if __name__ == '__main__':
    unittest.main()
