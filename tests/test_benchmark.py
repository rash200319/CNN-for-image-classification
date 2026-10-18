"""Unit tests for the hardware profiler and MAC/FLOP calculations."""

import unittest
from src.benchmark import compute_macs_and_flops
from src.config import IMAGE_SIZE, MAX_IMAGE_SIZE, validate_image_size
from src.data.dataset import build_transforms
from src.models.model_a import ModelA
from src.models.model_b import ModelB


class TestBenchmarkProfiler(unittest.TestCase):

    def test_assignment_maximum_resolution_is_enforced(self):
        self.assertEqual(IMAGE_SIZE, 64)
        self.assertEqual(MAX_IMAGE_SIZE, 64)
        self.assertEqual(validate_image_size(64), 64)
        with self.assertRaises(ValueError):
            validate_image_size(65)

    def test_resize_pipeline_rejects_resolution_above_assignment_limit(self):
        with self.assertRaises(ValueError):
            build_transforms(65, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def test_model_a_macs(self):
        model = ModelA(num_classes=10)
        macs, flops = compute_macs_and_flops(model, (64, 64))

        self.assertAlmostEqual(macs / 1e6, 11.73, delta=0.05)
        self.assertEqual(flops, macs * 2)

    def test_model_b_mac_reduction(self):
        model_a = ModelA(num_classes=10)
        model_b = ModelB(num_classes=10)

        macs_a, _ = compute_macs_and_flops(model_a, (64, 64))
        macs_b, _ = compute_macs_and_flops(model_b, (64, 64))

        self.assertLess(macs_b, macs_a)
        reduction = (1 - macs_b / macs_a) * 100
        self.assertGreater(reduction, 50.0, f"Expected >50% reduction in MACs, got {reduction:.2f}%")


if __name__ == "__main__":
    unittest.main()
