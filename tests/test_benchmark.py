"""Unit tests for the hardware profiler and MAC/FLOP calculations."""

import unittest
from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.benchmark import compute_macs_and_flops


class TestBenchmarkProfiler(unittest.TestCase):

    def test_model_a_macs(self):
        """Model A MAC count should be approximately 11.73 M-MACs."""
        model = ModelA(num_classes=10)
        macs, flops = compute_macs_and_flops(model, (64, 64))

        # Expected: ~11,725,184 MACs
        self.assertAlmostEqual(macs / 1e6, 11.73, delta=0.05)
        self.assertEqual(flops, macs * 2)

    def test_model_b_mac_reduction(self):
        """Model B must have significantly fewer MACs than Model A."""
        model_a = ModelA(num_classes=10)
        model_b = ModelB(num_classes=10)

        macs_a, _ = compute_macs_and_flops(model_a, (64, 64))
        macs_b, _ = compute_macs_and_flops(model_b, (64, 64))

        # Model B MACs should be ~5.14M, which is >50% reduction compared to Model A
        self.assertLess(macs_b, macs_a)
        reduction = (1 - macs_b / macs_a) * 100
        self.assertGreater(reduction, 50.0, f"Expected >50% reduction in MACs, got {reduction:.2f}%")


if __name__ == "__main__":
    unittest.main()
