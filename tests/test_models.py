"""Unit tests for EN3150 Assignment 03 custom and pretrained architectures.

Validates:
1. Model B adheres strictly to the sub-100,000 parameter constraint.
2. Exact parameter counts match theoretical calculations in the assignment report.
3. Forward pass produces correct output tensor dimensions (batch_size, num_classes).
4. Depthwise separable block architecture integrity.
5. Pretrained backbones (MobileNetV2, EfficientNet-B0) initialize with the correct 10-class classifier.
"""

import unittest
import torch
import torch.nn as nn

from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.models.blocks import DepthwiseSeparableConv
from src.models.pretrained import build_pretrained, PRETRAINED_MODELS
from src.utils import count_parameters
from src.config import MODEL_B_MAX_PARAMS


class TestModelArchitectures(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.in_channels = 3
        self.img_size = 64
        self.num_classes = 10
        self.dummy_input_64 = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)

    def test_depthwise_separable_block(self):
        """Test DepthwiseSeparableConv block preserves spatial resolution and channels correctly."""
        in_c, out_c = 16, 32
        block = DepthwiseSeparableConv(in_c, out_c)
        x = torch.randn(2, in_c, 32, 32)
        out = block(x)

        self.assertEqual(out.shape, (2, out_c, 32, 32),
                         f"Expected shape (2, {out_c}, 32, 32), got {out.shape}")

    def test_model_a_parameters_and_shape(self):
        """Test Model A parameter count and output classification tensor shape."""
        model = ModelA(num_classes=self.num_classes)
        params = count_parameters(model)

        # Expected verified parameter count from report / hand calculation
        expected_params = 549_290
        self.assertEqual(params, expected_params,
                         f"Model A parameters mismatch: expected {expected_params:,}, got {params:,}")

        out = model(self.dummy_input_64)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes),
                         f"Expected output shape ({self.batch_size}, {self.num_classes}), got {out.shape}")

    def test_model_b_parameter_budget(self):
        """Test Model B strictly respects the assignment's <= 100,000 parameter budget."""
        model = ModelB(num_classes=self.num_classes)
        params = count_parameters(model)

        self.assertLessEqual(params, MODEL_B_MAX_PARAMS,
                             f"Model B parameter count {params:,} exceeds budget of {MODEL_B_MAX_PARAMS:,}")

        # Verified exact count
        expected_params = 13_288
        self.assertEqual(params, expected_params,
                         f"Model B parameters mismatch: expected {expected_params:,}, got {params:,}")

    def test_model_b_output_shape(self):
        """Test Model B forward pass output tensor dimensions."""
        model = ModelB(num_classes=self.num_classes)
        out = model(self.dummy_input_64)

        self.assertEqual(out.shape, (self.batch_size, self.num_classes),
                         f"Expected output shape ({self.batch_size}, {self.num_classes}), got {out.shape}")

    def test_pretrained_models_classifier_replacement(self):
        """Test that pretrained backbones adapt to EuroSAT 10 classes and produce valid output shapes."""
        for model_name in PRETRAINED_MODELS:
            model = build_pretrained(model_name, num_classes=self.num_classes)
            model.eval()

            # Test input size: 128 for MobileNetV2, 64 for EfficientNet-B0
            size = 128 if model_name == "mobilenet_v2" else 64
            dummy_input = torch.randn(2, 3, size, size)

            with torch.no_grad():
                out = model(dummy_input)

            self.assertEqual(out.shape, (2, self.num_classes),
                             f"{model_name} output shape mismatch: expected (2, {self.num_classes}), got {out.shape}")


if __name__ == "__main__":
    unittest.main()

