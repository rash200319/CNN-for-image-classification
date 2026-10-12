"""Unit tests for the single-image inference pipeline."""

import unittest
from pathlib import Path
import torch
from PIL import Image

from src.models.model_b import ModelB
from src.predict import get_inference_transform, predict_image
from src.config import CLASS_NAMES


class TestPredictPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_img_path = Path("tests/temp_test_image.jpg")
        # Create a dummy RGB image
        img = Image.new("RGB", (64, 64), color=(73, 109, 137))
        img.save(self.temp_img_path)

        self.model = ModelB(num_classes=10)
        self.device = torch.device("cpu")
        self.transform = get_inference_transform("b", img_size=64)

    def tearDown(self):
        if self.temp_img_path.exists():
            self.temp_img_path.unlink()

    def test_predict_image_structure(self):
        """predict_image must return top-k sorted predictions with valid probabilities."""
        results = predict_image(self.model, self.temp_img_path, self.transform, self.device, top_k=3)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn(r["class_name"], CLASS_NAMES)
            self.assertGreaterEqual(r["confidence"], 0.0)
            self.assertLessEqual(r["confidence"], 1.0)
            self.assertAlmostEqual(r["confidence_percent"], r["confidence"] * 100, delta=0.01)

        # Confidence should be in descending order
        self.assertGreaterEqual(results[0]["confidence"], results[1]["confidence"])
        self.assertGreaterEqual(results[1]["confidence"], results[2]["confidence"])


if __name__ == "__main__":
    unittest.main()
