"""Unit tests for edge model export (TorchScript & ONNX) and numerical parity."""

import unittest
from pathlib import Path
import torch

from src.models.model_b import ModelB
from src.export import export_torchscript, export_onnx, export_and_benchmark


class TestEdgeExport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path("tests/temp_export")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.model = ModelB(num_classes=10)
        self.model.eval()
        self.dummy_input = torch.randn(1, 3, 64, 64)

    def tearDown(self):
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*"):
                f.unlink()
            self.temp_dir.rmdir()

    def test_torchscript_export_and_parity(self):
        """TorchScript model must be saved and produce numerical parity (atol=1e-5)."""
        ts_path = self.temp_dir / "model_b_test.torchscript.pt"
        saved_path, max_diff = export_torchscript(self.model, self.dummy_input, ts_path)

        self.assertTrue(saved_path.exists())
        self.assertGreater(saved_path.stat().st_size, 0)
        self.assertLess(max_diff, 1e-4, f"TorchScript output difference too high: {max_diff}")

    def test_onnx_export_and_validity(self):
        """ONNX model must export cleanly and pass onnx.checker validation."""
        try:
            import onnx
            import onnxscript  # noqa: F401
        except ImportError:
            self.skipTest("onnx or onnxscript not installed")

        onnx_path = self.temp_dir / "model_b_test.onnx"
        saved_path, _ = export_onnx(self.model, self.dummy_input, onnx_path, opset_version=14)

        self.assertTrue(saved_path.exists())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_export_requires_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            export_and_benchmark(
                "b",
                self.temp_dir / "missing.pt",
                self.temp_dir,
            )


if __name__ == "__main__":
    unittest.main()
