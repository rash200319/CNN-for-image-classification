"""Unit tests for dataset split verification and profiling logic."""

import unittest
from src.config import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT
from src.data.verify_dataset import verify_splits


class TestDatasetVerification(unittest.TestCase):

    def test_verify_splits_total_and_counts(self):
        total = 27000
        df = verify_splits(total)

        self.assertEqual(df["Samples"].sum(), total)

        train_count = df.loc[df["Split"] == "Training", "Samples"].values[0]
        val_count = df.loc[df["Split"] == "Validation", "Samples"].values[0]
        test_count = df.loc[df["Split"] == "Testing", "Samples"].values[0]

        self.assertEqual(train_count, int(total * TRAIN_SPLIT))
        self.assertAlmostEqual(val_count / total, VAL_SPLIT, delta=0.01)
        self.assertAlmostEqual(test_count / total, TEST_SPLIT, delta=0.01)


if __name__ == "__main__":
    unittest.main()
