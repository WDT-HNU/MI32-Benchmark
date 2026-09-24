import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mi3_eegnet import MI3RawTrials


class MI3RawTrialsSemanticsTest(unittest.TestCase):
    def _build_dataset(self, root: Path, measured_mask_all_true: bool = True):
        (root / "data").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"subject_id": 0, "subject_name": "sub-0"}]).to_csv(
            root / "subjects.csv", index=False
        )
        x = np.arange(2 * 32 * 750, dtype=np.float32).reshape(2, 32, 750)
        y = np.array([0, 2], dtype=np.int64)
        channel_mask = np.ones((2, 32), dtype=bool)
        measured_mask = np.ones((2, 32), dtype=bool)
        if not measured_mask_all_true:
            measured_mask[0, 0] = False
        np.savez_compressed(
            root / "data" / "sub-0.npz",
            X=x,
            y=y,
            channel_mask=channel_mask,
            measured_mask=measured_mask,
        )

    def test_loader_preserves_raw_values_and_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_dataset(root)
            dataset = MI3RawTrials(root, [0], preload=False)
            x, y, sid = dataset[0]
            self.assertEqual(int(y), 0)
            self.assertEqual(int(sid), 0)
            expected = np.arange(32 * 750, dtype=np.float32).reshape(32, 750)
            self.assertTrue(np.array_equal(x.numpy(), expected))

    def test_loader_preserves_raw_values_with_partial_measured_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_dataset(root, measured_mask_all_true=False)
            dataset = MI3RawTrials(root, [0], preload=False)
            x, y, sid = dataset[0]
            self.assertEqual(int(y), 0)
            self.assertEqual(int(sid), 0)
            expected = np.arange(32 * 750, dtype=np.float32).reshape(32, 750)
            self.assertTrue(np.array_equal(x.numpy(), expected))


if __name__ == "__main__":
    unittest.main()
