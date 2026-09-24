import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]


def find_dataset_root():
    candidates = []
    for env_name in ("MI32_DATA_ROOT", "DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.extend([
        Path(__file__).resolve().parents[1] / "datasets" / "mi32" / "full",
        Path("/root/autodl-tmp/datasets/MI32_COMMON32_V4"),
    ])
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "data").is_dir():
            return candidate
    raise unittest.SkipTest("MI-32 dataset root not available")


class MI32DataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = find_dataset_root()
        cls.config = json.loads((cls.dataset_root / "config.json").read_text(encoding="utf-8"))
        cls.subjects = pd.read_csv(cls.dataset_root / "subjects.csv")
        cls.splits = pd.read_csv(cls.dataset_root / "splits.csv")

    def test_metadata_declares_the_frozen_mi32_contract(self):
        self.assertEqual(self.config["version"], "4.0-common32")
        self.assertEqual(self.config["n_channels"], 32)
        self.assertEqual(self.config["n_samples"], 750)
        self.assertEqual(
            [name.casefold() for name in self.config["channel_order"]],
            [name.casefold() for name in EXPECTED_CHANNEL_ORDER],
        )
        self.assertTrue(self.config["channel_mask"])
        self.assertTrue(self.config["measured_mask"])
        self.assertFalse(self.config["zero_fill"])
        self.assertIn("channel_mask_semantics", self.config)
        self.assertIn("measured_mask_semantics", self.config)
        self.assertEqual(self.config["quality_flags_file"], "quality_flags.csv")
        self.assertIn("hands/both_hands/both_hand events excluded", self.config["label_boundary_event_policy"])

    def test_npz_and_loader_preserve_interpolated_channels(self):
        from mi3_eegnet import MI3RawTrials

        first_subject = int(self.subjects.iloc[0]["subject_id"])
        subject_name = self.subjects.loc[self.subjects["subject_id"] == first_subject, "subject_name"].iloc[0]
        subject_path = self.dataset_root / "data" / f"{subject_name}.npz"

        with np.load(subject_path, allow_pickle=False) as z:
            self.assertIn("X", z.files)
            self.assertIn("y", z.files)
            self.assertIn("channel_mask", z.files)
            self.assertIn("measured_mask", z.files)
            x = z["X"]
            y = z["y"]
            channel_mask = z["channel_mask"].astype(bool)
            measured_mask = z["measured_mask"].astype(bool)

        self.assertEqual(x.shape[1:], (32, 750))
        self.assertEqual(channel_mask.shape, measured_mask.shape)
        self.assertEqual(channel_mask.shape[1], 32)
        self.assertTrue(np.isfinite(x).all())
        trial_mask = measured_mask[0]
        self.assertTrue(channel_mask[0].all())
        if (~trial_mask).any():
            interpolated = x[0, ~trial_mask, :]
            self.assertTrue(np.isfinite(interpolated).all())
            self.assertGreater(float(np.abs(interpolated).sum()), 0.0)

        dataset = MI3RawTrials(self.dataset_root, [first_subject], max_trials_per_subject=1, preload=False)
        trial_index = int(dataset.selected[first_subject][0])
        loaded_x, loaded_y, loaded_sid = dataset[0]
        self.assertEqual(int(loaded_sid), first_subject)
        self.assertEqual(int(loaded_y), int(y[trial_index]))
        self.assertEqual(loaded_x.shape, (32, 750))
        self.assertTrue(np.array_equal(loaded_x.numpy(), x[trial_index].astype(np.float32, copy=False)))

    def test_subject_splits_do_not_overlap(self):
        self.assertFalse(self.splits.duplicated(["dataset", "subject_id"]).any())
        self.assertEqual(
            set(self.splits["subject_id"].astype(int)),
            set(self.subjects["subject_id"].astype(int)),
        )
        test = set(self.splits[self.splits["fold"] == 0]["subject_id"].astype(int))
        val = set(self.splits[self.splits["fold"] == 1]["subject_id"].astype(int)) - test
        train = set(self.subjects["subject_id"].astype(int)) - test - val
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertTrue(train and val and test)

    def test_quality_review_flags_are_self_contained(self):
        flags = pd.read_csv(self.dataset_root / self.config["quality_flags_file"])
        trials = pd.read_csv(self.dataset_root / "trials.csv", usecols=["trial_id", "label"])
        self.assertEqual(len(flags), 501)
        self.assertEqual(flags["trial_id"].nunique(), 501)
        self.assertEqual(set(flags["reason"]), {"robust_extreme_review"})
        self.assertTrue((flags["flat_channels"] == 0).all())
        self.assertTrue(np.isfinite(flags[["peak_abs_v", "rms_v"]].to_numpy()).all())
        mapped = flags.merge(trials, on="trial_id", validate="one_to_one", suffixes=("_flag", "_trial"))
        self.assertEqual(len(mapped), 501)
        self.assertTrue((mapped["label_flag"] == mapped["label_trial"]).all())


if __name__ == "__main__":
    unittest.main()
