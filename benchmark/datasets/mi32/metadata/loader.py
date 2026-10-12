from pathlib import Path
import json
import numpy as np
import pandas as pd


class UEHFMI32Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.subjects = pd.read_csv(self.root / "subjects.csv")
        self.trials = pd.read_csv(self.root / "trials.csv", low_memory=False)
        self._index = self.trials.set_index("trial_id", drop=False)

    def load_subject(self, subject_id):
        row = self.subjects.loc[self.subjects.subject_id == int(subject_id)]
        if row.empty:
            raise KeyError(f"unknown subject_id: {subject_id}")
        name = row.iloc[0].subject_name
        with np.load(self.root / "data" / f"{name}.npz", allow_pickle=False) as z:
            return (
                z["X"].copy(),
                z["y"].copy(),
                z["channel_mask"].astype(bool).copy(),
                z["measured_mask"].astype(bool).copy(),
            )

    def load_trial(self, trial_id):
        if int(trial_id) not in self._index.index:
            raise KeyError(f"unknown trial_id: {trial_id}")
        row = self._index.loc[int(trial_id)]
        subject_id = int(row.subject_id)
        name = self.subjects.loc[self.subjects.subject_id == subject_id].iloc[0].subject_name
        idx = int(row.subject_trial_id)
        with np.load(self.root / "data" / f"{name}.npz", allow_pickle=False) as z:
            return (
                z["X"][idx].copy(),
                int(z["y"][idx]),
                z["channel_mask"][idx].astype(bool).copy(),
                z["measured_mask"][idx].astype(bool).copy(),
                subject_id,
            )

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, trial_id):
        return self.load_trial(trial_id)
