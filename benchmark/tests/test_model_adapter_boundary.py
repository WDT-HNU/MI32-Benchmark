"""Architecture boundary tests for model adapters versus dataset adapters."""

from __future__ import annotations

import inspect
import unittest

from model_adapters.base import ModelAdapter
from model_adapters.registry import get_adapter, list_adapters


CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

NEEDS_CHANNEL_ORDER = {"tsception", "rgnn", "labram", "eegmamba", "codebrain", "uni_ntfm"}
FORBIDDEN_DATASET_OPERATIONS = (
    "set_eeg_reference",
    "interpolate_bads",
    "train_test_split",
    "canonical_label",
    "global_subject_id",
)


class ModelAdapterBoundaryTest(unittest.TestCase):
    def make_adapter(self, name):
        kwargs = {"channel_order": CHANNEL_ORDER} if name in NEEDS_CHANNEL_ORDER else {}
        return get_adapter(name, **kwargs)

    def test_all_selected_models_use_explicit_model_adapter_type(self):
        self.assertEqual(
            list_adapters(),
            ["codebrain", "eegconformer", "eegmamba", "eegnet", "labram", "rgnn", "tsception", "uni_ntfm"],
        )
        for name in list_adapters():
            with self.subTest(model=name):
                adapter = self.make_adapter(name)
                self.assertIsInstance(adapter, ModelAdapter)
                self.assertEqual(adapter.manifest()["adapter_kind"], "model_input_adapter")

    def test_model_adapters_accept_only_frozen_harmonized_mi32_contract(self):
        for name in list_adapters():
            with self.subTest(model=name):
                contract = self.make_adapter(name).input_contract
                self.assertEqual(contract["channels"], 32)
                self.assertEqual(contract["samples"], 750)
                self.assertEqual(contract["fs"], 250)
                self.assertEqual(contract["unit"], "V")

    def test_model_adapter_sources_do_not_reinterpret_dataset_semantics(self):
        for name in list_adapters():
            source = inspect.getsource(inspect.getmodule(type(self.make_adapter(name))))
            folded = source.casefold()
            with self.subTest(model=name):
                for forbidden in FORBIDDEN_DATASET_OPERATIONS:
                    self.assertNotIn(forbidden.casefold(), folded)


if __name__ == "__main__":
    unittest.main()
