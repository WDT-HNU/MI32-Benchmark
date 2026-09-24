import tempfile
import unittest
from pathlib import Path

import torch

from mi3_eegnet import EEGNet, sha256_file, snapshot_executed_source, validate_eegnet_structure


class EEGNetStructureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260817)
        self.model = EEGNet(
            n_chans=32,
            n_times=750,
            n_classes=3,
            dropout=0.5,
            f1=8,
            d=2,
            kern_length=64,
            separable_kernel=16,
            pool1=4,
            pool2=8,
        )

    def test_official_aligned_mi32_structure(self):
        report = validate_eegnet_structure(self.model, batch_size=2)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["trainable_parameters"], 2723)
        self.assertEqual(report["synthetic_input_shape"], [2, 32, 750])
        self.assertEqual(report["feature_map_shape"], [2, 16, 1, 23])
        self.assertEqual(report["classifier_in_features"], 368)
        self.assertEqual(report["output_shape"], [2, 3])
        self.assertEqual(report["spatial_groups"], 8)
        self.assertEqual(report["spatial_kernel"], [32, 1])
        self.assertEqual(report["separable_depthwise_groups"], 16)
        self.assertEqual(report["separable_kernel"], [1, 16])
        self.assertEqual(report["linear_layers"], 1)

    def test_forward_validates_canonical_input_shape(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 31, 750))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 32, 749))

    def test_batchnorm_and_dropout_defaults(self):
        batch_norms = [
            module for module in self.model.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ]
        self.assertEqual(len(batch_norms), 3)
        for batch_norm in batch_norms:
            self.assertEqual(batch_norm.eps, 1e-3)
            self.assertEqual(batch_norm.momentum, 0.01)
            self.assertTrue(torch.equal(batch_norm.weight, torch.ones_like(batch_norm.weight)))
            self.assertTrue(torch.equal(batch_norm.bias, torch.zeros_like(batch_norm.bias)))
        dropouts = [module for module in self.model.modules() if isinstance(module, torch.nn.Dropout)]
        self.assertEqual(len(dropouts), 2)
        self.assertTrue(all(module.p == 0.5 for module in dropouts))

    def test_temporal_settings_are_explicit_without_changing_defaults(self):
        self.assertEqual(self.model.kern_length, 64)
        self.assertEqual(self.model.separable_kernel, 16)
        self.assertEqual(self.model.pool1, 4)
        self.assertEqual(self.model.pool2, 8)
        self.assertEqual(self.model.temporal[1].kernel_size, (1, 64))
        self.assertEqual(self.model.separable[1].kernel_size, (1, 16))
        self.assertEqual(self.model.spatial[3].kernel_size, (1, 4))
        self.assertEqual(self.model.separable[5].kernel_size, (1, 8))

    def test_source_snapshot_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            report = snapshot_executed_source(Path(directory))
            snapshot = Path(report["source_snapshot"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(report["source_sha256"], sha256_file(snapshot))


if __name__ == "__main__":
    unittest.main()
