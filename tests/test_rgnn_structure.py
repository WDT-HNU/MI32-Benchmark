import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mi3_eegnet
from model_adapters.rgnn import RGNNAdapter, RGNN_GLOBAL_PAIRS as ADAPTER_GLOBAL_PAIRS
from make_rgnn_adjacency import inverse_square_adjacency
from mi3_eegnet import (
    RGNN,
    RGNN_BANDS,
    rgnn_objective,
    sha256_file,
    snapshot_executed_source,
    validate_rgnn_structure,
)


def find_mi32_root():
    candidates = []
    if os.environ.get("RGNN_TEST_DATA"):
        candidates.append(Path(os.environ["RGNN_TEST_DATA"]))
    candidates.extend([
        Path(__file__).resolve().parents[1] / "datasets" / "mi32" / "metadata",
        Path("C:/Users/lenovo/Documents/Codex/results_mi32_final/dataset"),
        Path("G:/MI-3-32通道插值"),
        Path("/root/autodl-tmp/datasets/MI32_COMMON32_V4"),
    ])
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "electrode_adjacency.npy").is_file():
            return candidate
    raise RuntimeError("MI-32 config.json and electrode_adjacency.npy are required for RGNN audit")


class RGNNStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_root = find_mi32_root()
        cls.config = json.loads((cls.data_root / "config.json").read_text(encoding="utf-8"))
        cls.distance_adjacency = np.load(cls.data_root / "electrode_adjacency.npy")

    def setUp(self):
        torch.manual_seed(20260817)
        self.adapter = RGNNAdapter(
            self.config["channel_order"],
            distance_adjacency=self.distance_adjacency,
        )
        self.adapter.fit(
            [(0, np.zeros((1, 32, 750), np.float32), np.zeros(1, dtype=np.int64))],
            "0" * 64,
        )
        self.model = RGNN(
            torch.from_numpy(self.distance_adjacency),
            self.config["channel_order"],
            n_classes=3,
            hidden=64,
            k=2,
            dropout=0.5,
            sfreq=250,
            n_times=750,
            l1_alpha=1e-5,
            adapter=self.adapter,
        )

    def test_distance_formula_channel_order_and_density(self):
        xyz_cm = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        expected = np.array([
            [1.0, 1.0, 5.0 / 9.0],
            [1.0, 1.0, 0.5],
            [5.0 / 9.0, 0.5, 1.0],
        ])
        np.testing.assert_allclose(inverse_square_adjacency(xyz_cm, delta=5.0), expected)

        self.assertEqual(self.distance_adjacency.shape, (32, 32))
        np.testing.assert_allclose(self.distance_adjacency, self.distance_adjacency.T, atol=0, rtol=0)
        np.testing.assert_allclose(np.diag(self.distance_adjacency), np.ones(32))
        off_diagonal = ~np.eye(32, dtype=bool)
        self.assertAlmostEqual(float((self.distance_adjacency[off_diagonal] > 0.1).mean()), 0.1552419355)
        self.assertEqual(len(self.config["channel_order"]), 32)

    def test_learnable_symmetric_signed_adjacency(self):
        self.assertIsInstance(self.model.edge_values, nn.Parameter)
        self.assertTrue(self.model.edge_values.requires_grad)
        self.assertEqual(self.model.edge_values.numel(), 32 * 33 // 2)

        raw = self.model.raw_adjacency()
        self.assertTrue(torch.equal(raw, raw.T))
        self.assertTrue(torch.equal(raw.diagonal(), torch.ones(32)))
        expected_present = (
            ("Fp1", "Fp2"), ("AF3", "AF4"), ("FC5", "FC6"),
            ("CP5", "CP6"), ("O1", "O2"),
        )
        self.assertEqual(self.model.global_pairs_applied, expected_present)
        self.assertEqual(len(ADAPTER_GLOBAL_PAIRS), 9)
        for left, right in expected_present:
            i = self.model.channel_order.index(left)
            j = self.model.channel_order.index(right)
            self.assertAlmostEqual(
                raw[i, j].item(),
                self.model.distance_adjacency[i, j].item() - 1.0,
            )
            self.assertLess(raw[i, j].item(), 0.0)

    def test_abs_degree_k2_sum_pool_and_single_classifier(self):
        raw = self.model.raw_adjacency()
        degree = raw.abs().sum(1)
        manual_normalized = degree.rsqrt()[:, None] * raw * degree.rsqrt()[None, :]
        torch.testing.assert_close(self.model.normalized_adjacency(), manual_normalized)
        masked = self.model.normalized_adjacency(torch.tensor([[True] * 31 + [False]]))
        torch.testing.assert_close(masked, manual_normalized.unsqueeze(0))
        self.assertEqual(self.model.k, 2)
        self.assertIsInstance(self.model.classifier, nn.Linear)
        self.assertEqual((self.model.classifier.in_features, self.model.classifier.out_features), (64, 3))
        self.assertFalse(hasattr(self.model, "domain_classifier"))

        self.model.eval()
        x = torch.randn(2, 32, 750)
        node_features = self.adapter.transform(x)
        adjacency = self.model.normalized_adjacency(torch.zeros(2, 32, dtype=torch.bool))
        torch.testing.assert_close(adjacency, manual_normalized.unsqueeze(0).expand(2, -1, -1))
        propagated = node_features
        for _ in range(2):
            propagated = torch.einsum("bij,bjf->bif", adjacency, propagated)
        node_embeddings = torch.relu(self.model.node_projection(propagated))
        manual_output = self.model.classifier(node_embeddings.sum(1))
        torch.testing.assert_close(self.model(self.adapter.transform(x)), manual_output)

        x_zero = x.clone()
        x_zero[:, 0, :] = 0.0
        node_features_zero = self.adapter.transform(x_zero)
        propagated_zero = node_features_zero
        for _ in range(2):
            propagated_zero = torch.einsum("bij,bjf->bif", adjacency, propagated_zero)
        node_embeddings_zero = torch.relu(self.model.node_projection(propagated_zero))
        manual_zero_output = self.model.classifier(node_embeddings_zero.sum(1))
        torch.testing.assert_close(self.model(self.adapter.transform(x_zero)), manual_zero_output)

    def test_features_structure_objective_and_parameter_count(self):
        self.assertEqual(RGNN_BANDS, (
            ("delta", 1.0, 4.0),
            ("theta", 4.0, 8.0),
            ("alpha", 8.0, 13.0),
            ("beta", 13.0, 30.0),
            ("gamma", 30.0, 50.0),
        ))
        report = validate_rgnn_structure(self.model, batch_size=2)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["identity"], "RGNN backbone (without target-domain NodeDAT/EmotionDL)")
        self.assertEqual(report["node_feature_shape"], [2, 32, 5])
        self.assertEqual(report["pooling"], "sum")
        self.assertEqual(report["propagation_order"], 2)
        self.assertEqual(report["negative_edge_count_undirected"], 5)
        self.assertEqual(report["output_shape"], [2, 3])
        self.assertEqual(report["trainable_parameters"], 1107)

        self.model.eval()
        logits = self.model(self.adapter.transform(torch.randn(2, 32, 750)))
        targets = torch.tensor([0, 1])
        criterion = nn.CrossEntropyLoss()
        objective = rgnn_objective(self.model, logits, targets, criterion)
        expected = criterion(logits, targets) + 1e-5 * self.model.edge_values.abs().sum()
        torch.testing.assert_close(objective, expected)
        self.assertIn("rgnn_objective", inspect.getsource(mi3_eegnet.main))

    def test_source_snapshot_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            report = snapshot_executed_source(Path(directory))
            snapshot = Path(report["source_snapshot"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(report["source_sha256"], sha256_file(snapshot))


if __name__ == "__main__":
    unittest.main()
