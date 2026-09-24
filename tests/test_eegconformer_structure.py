import inspect
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import mi3_eegnet
from mi3_eegnet import (
    EEGConformer,
    EEGConformerFeedForward,
    EEGConformerMultiHeadAttention,
    EEGConformerReduceMean,
    expected_eegconformer_trainable_parameters,
    sha256_file,
    snapshot_executed_source,
    validate_eegconformer_structure,
)


class EEGConformerStructureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260817)
        self.model = EEGConformer(
            n_chans=32,
            n_times=750,
            n_classes=3,
            dropout=0.5,
            emb_size=40,
            depth=6,
            heads=10,
            expansion=4,
        )

    def test_patch_embedding_and_mi32_tokens(self):
        shallow = self.model.patch.shallownet
        temporal, spatial, batch_norm, activation, pooling, dropout = shallow
        self.assertEqual(
            (
                temporal.in_channels,
                temporal.out_channels,
                temporal.kernel_size,
                temporal.stride,
                temporal.padding,
                temporal.bias is not None,
            ),
            (1, 40, (1, 25), (1, 1), (0, 0), True),
        )
        self.assertEqual(
            (
                spatial.in_channels,
                spatial.out_channels,
                spatial.kernel_size,
                spatial.stride,
                spatial.padding,
                spatial.bias is not None,
            ),
            (40, 40, (32, 1), (1, 1), (0, 0), True),
        )
        self.assertIsInstance(batch_norm, nn.BatchNorm2d)
        self.assertEqual((batch_norm.eps, batch_norm.momentum), (1e-5, 0.1))
        self.assertEqual(batch_norm.num_batches_tracked.item(), 0)
        torch.testing.assert_close(batch_norm.running_mean, torch.zeros(40))
        torch.testing.assert_close(batch_norm.running_var, torch.ones(40))
        self.assertIsInstance(activation, nn.ELU)
        self.assertEqual((pooling.kernel_size, pooling.stride), ((1, 75), (1, 15)))
        self.assertEqual(dropout.p, 0.5)

        projection = self.model.patch.projection
        self.assertEqual(
            (
                projection.in_channels,
                projection.out_channels,
                projection.kernel_size,
                projection.stride,
                projection.bias is not None,
            ),
            (40, 40, (1, 1), (1, 1), True),
        )
        self.model.eval()
        with torch.no_grad():
            tokens = self.model.patch(torch.zeros(2, 1, 32, 750))
        self.assertEqual(self.model.tokens, 44)
        self.assertEqual(tuple(tokens.shape), (2, 44, 40))

    def test_official_attention_encoder_and_initialization(self):
        self.assertEqual(len(self.model.transformer), 6)
        self.assertEqual(len({id(block) for block in self.model.transformer}), 6)
        first_query = self.model.transformer[0].attention.function[1].queries.weight
        second_query = self.model.transformer[1].attention.function[1].queries.weight
        self.assertFalse(torch.equal(first_query, second_query))

        for block in self.model.transformer:
            attention_path = block.attention.function
            self.assertIsInstance(attention_path[0], nn.LayerNorm)
            self.assertIsInstance(attention_path[1], EEGConformerMultiHeadAttention)
            self.assertIsInstance(attention_path[2], nn.Dropout)
            attention = attention_path[1]
            self.assertEqual((attention.emb_size, attention.num_heads, attention.head_dim), (40, 10, 4))
            self.assertEqual(attention.scaling, math.sqrt(40))
            self.assertNotEqual(attention.scaling, math.sqrt(4))
            self.assertEqual((attention.att_drop.p, attention_path[2].p), (0.5, 0.5))

            feed_forward_path = block.feed_forward.function
            self.assertIsInstance(feed_forward_path[0], nn.LayerNorm)
            self.assertIsInstance(feed_forward_path[1], EEGConformerFeedForward)
            self.assertIsInstance(feed_forward_path[2], nn.Dropout)
            feed_forward = feed_forward_path[1]
            self.assertEqual((feed_forward[0].in_features, feed_forward[0].out_features), (40, 160))
            self.assertIsInstance(feed_forward[1], nn.GELU)
            self.assertEqual(feed_forward[2].p, 0.5)
            self.assertEqual((feed_forward[3].in_features, feed_forward[3].out_features), (160, 40))

        self.assertFalse(any(isinstance(module, nn.TransformerEncoder) for module in self.model.modules()))
        state_names = [name.casefold() for name, _ in self.model.named_parameters()]
        state_names.extend(name.casefold() for name, _ in self.model.named_buffers())
        self.assertFalse(any("pos" in name or "cls_token" in name for name in state_names))

    def test_active_flatten_classifier_and_inactive_clshead(self):
        clshead = self.model.classifier.clshead
        fc = self.model.classifier.fc
        self.assertIsInstance(clshead[0], EEGConformerReduceMean)
        self.assertIsInstance(clshead[1], nn.LayerNorm)
        self.assertEqual((clshead[2].in_features, clshead[2].out_features), (40, 3))
        self.assertEqual(
            (
                fc[0].in_features,
                fc[0].out_features,
                fc[2].p,
                fc[3].in_features,
                fc[3].out_features,
                fc[5].p,
                fc[6].in_features,
                fc[6].out_features,
            ),
            (1760, 256, 0.5, 256, 32, 0.3, 32, 3),
        )

        calls = {"clshead": 0, "fc": 0}
        cls_hook = clshead.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("clshead", calls["clshead"] + 1)
        )
        fc_hook = fc.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("fc", calls["fc"] + 1)
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.zeros(2, 32, 750))
        cls_hook.remove()
        fc_hook.remove()
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertEqual(calls, {"clshead": 0, "fc": 1})

    def test_parameter_count_validator_and_training_entry(self):
        self.assertEqual(expected_eegconformer_trainable_parameters(self.model), 631662)
        self.assertEqual(sum(parameter.numel() for parameter in self.model.parameters()), 631662)
        report = validate_eegconformer_structure(self.model, batch_size=2)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["patch_shape"], [2, 44, 40])
        self.assertEqual(report["output_shape"], [2, 3])
        self.assertEqual(report["attention_scaling"], "sqrt(embedding_size)")
        self.assertEqual(report["official_paper_url"], "https://arxiv.org/abs/2106.07336")
        self.assertEqual(report["official_repository_url"], "https://github.com/eeyhsong/EEG-Conformer")
        self.assertEqual(report["active_classifier"], "Flatten(1760)-256-32-3")
        self.assertEqual(report["inactive_clshead_calls"], 0)
        self.assertEqual(report["trainable_parameters"], 631662)

        main_source = inspect.getsource(mi3_eegnet.main)
        self.assertIn("validate_eegconformer_structure(model)", main_source)
        self.assertIn('"eegconformer_structure_audit"', main_source)
        self.assertIn("9ae149ba62487ceae723277d13adac27837113d2", main_source)

    def test_source_snapshot_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            report = snapshot_executed_source(Path(directory))
            snapshot = Path(report["source_snapshot"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(report["source_sha256"], sha256_file(snapshot))


if __name__ == "__main__":
    unittest.main()
