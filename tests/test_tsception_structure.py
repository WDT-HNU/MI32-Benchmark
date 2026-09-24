import inspect
import unittest

import torch
import torch.nn as nn

from mi3_eegnet import (
    TSception,
    generate_TS_channel_order,
    validate_tsception_structure,
)


MI32_CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

OFFICIAL_MI32_TS_ORDER = [
    "Fp1", "AF3", "F7", "F3", "FC5", "FC3", "T7", "C3", "CP5",
    "CP3", "P7", "P3", "O1", "Fp2", "AF4", "F8", "F4", "FC6",
    "FC4", "T8", "C4", "CP6", "CP4", "P8", "P4", "O2",
]

OFFICIAL_MI32_INDICES = [
    0, 2, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29,
    1, 3, 8, 7, 13, 12, 18, 17, 23, 22, 28, 27, 31,
]


def official_generate_TS_channel_order(original_order):
    """Verbatim algorithmic behavior of official utils.py at commit 9efd666."""
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        chan_name_len = len(channel)
        k = 0
        for character in [*channel[:]]:
            if character.isdigit():
                k += 1
        if k != 0:
            chan_name.append(channel[:chan_name_len - k])
            chan_num.append(int(channel[chan_name_len - k:]))
            chan_final.append(channel)
    chan_pair = []
    for index, number in enumerate(chan_num):
        if number % 2 == 0:
            chan_pair.append(chan_name[index] + str(number - 1))
        else:
            chan_pair.append(chan_name[index] + str(number + 1))
    chan_no_duplicate = []
    [
        chan_no_duplicate.extend([channel, chan_pair[index]])
        for index, channel in enumerate(chan_final)
        if channel not in chan_no_duplicate
    ]
    return chan_no_duplicate[0::2] + chan_no_duplicate[1::2]


class OfficialTSceptionV2(nn.Module):
    """Reference transcription of official code/networks.py at commit 9efd666."""

    @staticmethod
    def conv_block(in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(self):
        super().__init__()
        num_t, num_s = 15, 15
        self.t1 = self.conv_block(1, num_t, (1, 125), 1, 8)
        self.t2 = self.conv_block(1, num_t, (1, 62), 1, 8)
        self.t3 = self.conv_block(1, num_t, (1, 31), 1, 8)
        self.s1 = self.conv_block(num_t, num_s, (26, 1), 1, 2)
        self.s2 = self.conv_block(num_t, num_s, (13, 1), (13, 1), 2)
        self.fusion = self.conv_block(num_s, num_s, (3, 1), 1, 4)
        self.bn_t = nn.BatchNorm2d(num_t)
        self.bn_s = nn.BatchNorm2d(num_s)
        self.bn_fusion = nn.BatchNorm2d(num_s)
        self.fc = nn.Sequential(
            nn.Linear(num_s, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        out = torch.cat((self.t1(x), self.t2(x), self.t3(x)), dim=-1)
        out = self.bn_t(out)
        out = torch.cat((self.s1(out), self.s2(out)), dim=2)
        out = self.bn_s(out)
        out = self.bn_fusion(self.fusion(out))
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        return self.fc(out)


def block_signature(block):
    convolution, activation, pooling = block
    return {
        "convolution": (
            convolution.in_channels,
            convolution.out_channels,
            convolution.kernel_size,
            convolution.stride,
            convolution.padding,
            convolution.bias is not None,
        ),
        "activation": (type(activation), activation.negative_slope, activation.inplace),
        "pooling": (pooling.kernel_size, pooling.stride, pooling.padding),
    }


class TSceptionStructureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260817)
        self.model = TSception(
            MI32_CHANNEL_ORDER,
            n_classes=3,
            sampling_rate=250,
            num_t=15,
            num_s=15,
            hidden=32,
            dropout=0.5,
        )

    def test_official_channel_generator_and_mi32_mapping(self):
        reference = official_generate_TS_channel_order(MI32_CHANNEL_ORDER)
        self.assertEqual(reference, OFFICIAL_MI32_TS_ORDER)
        self.assertEqual(generate_TS_channel_order(MI32_CHANNEL_ORDER), reference)
        self.assertEqual(list(self.model.ts_channel_order), OFFICIAL_MI32_TS_ORDER)
        self.assertEqual(self.model.channel_indices.tolist(), OFFICIAL_MI32_INDICES)

        left = self.model.ts_channel_order[:13]
        right = self.model.ts_channel_order[13:]
        for left_channel, right_channel in zip(left, right):
            left_prefix = left_channel.rstrip("0123456789")
            left_number = int(left_channel[len(left_prefix):])
            self.assertEqual(right_channel, left_prefix + str(left_number + 1))
        self.assertTrue(set(("Fz", "FCz", "Cz", "CPz", "Pz", "Oz")).isdisjoint(self.model.ts_channel_order))

    def test_model_matches_official_v2_layer_by_layer(self):
        reference = OfficialTSceptionV2()
        for name in ("t1", "t2", "t3", "s1", "s2", "fusion"):
            self.assertEqual(block_signature(getattr(self.model, name)), block_signature(getattr(reference, name)))

        for name in ("bn_t", "bn_s", "bn_fusion"):
            current, official = getattr(self.model, name), getattr(reference, name)
            self.assertEqual(
                (current.num_features, current.eps, current.momentum, current.affine, current.track_running_stats),
                (official.num_features, official.eps, official.momentum, official.affine, official.track_running_stats),
            )
        self.assertEqual(str(self.model.fc), str(reference.fc))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            sum(parameter.numel() for parameter in reference.parameters()),
        )

    def test_mi32_shapes_parameter_count_and_forward(self):
        report = validate_tsception_structure(self.model, n_times=750, batch_size=2)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["variant"], "TSception-v2 (TAFFC 2022/2023)")
        self.assertEqual(report["temporal_kernels"], [125, 62, 31])
        self.assertEqual(report["selected_channel_count"], 26)
        self.assertEqual(report["trainable_parameters"], 13511)
        self.assertEqual(report["feature_map_shape"], [2, 15, 1, 31])
        self.assertEqual(report["output_shape"], [2, 3])
        self.assertEqual(report["feature_shapes"]["bn_t"], [2, 15, 26, 254])
        self.assertEqual(report["feature_shapes"]["bn_s"], [2, 15, 3, 127])
        self.assertEqual(report["feature_shapes"]["bn_fusion"], [2, 15, 1, 31])
        self.assertEqual(report["official_paper_url"], "https://doi.org/10.1109/TAFFC.2022.3169001")
        self.assertEqual(report["official_repository_url"], "https://github.com/yi-ding-cs/TSception")

    def test_forward_validates_canonical_mi32_shape(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 31, 750))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 32, 749))

    def test_official_pytorch_initialization_policy_is_unchanged(self):
        source = inspect.getsource(TSception)
        self.assertNotIn("reset_parameters(", source)
        self.assertNotIn("nn.init.", source)
        for batch_norm in (self.model.bn_t, self.model.bn_s, self.model.bn_fusion):
            self.assertTrue(torch.equal(batch_norm.weight, torch.ones_like(batch_norm.weight)))
            self.assertTrue(torch.equal(batch_norm.bias, torch.zeros_like(batch_norm.bias)))
            self.assertTrue(torch.equal(batch_norm.running_mean, torch.zeros_like(batch_norm.running_mean)))
            self.assertTrue(torch.equal(batch_norm.running_var, torch.ones_like(batch_norm.running_var)))


if __name__ == "__main__":
    unittest.main()
