"""统一 Model Input Adapter 契约测试。

覆盖: manifest 完整性 / fit 仅训练集 / transform 确定性 / 消融三态 / 未 fit 拒绝。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_adapters.registry import get_adapter, list_adapters

SPLIT_HASH = "a" * 64  # 测试用假训练划分 hash (真实运行由数据层提供)

# MI-32 真实 32 通道序 (results_mi32_final/dataset/config.json)
CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

# 中线通道 (无数字后缀, 官方算法排除)
MIDLINE = {"Fz", "FCz", "Cz", "CPz", "Pz", "Oz"}


# 所有需要 channel_order 的 adapter (传统+foundation)
NEEDS_CHANNEL_ORDER = {"tsception", "rgnn", "labram", "eegmamba", "codebrain", "uni_ntfm"}


def make_adapter(name):
    if name in NEEDS_CHANNEL_ORDER:
        return get_adapter(name, channel_order=CHANNEL_ORDER)
    return get_adapter(name)


def fake_train_subjects(n_subjects=4, n_trials=8, seed=0):
    rng = np.random.default_rng(seed)
    for sid in range(n_subjects):
        X = rng.normal(0.0, 1e-4, size=(n_trials, 32, 750)).astype(np.float32)  # V 量级
        y = rng.integers(0, 3, size=n_trials)
        yield sid, X, y


class TestAdapterContract(unittest.TestCase):
    def test_all_adapters_registered(self):
        for name in ("eegnet", "tsception", "eegconformer", "rgnn",
                     "labram", "eegmamba", "codebrain", "uni_ntfm"):
            self.assertIn(name, list_adapters())

    def test_manifest_fields_complete(self):
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                m = a.manifest()
                for key in ("adapter_kind", "adapter", "model", "official_reference", "input_contract",
                            "output_contract", "source_sha256", "ablation_states"):
                    self.assertIn(key, m, f"{name} manifest 缺 {key}")
                self.assertEqual(m["adapter_kind"], "model_input_adapter")
                self.assertEqual(m["adapter"], name)
                self.assertEqual(len(m["source_sha256"]), 64)

    def test_fit_requires_train_split_and_validates(self):
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                subjects = list(fake_train_subjects())
                stats = a.fit(subjects, SPLIT_HASH)
                stats.validate()
                self.assertEqual(stats.source_split, "train")
                self.assertEqual(stats.split_hash, SPLIT_HASH)
                self.assertEqual(stats.n_subjects, 4)

    def test_transform_requires_fit_first(self):
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                x = np.zeros((2, 32, 750), np.float32)
                with self.assertRaises(RuntimeError):
                    a.transform(x)

    def test_transform_output_finite_and_shape_preserved(self):
        # 各 adapter 输出 shape 由其 output_contract 决定 (如 RGNN 是 [B,32,5] 特征)。
        expected = {
            "eegnet": (3, 32, 750),
            "tsception": (3, 32, 750),
            "eegconformer": (3, 32, 750),
            "rgnn": (3, 32, 5),
            "labram": (3, 32, 600),
            "eegmamba": (3, 32, 3, 200),
            "codebrain": (3, 32, 3, 200),
            "uni_ntfm": (3, 5, 24, 750),
        }
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                x = np.zeros((3, 32, 750), np.float32)
                out = a.transform(x)
                self.assertEqual(out.shape, expected[name], f"{name} 输出 shape 不符合契约")
                self.assertTrue(np.all(np.isfinite(out)))

    def test_transform_deterministic(self):
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                x = np.random.default_rng(1).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
                np.testing.assert_array_equal(a.transform(x), a.transform(x))

    def test_ablation_states_have_none_naive_official(self):
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                states = a.ablation_states()
                for key in ("none", "naive", "official"):
                    self.assertIn(key, states, f"{name} 消融缺 {key}")

    def test_naive_differs_from_official(self):
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                naive = a.ablation_states()["naive"]
                naive.fit(fake_train_subjects(), SPLIT_HASH)  # 统一生命周期: naive 也须 fit
                x = np.random.default_rng(2).normal(0, 1e-4, (4, 32, 750)).astype(np.float32)
                self.assertFalse(
                    np.allclose(a.transform(x), naive.transform(x)),
                    f"{name} naive 与 official 输出不应相同",
                )

    def test_manifest_saves_and_roundtrips(self):
        import json, tempfile
        for name in list_adapters():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                with tempfile.TemporaryDirectory() as td:
                    p = a.save_manifest(td)
                    m = json.loads(Path(p).read_text(encoding="utf-8"))
                    self.assertEqual(m["adapter"], name)
                    self.assertEqual(m["stats"]["source_split"], "train")
                    self.assertEqual(m["stats"]["n_subjects"], 4)


class TestTSceptionAdapterSpecific(unittest.TestCase):
    """TSception 专项: 26 通道选择 / 中线排除 / 只标准化选定通道。"""

    def test_selects_26_paired_channels_excluding_midline(self):
        a = make_adapter("tsception")
        names = a.ts_channel_order
        self.assertEqual(len(names), 26)
        for n in names:
            self.assertNotIn(n, MIDLINE, f"中线通道 {n} 不应被选中")
        # 所有选中通道必须有数字后缀 (成对通道)
        for n in names:
            self.assertTrue(any(c.isdigit() for c in n), f"{n} 无数字后缀")

    def test_official_pairing_odd_left_even_right(self):
        """奇数数字后缀=左半球先, 偶数=右半球后 (官方算法)。"""
        a = make_adapter("tsception")
        nums = [int("".join(c for c in n if c.isdigit())) for n in a.ts_channel_order]
        first_half = nums[: len(nums) // 2]
        second_half = nums[len(nums) // 2:]
        self.assertTrue(all(n % 2 == 1 for n in first_half), f"左半球应为奇数: {first_half}")
        self.assertTrue(all(n % 2 == 0 for n in second_half), f"右半球应为偶数: {second_half}")

    def test_transform_only_standardizes_selected_channels(self):
        a = make_adapter("tsception")
        a.fit(fake_train_subjects(), SPLIT_HASH)
        idx = a.channel_indices
        x = np.random.default_rng(3).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
        out = a.transform(x)
        # 未选中通道必须原样保留
        mask = np.ones(32, bool)
        mask[idx] = False
        np.testing.assert_array_equal(out[:, mask, :], x[:, mask, :])
        # 选中通道必须被改变 (标准化)
        self.assertFalse(np.allclose(out[:, idx, :], x[:, idx, :]))


class TestRGNNAdapterSpecific(unittest.TestCase):
    """RGNN 专项: 五频带特征 shape / 邻接全局对。"""

    def test_output_is_5_band_features(self):
        a = make_adapter("rgnn")
        a.fit(fake_train_subjects(), SPLIT_HASH)
        x = np.random.default_rng(4).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
        out = a.transform(x)
        self.assertEqual(out.shape, (2, 32, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_naive_linear_power_differs_from_official_log(self):
        a = make_adapter("rgnn")
        a.fit(fake_train_subjects(), SPLIT_HASH)
        naive = a.ablation_states()["naive"]
        naive.fit(fake_train_subjects(), SPLIT_HASH)
        x = np.random.default_rng(5).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
        self.assertFalse(np.allclose(a.transform(x), naive.transform(x)))

    def test_adjacency_requires_data_file(self):
        a = make_adapter("rgnn")
        a.fit(fake_train_subjects(), SPLIT_HASH)
        with self.assertRaises(RuntimeError):
            a.adjacency()  # 未提供 electrode_adjacency.npy 必须拒绝


class TestFoundationAdaptersSpecific(unittest.TestCase):
    """Foundation 专项: 重采样契约 / 区域映射 / padding。"""

    def test_labram_output_is_600_samples_at_microvolts(self):
        a = make_adapter("labram")
        a.fit(fake_train_subjects(), SPLIT_HASH)
        x = np.random.default_rng(6).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
        out = a.transform(x)
        self.assertEqual(out.shape, (2, 32, 600))
        # µV 量级: V(1e-4) * 1e6 ~= 1e2, 应远大于原始 V 量级
        self.assertGreater(np.abs(out).mean(), 1.0)

    def test_resample_produces_exact_600_and_patch_layout(self):
        for name in ("eegmamba", "codebrain"):
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                x = np.random.default_rng(7).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
                out = a.transform(x)
                self.assertEqual(out.shape, (2, 32, 3, 200))

    def test_official_unit_contracts_are_numerically_enforced(self):
        """锁死三种 foundation 输入单位，防止再次出现可运行但尺度错误。"""
        cases = {
            "labram": 1.0,      # 1e-6 V -> 1 µV
            "eegmamba": 0.01,   # 1e-6 V -> 0.01 µV/100
            "codebrain": 0.01,  # 1e-6 V -> 1 µV -> official downstream /100
        }
        x = np.full((2, 32, 750), 1e-6, dtype=np.float32)
        for name, expected in cases.items():
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                out = a.transform(x).reshape(2, 32, 600)
                observed = float(np.median(out[..., 20:-20]))
                self.assertTrue(
                    np.isclose(observed, expected, rtol=1e-3, atol=1e-6),
                    f"{name} 单位尺度错误: observed={observed}, expected={expected}",
                )

    def test_codebrain_naive_changes_only_resampling_method(self):
        """CodeBrain naive 必须保留 official 的 V->µV/100 单位合同。"""
        official = make_adapter("codebrain")
        official.fit(fake_train_subjects(), SPLIT_HASH)
        naive = official.ablation_states()["naive"]
        naive.fit(fake_train_subjects(), SPLIT_HASH)
        x = np.full((2, 32, 750), 1e-6, dtype=np.float32)
        official_out = official.transform(x).reshape(2, 32, 600)
        naive_out = naive.transform(x).reshape(2, 32, 600)
        self.assertTrue(
            np.isclose(float(np.median(official_out[..., 20:-20])), 0.01, rtol=1e-3),
            "CodeBrain official 单位尺度偏离 µV/100 合同",
        )
        self.assertTrue(
            np.isclose(float(np.median(naive_out[..., 20:-20])), 0.01, rtol=1e-3),
            "CodeBrain naive 未保留 official 的 µV/100 单位合同",
        )

    def test_naive_linear_resample_differs_from_official_sinc(self):
        for name in ("labram", "eegmamba", "codebrain"):
            with self.subTest(adapter=name):
                a = make_adapter(name)
                a.fit(fake_train_subjects(), SPLIT_HASH)
                naive = a.ablation_states()["naive"]
                naive.fit(fake_train_subjects(), SPLIT_HASH)
                x = np.random.default_rng(8).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
                self.assertFalse(np.allclose(a.transform(x), naive.transform(x)),
                                 f"{name} naive 与 official 输出不应相同")

    def test_uni_region_mapping_and_padding(self):
        a = make_adapter("uni_ntfm")
        a.fit(fake_train_subjects(), SPLIT_HASH)
        x = np.random.default_rng(9).normal(0, 1e-4, (2, 32, 750)).astype(np.float32)
        out = a.transform(x)
        self.assertEqual(out.shape, (2, 5, 24, 750))
        pad = a.padding_mask(2)
        self.assertEqual(pad.shape, (2, 5, 24))
        self.assertEqual(pad.dtype, np.bool_)
        # 至少有一个 padding 位置 (32 通道不可能填满 5x24=120 槽)
        self.assertTrue(pad.any())
        # padding 位置必须全零
        self.assertTrue(np.all(out[pad] == 0.0))


class TestEEGConformerAdapterSpecific(unittest.TestCase):
    """Conformer 专项: 全局标准化 = 训练集统计, 与逐trial不同。"""

    def test_global_stats_are_train_wide(self):
        a = make_adapter("eegconformer")
        a.fit(fake_train_subjects(n_subjects=3, n_trials=6), SPLIT_HASH)
        self.assertIn("mean", a.stats.values)
        self.assertIn("std", a.stats.values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
