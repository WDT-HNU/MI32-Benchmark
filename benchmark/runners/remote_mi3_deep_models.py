import argparse
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, Sampler


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MI3RawTrials(Dataset):
    """MI-3 depth view: zero-filled 128-channel trials plus implicit zero mask."""

    def __init__(self, root, subject_ids, max_trials_per_subject=0, preload=False):
        self.root = Path(root)
        self.subjects = pd.read_csv(self.root / "subjects.csv").set_index("subject_id")
        self.items = []
        self.by_subject = []
        self.selected = {}
        self.cache = OrderedDict()
        self.preload = preload
        self.subject_ids = sorted(map(int, subject_ids))
        for sid in self.subject_ids:
            _, labels, _ = self._read_subject_raw(sid, signals=False)
            n = len(labels)
            if max_trials_per_subject and n > max_trials_per_subject:
                # Preserve the agreed 1:1:2 class ratio inside every tuning subject.
                q = max_trials_per_subject // 4
                quotas = [q, q, max_trials_per_subject - 2 * q]
                chosen_parts = []
                for label, quota in enumerate(quotas):
                    candidates = np.flatnonzero(labels == label)
                    if len(candidates) < quota:
                        raise RuntimeError(f"subject {sid}: insufficient class {label} trials")
                    positions = np.linspace(0, len(candidates) - 1, quota, dtype=int)
                    chosen_parts.append(candidates[positions])
                chosen = np.concatenate(chosen_parts)
            else:
                chosen = np.arange(n)
            self.selected[sid] = chosen
            group = []
            for local_trial, _ in enumerate(chosen):
                group.append(len(self.items))
                self.items.append((sid, local_trial))
            self.by_subject.append(group)
        if preload:
            for sid in self.subject_ids:
                self.cache[sid] = self._load_and_normalize(sid)

    def _path(self, sid):
        name = self.subjects.loc[sid, "subject_name"]
        return self.root / "data" / f"{name}.npz"

    def _read_subject_raw(self, sid, signals=True):
        with np.load(self._path(sid), allow_pickle=False) as z:
            x = z["X"].copy() if signals else None
            return x, z["y"].astype(np.int64).copy(), z["channel_mask"].astype(bool).copy()

    def _load_and_normalize(self, sid):
        x, y, mask = self._read_subject_raw(sid, signals=True)
        chosen = self.selected[sid]
        x, y, mask = x[chosen], y[chosen], mask[chosen]
        x = x.astype(np.float32, copy=False)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / np.maximum(std, 1e-6)
        x[~mask] = 0.0
        return x, y, mask

    def _load(self, sid):
        if sid in self.cache:
            return self.cache[sid]
        value = self._load_and_normalize(sid)
        if not self.preload:
            self.cache.clear()
        self.cache[sid] = value
        return value

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sid, tid = self.items[index]
        x, y, _ = self._load(sid)
        return torch.from_numpy(x[tid]), int(y[tid]), sid


class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.groups = dataset.by_subject
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        groups = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(groups)
        for group_id in groups:
            ids = list(self.groups[group_id])
            if self.shuffle:
                random.shuffle(ids)
            for start in range(0, len(ids), self.batch_size):
                yield ids[start:start + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups)


class EEGNet(nn.Module):
    """EEGNet-8,2 architecture adapted from Lawhern et al. for 128 x 750 input."""

    def __init__(self, n_chans=128, n_times=750, n_classes=3, dropout=0.5, f1=8, d=2):
        super().__init__()
        f2 = f1 * d
        self.temporal = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, f1, (1, 64), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f2, (n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(f2, f2, (1, 16), groups=f2, bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            features = self.separable(self.spatial(self.temporal(dummy))).numel()
        self.classifier = nn.Linear(features, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return self.classifier(x.flatten(1))

    def constrain(self):
        # Max-norm constraints used by the original EEGNet formulation.
        with torch.no_grad():
            self.spatial[0].weight.renorm_(p=2, dim=0, maxnorm=1.0)
            self.classifier.weight.renorm_(p=2, dim=0, maxnorm=0.25)


class TSception(nn.Module):
    """Official TSception-v2 topology with a symmetric channel subset."""

    def __init__(self, channel_order, n_classes=3, sampling_rate=250, num_t=15, num_s=15,
                 hidden=32, dropout=0.5):
        super().__init__()
        pairs = [
            ("Fp1", "Fp2"), ("AF3", "AF4"), ("AF7", "AF8"),
            ("F1", "F2"), ("F3", "F4"), ("F5", "F6"), ("F7", "F8"),
            ("FC1", "FC2"), ("FC3", "FC4"), ("FC5", "FC6"),
            ("FT7", "FT8"), ("FT9", "FT10"),
            ("C1", "C2"), ("C3", "C4"), ("C5", "C6"), ("T7", "T8"), ("T9", "T10"),
            ("CP1", "CP2"), ("CP3", "CP4"), ("CP5", "CP6"), ("TP7", "TP8"),
            ("P1", "P2"), ("P3", "P4"), ("P5", "P6"), ("P7", "P8"), ("P9", "P10"),
            ("PO3", "PO4"), ("PO5", "PO6"), ("PO7", "PO8"), ("PO9", "PO10"),
            ("O1", "O2"), ("I1", "I2"),
        ]
        lookup = {name.casefold(): i for i, name in enumerate(channel_order)}
        valid_pairs = [(a, b) for a, b in pairs if a.casefold() in lookup and b.casefold() in lookup]
        ordered = [a for a, _ in valid_pairs] + [b for _, b in valid_pairs]
        self.register_buffer("channel_indices", torch.tensor([lookup[x.casefold()] for x in ordered]))
        n_chans = len(ordered)
        if n_chans < 2 or n_chans % 2:
            raise RuntimeError("TSception requires paired left/right channels")

        def block(in_chans, out_chans, kernel, stride, pool):
            return nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
            )

        self.t1 = block(1, num_t, (1, int(0.5 * sampling_rate)), 1, 8)
        self.t2 = block(1, num_t, (1, int(0.25 * sampling_rate)), 1, 8)
        self.t3 = block(1, num_t, (1, int(0.125 * sampling_rate)), 1, 8)
        self.bn_t = nn.BatchNorm2d(num_t)
        self.s1 = block(num_t, num_s, (n_chans, 1), 1, 2)
        self.s2 = block(num_t, num_s, (n_chans // 2, 1), (n_chans // 2, 1), 2)
        self.bn_s = nn.BatchNorm2d(num_s)
        self.fusion = block(num_s, num_s, (3, 1), 1, 4)
        self.bn_fusion = nn.BatchNorm2d(num_s)
        self.fc = nn.Sequential(
            nn.Linear(num_s, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        x = x.index_select(1, self.channel_indices).unsqueeze(1)
        out = torch.cat([self.t1(x), self.t2(x), self.t3(x)], dim=-1)
        out = self.bn_t(out)
        out = torch.cat([self.s1(out), self.s2(out)], dim=2)
        out = self.bn_s(out)
        out = self.bn_fusion(self.fusion(out))
        out = out.mean(dim=-1).squeeze(-1)
        return self.fc(out)


class EEGConformer(nn.Module):
    """Official EEG-Conformer convolutional tokenizer and six-layer Transformer."""

    def __init__(self, n_chans=128, n_times=750, n_classes=3, dropout=0.5,
                 emb_size=40, depth=6, heads=10):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 25), (1, 1)),
            nn.Conv2d(emb_size, emb_size, (n_chans, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(emb_size))
        with torch.no_grad():
            tokens = self.patch(torch.zeros(1, 1, n_chans, n_times)).shape[-1]
        self.fc = nn.Sequential(
            nn.Linear(tokens * emb_size, 256), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        x = self.transformer(x)
        return self.fc(x.flatten(1))


class RGNN(nn.Module):
    """RGNN/SGC with five-band log-power (DE-equivalent) node features."""

    def __init__(self, initial_adjacency, n_classes=3, hidden=64, k=2, dropout=0.5,
                 sfreq=250, n_times=750):
        super().__init__()
        n_chans = initial_adjacency.shape[0]
        self.n_chans = n_chans
        frequencies = torch.fft.rfftfreq(n_times, d=1.0 / sfreq)
        bands = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 50))
        band_masks = torch.stack([
            (frequencies >= low) & (frequencies < high) for low, high in bands
        ])
        self.register_buffer("band_masks", band_masks)
        self.node_projection = nn.Linear(len(bands), hidden)
        initial = initial_adjacency.float()
        tril = torch.tril_indices(n_chans, n_chans)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])
        self.edge_values = nn.Parameter(initial[tril[0], tril[1]])
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def adjacency(self, valid):
        a = torch.zeros(self.n_chans, self.n_chans, device=self.edge_values.device, dtype=self.edge_values.dtype)
        a[self.tril_rows, self.tril_cols] = self.edge_values
        a = a + a.transpose(0, 1) - torch.diag(a.diagonal())
        node_mask = valid.to(a.dtype)
        a = a[None] * node_mask[:, :, None] * node_mask[:, None, :]
        degree = a.abs().sum(2).clamp_min(1e-6)
        inv = degree.rsqrt()
        return inv[:, :, None] * a * inv[:, None, :]

    def forward(self, x):
        valid = x.abs().sum(-1) > 0
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        features = torch.stack([
            power[..., mask].mean(-1).clamp_min(1e-8).log()
            for mask in self.band_masks
        ], dim=-1)
        features = self.node_projection(features)
        features = features * valid[..., None]
        a = self.adjacency(valid)
        for _ in range(self.k):
            features = torch.einsum("bij,bjh->bih", a, features)
        features = torch.relu(features).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return self.classifier(self.dropout(features))

    def regularization(self):
        return self.edge_values.abs().sum()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, subjects = [], [], []
    loss_sum = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    for x, y, sid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss_sum += ce(logits.float(), y).item() * len(y)
        n += len(y)
        ys.extend(y.cpu().tolist())
        ps.extend(logits.argmax(1).cpu().tolist())
        subjects.extend(sid.tolist())
    recalls = recall_score(ys, ps, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(ys, ps),
        "recall_0": recalls[0],
        "recall_1": recalls[1],
        "recall_2": recalls[2],
        "n_trials": n,
        "n_subjects": len(set(subjects)),
        "confusion_matrix": confusion_matrix(ys, ps, labels=[0, 1, 2]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["EEGNet", "TSception", "RGNN", "EEG-Conformer"], default="EEGNet")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--max-trials-per-subject", type=int, default=0)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--balanced-loss", action="store_true")
    ap.add_argument("--final-test", action="store_true", help="Only use after hyperparameters are frozen.")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(Path(args.data) / "splits.csv")
    subjects = pd.read_csv(Path(args.data) / "subjects.csv")
    test = set(split.loc[split.fold == args.fold, "subject_id"].astype(int))
    val_fold = 1 if args.fold == 0 else 0
    val = set(split.loc[split.fold == val_fold, "subject_id"].astype(int)) - test
    all_ids = set(subjects.subject_id.astype(int))
    train = all_ids - test - val
    if not train or not val or not test:
        raise RuntimeError("empty subject split")

    split_sets = [("train", train), ("val", val)]
    if args.final_test:
        split_sets.append(("test", test))
    datasets = {
        name: MI3RawTrials(args.data, ids, args.max_trials_per_subject, args.preload and name != "test")
        for name, ids in split_sets
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_sampler=SubjectBatchSampler(dataset, args.batch_size, name == "train"),
            num_workers=0,
            pin_memory=True,
        )
        for name, dataset in datasets.items()
    }

    device = torch.device("cuda")
    config = json.loads((Path(args.data) / "config.json").read_text(encoding="utf-8"))
    adjacency_path = Path(args.data) / "electrode_adjacency.npy"
    if args.model == "RGNN" and not adjacency_path.exists():
        raise RuntimeError(f"missing distance-based RGNN adjacency: {adjacency_path}")
    constructors = {
        "EEGNet": lambda: EEGNet(dropout=args.dropout),
        "TSception": lambda: TSception(config["channel_order"], dropout=args.dropout),
        "RGNN": lambda: RGNN(torch.from_numpy(np.load(adjacency_path)), dropout=args.dropout),
        "EEG-Conformer": lambda: EEGConformer(dropout=args.dropout),
    }
    model = constructors[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = None
    training_counts = np.zeros(3, dtype=np.int64)
    for sid in sorted(train):
        _, labels, _ = datasets["train"]._read_subject_raw(sid, signals=False)
        selected_labels = labels[datasets["train"].selected[sid]]
        training_counts += np.bincount(selected_labels, minlength=3)[:3]
    if args.balanced_loss:
        class_weights = torch.tensor(
            training_counts.sum() / (3.0 * training_counts), dtype=torch.float32, device=device
        )
    print(json.dumps({
        "training_class_counts": training_counts.tolist(),
        "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_dataset_instantiated": args.final_test,
    }), flush=True)

    ce = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda")
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        start = time.time()
        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = ce(logits, y)
                if hasattr(model, "regularization"):
                    loss = loss + 1e-5 * model.regularization()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if hasattr(model, "constrain"):
                model.constrain()
            total_loss += loss.item() * len(y)
            n += len(y)
        val_metrics = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n, 1),
            "seconds": time.time() - start,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best.pt")

    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result_rows = []
    result_splits = ["val", "test"] if args.final_test else ["val"]
    for name in result_splits:
        metrics = evaluate(model, loaders[name], device)
        result_rows.append({
            "model": args.model,
            "implementation": {
                "EEGNet": "faithful PyTorch reimplementation of EEGNet-8,2",
                "TSception": "official TSception-v2 topology adapted to symmetric MI-3 channels",
                "RGNN": "official RGNN/SGC formulation with learnable symmetric adjacency adapted to raw MI-3 trials",
                "EEG-Conformer": "official EEG-Conformer topology adapted to 128-channel MI-3 input",
            }[args.model],
            "pretraining": "none supervised from scratch",
            "fold": args.fold,
            "split": name,
            "seed": args.seed,
            "best_epoch": checkpoint["epoch"],
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out / "results.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "args": vars(args),
        "train_subjects": sorted(train),
        "val_subjects": sorted(val),
        "test_subjects": sorted(test),
        "training_class_counts": training_counts.tolist(),
        "test_was_not_used_for_selection": True,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(result_rows).to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
