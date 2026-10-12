import argparse
import json
from pathlib import Path

import numpy as np


def inverse_square_adjacency(xyz_cm, delta=5.0):
    """RGNN equation (9): min(1, delta / d^2), with an identity diagonal."""
    xyz_cm = np.asarray(xyz_cm, dtype=np.float64)
    if xyz_cm.ndim != 2 or xyz_cm.shape[1] != 3:
        raise ValueError("RGNN coordinates must have shape [channels, 3]")
    distance = np.linalg.norm(xyz_cm[:, None, :] - xyz_cm[None, :, :], axis=-1)
    adjacency = np.minimum(1.0, float(delta) / np.maximum(distance ** 2, 1e-12))
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def main():
    import mne

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=5.0)
    args = ap.parse_args()
    channels = json.loads(Path(args.config).read_text(encoding="utf-8"))["channel_order"]
    positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lookup = {key.casefold(): value for key, value in positions.items()}
    xyz = np.stack([lookup[channel.casefold()] for channel in channels]).astype(np.float64)
    # RGNN equation (9) uses inverse squared physical distance with delta=5.
    # MNE coordinates are metres, while the paper's electrode coordinates are centimetres.
    xyz_cm = xyz * 100.0
    adjacency = inverse_square_adjacency(xyz_cm, delta=args.delta)
    np.save(args.out, adjacency.astype(np.float32))
    print(json.dumps({
        "shape": list(adjacency.shape),
        "delta": args.delta,
        "non_negligible_fraction": float((adjacency > 0.1).mean()),
        "min": float(adjacency.min()),
        "max": float(adjacency.max()),
        "montage": "MNE standard_1005",
    }))


if __name__ == "__main__":
    main()
