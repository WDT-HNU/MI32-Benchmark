# Third-party sources and redistribution boundaries

This repository does not vendor upstream model repositories or pretrained checkpoints. Fetch tools download pinned identities into Git-ignored local directories.

Paths on this page are relative to `benchmark/`.

Observed upstream licences and the Braindecode 1.7.0 notice required by the LaBraM source snapshot are retained under `THIRD_PARTY_LICENSES/`. Absence of a licence file—particularly for the audited Uni-NTFM source and RGNN snapshot—means redistribution permission has not been established; it does not place the work in the public domain.

| Component | Upstream | Pinned identity | Observed upstream licence |
|---|---|---|---|
| EEGNet | https://github.com/vlawhern/arl-eegmodels | `4a512e503198db2010848813ead9afbf8cd54c97` | see upstream repository |
| TSception | https://github.com/yi-ding-cs/TSception | `9efd666b618d006e32e6da1d30dbc79b1d190604` | custom; see upstream repository |
| RGNN | https://github.com/zhongpeixiang/RGNN | `685b1a2645185bd8129c04a789dbfde8ad896a59` | see upstream repository |
| EEG-Conformer | https://github.com/eeyhsong/EEG-Conformer | `9ae149ba62487ceae723277d13adac27837113d2` | see upstream repository |
| LaBraM | https://github.com/935963004/LaBraM | `c431221e6cfd23dbfa9950e0180682fb322b0548` | MIT |
| EEGMamba | https://github.com/wjq-learning/EEGMamba | `dbc83fa072744201e8897aeb9f65007b952ad323` | MIT |
| CodeBrain | https://github.com/jingyingma01/CodeBrain | `22d350caf68246d2fda4f630ef837420db3fb130` | Apache-2.0 |
| Uni-NTFM | https://github.com/Zhisheng-researcher/Uni-NTFM | `c0ce0152f94366e59b31b3fb2c108ce909bcd95c` | no explicit LICENSE observed |

Checkpoint URL, version, size, and hash are recorded in `configs/models.json`. Downloading a file does not grant rights beyond its upstream terms.

MI32 signals originate from BNCI2014_001, PhysionetMI, Schirrmeister2017, Stieger2021, Wairagkar2018, Weibo2014, Zhou2016, and Zhou2020. Users must cite every source actually used. Before publishing a merged signal package, maintainers must complete the source-by-source redistribution review described in `datasets/mi32/DATASET_CARD.md`.

The P300 card documents a MOABB-based selection route but does not redistribute component recordings. Each selected dataset remains subject to its original access, citation, and licence terms.
