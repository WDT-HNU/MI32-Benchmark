# Third-party sources and licensing boundary

This repository does not vendor upstream model repositories or pretrained weights. The bootstrap
tools clone/download pinned identities into ignored local directories.

License texts available from the audited upstream checkouts, plus the Braindecode 1.7.0 license
and notice required by the LaBraM source snapshot, are preserved in `THIRD_PARTY_LICENSES/`.
Absence of a license file (notably the audited Uni-NTFM checkout and RGNN snapshot) must be
treated as “no redistribution permission established,” not as public-domain permission.

| Component | Upstream | Pinned identity | Upstream license observed |
|---|---|---|---|
| EEGNet | https://github.com/vlawhern/arl-eegmodels | `4a512e503198db2010848813ead9afbf8cd54c97` | see upstream |
| TSception | https://github.com/yi-ding-cs/TSception | `9efd666b618d006e32e6da1d30dbc79b1d190604` | custom/see upstream |
| RGNN | https://github.com/zhongpeixiang/RGNN | `685b1a2645185bd8129c04a789dbfde8ad896a59` | see upstream |
| EEG-Conformer | https://github.com/eeyhsong/EEG-Conformer | `9ae149ba62487ceae723277d13adac27837113d2` | see upstream |
| LaBraM | https://github.com/935963004/LaBraM | `c431221e6cfd23dbfa9950e0180682fb322b0548` | MIT |
| EEGMamba | https://github.com/wjq-learning/EEGMamba | `dbc83fa072744201e8897aeb9f65007b952ad323` | MIT |
| CodeBrain | https://github.com/jingyingma01/CodeBrain | `22d350caf68246d2fda4f630ef837420db3fb130` | Apache-2.0 |
| Uni-NTFM | https://github.com/Zhisheng-researcher/Uni-NTFM | `c0ce0152f94366e59b31b3fb2c108ce909bcd95c` | no explicit LICENSE observed |

Checkpoint URLs, revisions, sizes, and hashes are recorded in `configs/models.json`. Downloading
an artifact does not grant rights beyond its upstream terms.

The MI32 signal release is derived from BNCI2014_001, PhysionetMI, Schirrmeister2017,
Stieger2021, Wairagkar2018, Weibo2014, Zhou2016, and Zhou2020. Users must cite each source they
use. Before publishing the combined signal archive, maintainers must complete the per-source
redistribution review described in `datasets/mi32/DATASET_CARD.md`.
