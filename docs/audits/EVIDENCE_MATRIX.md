# Evidence matrix

| Component | Identity sealed | CPU structure | Real CUDA preflight | Formal result in repo | Status |
|---|---|---|---|---|---|
| Dataset v4 | manifest + metadata | contract tests | real-sample gate | used by available folds | PASS for sealed holders |
| EEGNet | source snapshot | yes | recorded | fold 0 | partial benchmark |
| TSception | commit + source snapshot | yes | recorded | fold 0 | partial benchmark |
| RGNN backbone | commit + source snapshot | yes | recorded | fold 0 | partial benchmark |
| EEG-Conformer | commit + source snapshot | yes | recorded | fold 0 | partial benchmark |
| LaBraM | converted + author checkpoint hashes | yes | recorded | folds 0/2 | partial benchmark |
| EEGMamba | commit + checkpoint + source hashes | yes; stub is structure-only | recorded with real Mamba2 | folds 0/2/3 | partial benchmark |
| CodeBrain | commit + checkpoint + source hashes | yes | recorded | folds 0/2/3 | partial benchmark |
| Uni-NTFM | commit | geometry test | supported | no formal reproduction | protocol track only |

“Partial benchmark” means the planned fold matrix is incomplete. Current result CSVs and run
manifests are published under `results/mi32-common32-v4/current-partial-20260911/`. The older
fold-0 package is retained only as historical audit evidence because foundation input contracts
were subsequently hardened.
