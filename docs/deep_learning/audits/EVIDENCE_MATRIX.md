# Deep-model evidence matrix

| Component | Identity sealed | CPU structure | Real CUDA preflight | Repository evidence | Status |
|---|---|---|---|---|---|
| Dataset v4 | manifest + metadata | contract tests | real-sample gate | used-fold records | sealed files |
| EEGNet | source snapshot | yes | recorded | fold 0 | published evidence |
| TSception | commit + source snapshot | yes | recorded | fold 0 | published evidence |
| RGNN backbone | commit + source snapshot | yes | recorded | fold 0 | published evidence |
| EEG-Conformer | commit + source snapshot | yes | recorded | fold 0 | published evidence |
| LaBraM | converted weight + author-weight hash | yes | recorded | folds 0/2 | published evidence |
| EEGMamba | commit + checkpoint + source hashes | yes; stub is structural only | recorded with real Mamba2 | folds 0/2/3 | published evidence |
| CodeBrain | commit + checkpoint + source hashes | yes | recorded | folds 0/2/3 | published evidence |
| Uni-NTFM | commit | geometry tests | supported | separate protocol route | implementation published |

Current result CSVs and run manifests are under `benchmark/results/mi32-common32-v4/current-evidence-20260911/`. The historical fold-0 directory preserves an immutable audit trail of the earlier executed source but is not the current result table.
