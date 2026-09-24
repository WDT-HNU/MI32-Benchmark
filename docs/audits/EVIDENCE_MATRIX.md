# Evidence matrix

| Component | Identity sealed | CPU structure | Real CUDA preflight | Evidence in repo | Status |
|---|---|---|---|---|---|
| Dataset v4 | manifest + metadata | contract tests | real-sample gate | used by recorded folds | sealed artifact |
| EEGNet | source snapshot | yes | recorded | fold 0 | evidence published |
| TSception | commit + source snapshot | yes | recorded | fold 0 | evidence published |
| RGNN backbone | commit + source snapshot | yes | recorded | fold 0 | evidence published |
| EEG-Conformer | commit + source snapshot | yes | recorded | fold 0 | evidence published |
| LaBraM | converted + author checkpoint hashes | yes | recorded | folds 0/2 | evidence published |
| EEGMamba | commit + checkpoint + source hashes | yes; stub is structure-only | recorded with real Mamba2 | folds 0/2/3 | evidence published |
| CodeBrain | commit + checkpoint + source hashes | yes | recorded | folds 0/2/3 | evidence published |
| Uni-NTFM | commit | geometry test | supported | separate protocol track | implementation published |

Current result CSVs and paired run manifests are published under
`results/mi32-common32-v4/current-evidence-20260911/`. The historical fold-0 directory provides an
additional immutable audit trail for earlier executed sources.
