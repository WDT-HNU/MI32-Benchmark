# Model cards

These cards define what each benchmark name means in this repository. A model name without its
upstream commit, adapter contract, checkpoint identity (when applicable), and result track is not
a reproducible model identity.

| Card | Track |
|---|---|
| [EEGNet](EEGNet.md) | formal reproduction |
| [TSception](TSception.md) | formal reproduction |
| [RGNN](RGNN.md) | formal backbone benchmark |
| [EEG-Conformer](EEG-Conformer.md) | formal reproduction |
| [LaBraM](LaBraM.md) | formal checkpoint adaptation |
| [EEGMamba](EEGMamba.md) | formal checkpoint adaptation |
| [CodeBrain](CodeBrain.md) | formal checkpoint adaptation |
| [Uni-NTFM](Uni-NTFM.md) | protocol benchmark only |

All cards inherit the subject-level split, validation-only selection, and one-shot test rules in
`docs/BENCHMARK_PROTOCOL.md`.
