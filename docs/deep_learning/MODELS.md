# Deep-model definitions

The public MI3 runners start from the same `[B,32,750]`, 250 Hz, volt-scale tensor. A model-specific adapter performs only the required transformation; learned calibration quantities are fitted on training subjects only.

| Model | Repository implementation | Adapter output | Classification path |
|---|---|---|---|
| EEGNet | EEGNet-8,2; 64-sample temporal kernel; depthwise spatial convolution across all channels | fixed scaling and train-only calibration | one three-class linear layer |
| TSception | V2 temporal branches, asymmetric spatial branches, and fusion module | official 26-channel left/right paired order | V2 classifier |
| RGNN | signed graph, second-order SGC, and sum pooling | `[B,32,5]` five-band log-power features | RGNN backbone head |
| EEG-Conformer | official patch embedding and six-layer encoder | train-only global standardization | flattened MLP |
| LaBraM | backbone equivalent to Braindecode 1.7.0 with official weights | µV, band-limited 250→200 Hz, `[B,32,3,200]` | mean of non-CLS tokens, then `Linear(200,3)` |
| EEGMamba | official 12-layer Mamba2 backbone and weights | V→µV/100, 200 Hz, 3 patches | all-patch classification head |
| CodeBrain | official EEGSSM backbone and weights | V→µV/100, 200 Hz, 3 patches | flatten-all-patches MLP |
| Uni-NTFM | public backbone with an explicitly labelled supervised wrapper | deterministic five-region representation | protocol-benchmark head |

Adapters expose `none`, `naive`, and `official` modes for ablation. Formal benchmark runs use `official`.

## Frozen identities

`configs/models.json` records each upstream URL and exact commit. LaBraM, EEGMamba, and CodeBrain also record checkpoint URL, version, size, and SHA-256. Run manifests preserve the dataset manifest, split subjects, command-line arguments, adapter settings, and executed source snapshot.

If required upstream source or weights are missing, the runner stops instead of substituting a similar implementation. See the [model cards](model_cards/README.md) for detailed definitions and prohibited substitutes.
