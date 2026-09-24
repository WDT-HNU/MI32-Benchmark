# Models

Every runner starts from the same `[B, 32, 750]` tensor at 250 Hz in volts. The adapter beside each
model performs the model-specific work; any learned calibration is fitted on training subjects only.

| Model | Implementation used here | Adapter output | Classification path |
|---|---|---|---|
| EEGNet | EEGNet-8,2; 64-sample temporal kernel; full-channel depthwise spatial conv | fixed scale and train-only calibration | one linear 3-class head |
| TSception | V2 temporal, asymmetric spatial, and fusion blocks | official 26-channel left/right paired order | V2 classifier |
| RGNN | signed graph, second-order SGC, sum pooling | `[B,32,5]` five-band log-power | RGNN backbone head |
| EEG-Conformer | official patch embedding and six-layer encoder | train-only global standardization | flatten MLP |
| LaBraM | Braindecode 1.7.0 equivalent backbone with official weights | µV, 250→200 Hz, `[B,32,3,200]` | mean of non-CLS tokens, then `Linear(200,3)` |
| EEGMamba | official 12-layer Mamba2 backbone and checkpoint | V→µV/100, 200 Hz, three patches | all-patch head |
| CodeBrain | official EEGSSM backbone and checkpoint | V→µV/100, 200 Hz, three patches | flatten-all-patches MLP |
| Uni-NTFM | public backbone plus this project's supervised wrapper | deterministic five-region geometry | protocol-benchmark head |

The adapters expose `none`, `naive`, and `official` modes for ablation work. Benchmark runs use
`official`.

## Pinning the implementation

The exact upstream URL and commit for every model live in `configs/models.json`. LaBraM, EEGMamba,
and CodeBrain also record checkpoint URL, revision, size, and SHA-256. Run manifests add the data
manifest, split subjects, command-line arguments, adapter settings, and executed source.

If an upstream checkout or required checkpoint is missing, the runner stops instead of swapping in
a look-alike implementation. The per-model details and known non-equivalent substitutions are in
the [model cards](model_cards/README.md).
