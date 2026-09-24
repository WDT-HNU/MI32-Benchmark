# Models and adapter contracts

All runners receive the same sealed `[B,32,750] @ 250 Hz, Volt` tensor. Model-specific changes are
implemented as explicit `ModelAdapter` objects. Learned statistics must be fitted on training
subjects only.

| Model | Executed definition | Adapter output | Important boundary |
|---|---|---|---|
| EEGNet | EEGNet-8,2; 64-sample temporal kernel; depthwise full-channel spatial conv; separable conv | fixed scale and train-only calibration | one linear 3-class head; no added hidden layer |
| TSception | V2 temporal, asymmetric spatial, fusion topology | official 26-channel left/right paired order | six midline channels excluded by official pairing algorithm |
| RGNN | signed learnable graph, SGC order, sum pooling | `[B,32,5]` five-band log-power | backbone only; no target-domain NodeDAT/EmotionDL |
| EEG-Conformer | official patch embedding and six-layer encoder | train-only global standardization | flatten MLP is active; no invented CLS/position embedding |
| LaBraM | sealed Braindecode 1.7.0 equivalent backbone | µV, bandlimited 250→200 Hz, `[B,32,3,200]` | non-CLS token mean pooling and `Linear(200,3)` |
| EEGMamba | frozen official 12-layer Mamba2 backbone | V→µV/100, topology-preserving 32 channels, 200 Hz, 3 patches | no 32→60 interpolation, no 750→800 padding, all-patch head |
| CodeBrain | frozen official EEGSSM backbone | V→µV/100, topology-preserving 32 channels, 200 Hz, 3 patches | official-style flatten-all-patches downstream MLP |
| Uni-NTFM | official public backbone plus project supervised wrapper | deterministic five-region tensor geometry | protocol benchmark only; no official downstream checkpoint/head |

Each adapter exposes `none`, `naive`, and `official` ablation states so that the value of the
adaptation can be tested instead of assumed. The main benchmark always uses `official`.

## Identity requirements

- Model name alone is never sufficient evidence.
- Formal runs store upstream URL/commit, relevant source hashes, checkpoint URL/revision/size/hash,
  adapter manifest, dataset manifest, full arguments, split subject IDs, and executed source.
- If any required upstream or checkpoint is unavailable, the corresponding model is blocked; it
  is not silently replaced by a similar architecture.

Machine-readable identities live in `configs/models.json`.
