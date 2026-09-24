# Model cards

Click a model name to see the exact upstream commit, input conversion, classification head,
checkpoint hash (where applicable), and the test that guards the implementation.

| Model | How it is reported here |
|---|---|
| [EEGNet](EEGNet.md) | formal reproduction |
| [TSception](TSception.md) | formal reproduction |
| [RGNN](RGNN.md) | RGNN backbone benchmark |
| [EEG-Conformer](EEG-Conformer.md) | formal reproduction |
| [LaBraM](LaBraM.md) | checkpoint adaptation |
| [EEGMamba](EEGMamba.md) | checkpoint adaptation |
| [CodeBrain](CodeBrain.md) | checkpoint adaptation |
| [Uni-NTFM](Uni-NTFM.md) | separate protocol benchmark |

All eight use the subject split and validation/test rules in
[`docs/BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md).
