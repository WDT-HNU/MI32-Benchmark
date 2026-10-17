# Model cards

Each card records the exact upstream commit, input transformation, classification head, checkpoint hash where applicable, and structural gate.

| Model | Reporting class |
|---|---|
| [EEGNet](EEGNet.md) | formal reproduction |
| [TSception](TSception.md) | formal reproduction |
| [RGNN](RGNN.md) | RGNN backbone benchmark |
| [EEG-Conformer](EEG-Conformer.md) | formal reproduction |
| [LaBraM](LaBraM.md) | checkpoint adaptation |
| [EEGMamba](EEGMamba.md) | checkpoint adaptation |
| [CodeBrain](CodeBrain.md) | checkpoint adaptation |
| [Uni-NTFM](Uni-NTFM.md) | separate protocol benchmark |

The public MI3 runs follow the subject split and validation/test rules in [`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md). Cross-task summaries retain their own recorded protocols.
