# Project status

## Implemented scope

- Sealed MI32 common-32 v4 data contract and public metadata snapshot.
- Explicit model-input adapters for all eight selected models.
- Structure, boundary, unit-scale, and leakage gates for the selected implementations.
- Pinned identities for four classic and four foundation-model upstream repositories.
- Pinned checkpoint hashes for LaBraM, EEGMamba, and CodeBrain.
- Current-contract fold evidence snapshot and a separate historical audit archive.
- Public-repository structure, CI, download/verification tools, and AutoDL instructions.
- Dataset card, model cards, benchmark protocol, result interpretation, licensing notices, and
  contribution templates.

## Evaluation coverage

- EEGNet, TSception, RGNN backbone, and EEG-Conformer: fold 0 evidence.
- LaBraM: folds 0 and 2 evidence.
- EEGMamba and CodeBrain: folds 0, 2, and 3 evidence.
- Uni-NTFM: independently labeled protocol-benchmark implementation.

## Release model

Git contains code, metadata, manifests, and compact evidence. Signal archives, upstream source,
and pretrained checkpoints use separate retrieval paths and are verified against declared hashes.
The release checklist records the steps for attaching permanent artifact URLs, citations, and
source-specific distribution terms.
