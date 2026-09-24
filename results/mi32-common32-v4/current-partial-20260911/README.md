# Current partial result snapshot

This directory contains the latest available MI32 runs that match the sealed model-input unit
contracts as of 2026-09-11.

- Classic models: fold 0 only.
- CodeBrain: folds 0, 2, and 3.
- EEGMamba: folds 0, 2, and 3.
- LaBraM: folds 0 and 2.
- Uni-NTFM: absent from the formal-reproduction track.

Every result CSV is paired with its run manifest. Unequal fold coverage means these files must not
be collapsed into a final cross-model ranking. See `docs/RESULTS.md` for the exact interpretation.
