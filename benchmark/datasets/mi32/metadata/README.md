# UEHF-MI Three-Class Common-32 Montage v4

This is a self-contained 32-channel version of UEHF-MI Three-Class. Every trial has the same
ordered common 32-electrode standard_1005 subset shared exactly with the PooledMI binary dataset. A target channel that existed
in the source recording is copied bit-for-bit. A missing target channel is
estimated from all measured source channels with Perrin spherical-spline
interpolation using MNE-Python's standard_1005 coordinates (alpha=1e-5).

`channel_mask` is True for all 32 model-valid channels. `measured_mask` records
provenance: True means directly measured and False means interpolated. Models
must consume all 32 channels and may use `measured_mask` only as an optional
quality/provenance feature; it must not be used to zero interpolated values.

Signals remain in volts with no runtime normalization. The label mapping is
0=left_upper, 1=right_upper, and 2=non_upper. The dataset contains 230 subjects
and 97,608 trials, with an exact 1:1:2 class ratio.

`electrode_adjacency.npy` is the 32x32 RGNN distance adjacency regenerated for
this exact montage. `release_audit.json` records the independent source-to-release
audit, and `SHA256SUMS` seals every release file.

Known limitations: class 2 is a heterogeneous `non_upper` superclass that may
contain feet, tongue, or rest depending on the source dataset. The upstream v2.4
ontology excludes hands/both_hands/both_hand boundary events.

The source QC report marks 501 trials for robust-extreme review (no NaN, Inf, or
flat trials). `quality_flags.csv` maps every flag to its sealed global `trial_id`.
Those review-marked trials are retained; any exclusion or sensitivity analysis
must be declared before training and applied without consulting test results.
