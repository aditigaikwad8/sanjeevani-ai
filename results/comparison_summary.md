# Sanjeevani AI — Model Comparison Summary

## Dataset

**Combined 4-site UCI Heart Disease dataset — 918 patients**, built by
merging the Cleveland (303), Hungarian (293), Switzerland (123), and
V.A. Long Beach (199) sites from the official UCI collection, after
removing 2 exact duplicate patient records.

Two features — `ca` (66.3% missing) and `thal` (52.7% missing) — were
almost never recorded outside the Cleveland site. Rather than
imputing fabricated values for the majority of patients, both were
**dropped entirely** (threshold: drop any feature missing in more
than 40% of rows). The remaining smaller gaps, including `slope`
(33.4% missing), were filled using per-column medians. This leaves
**11 real, non-fabricated clinical features**.

*An earlier version of this project used only the Cleveland subset
(297 patients, 13 features). That dataset and its results are kept
for comparison — see "Dataset Expansion" below.*

## Feature selection for quantum models

Top 4 features selected via Random Forest importance ranking on the
combined dataset: **cp, thalach, age, chol**.

(On the original Cleveland-only dataset, the top 4 had been cp,
thal, thalach, oldpeak — this changed because `thal` was dropped for
data-quality reasons, and the ranking was naturally recomputed on
the larger, cleaner dataset.)

## Results

| Model | Features | Accuracy | F1 Score |
|---|---|---|---|
| Classical SVM | 11 (all) | 80.43% | 83.02% |
| Classical SVM | 4 (same as quantum) | 79.35% | 82.73% |
| **Hybrid Ensemble** (50% Classical + 25% QSVC + 25% VQC) | 4 (+ 11) | **78.26%** | **81.31%** |
| Equal-Weight Ensemble (Classical + QSVC + VQC) | 4 (+ 11) | 75.00% | 78.90% |
| Quantum Kernel SVM (QSVC) | 4 | 69.57% | 75.00% |
| Quantum-Only Ensemble (QSVC + VQC) | 4 | 67.39% | 72.97% |
| VQC | 4 | 64.13% | 67.65% |

## Dataset Expansion — Before vs After

A diagnostic check on the original Cleveland-only run showed QSVC
had a persistent ~22-point gap between training and test accuracy —
a classic sign of a quantum feature map that is too expressive for
the amount of training data available. Rather than accepting this as
a fixed limitation, the dataset was expanded to the combined 4-site
cohort and the quantum training sample sizes were increased
accordingly (QSVC: 180→250 patients, VQC: 120→180 patients).

| Model | Cleveland only (297 rows) | Combined 4-site (918 rows) | Change |
|---|---|---|---|
| Classical SVM (all features) | 86.67% | 80.43% | −6.24 pts |
| Classical SVM (4 features, matched) | 81.67% | 79.35% | −2.32 pts |
| **Quantum Kernel SVM (QSVC)** | **60.00%** | **69.57%** | **+9.57 pts** |
| VQC | 65.00% | 64.13% | −0.87 pts |
| QSVC train–test overfitting gap | 22.2 pts | 14.4 pts | **−7.8 pts** |
| Classical vs. quantum gap (matched 4 features) | 21.67 pts | 9.78 pts | **−11.89 pts** |

Classical accuracy dropped slightly — expected, since combining four
hospitals introduces genuine measurement variability across sites
that a single-site dataset doesn't have. QSVC, by contrast, improved
substantially and its overfitting gap nearly halved, supporting the
diagnosis that its original weak performance was significantly a
data-scarcity problem rather than a purely architectural one.

## Key finding

Classical SVM still outperforms both individual quantum approaches
on this dataset, even when restricted to the same 4 features the
quantum models use (79.35% vs 64–70%). This is consistent with
current literature: a 2025 npj Digital Medicine systematic review of
169 studies found no consistent quantum advantage yet in digital
health, and most published QML healthcare work to date relies on
simulators and comparatively small datasets, much like this project.

However, three more nuanced findings emerged from the full process:

1. **Data scarcity, not just model complexity, was driving QSVC's
   early weakness.** Expanding the dataset alone lifted QSVC's
   accuracy by nearly 10 points and nearly halved its overfitting
   gap, without changing the model's architecture at all.

2. **Which quantum method is "better" flipped with more data.**
   On the smaller Cleveland-only dataset, VQC clearly outperformed
   QSVC (65–68% vs 60%). On the larger combined dataset, QSVC
   overtook VQC (69.6% vs 64.1%), while VQC remained the more stable
   generalizer (a 2-point train-test gap vs. QSVC's 14). This
   suggests quantum kernel methods benefit more from additional
   training data than variational circuits do at this qubit scale —
   a distinction invisible in the original single-dataset experiment.

3. **A hybrid ensemble recovers most of the accuracy gap.** Blending
   all three models' predictions (weighted 50% classical, 25% QSVC,
   25% VQC) reaches 78.26% accuracy and 81.31% F1 — within about one
   point of the standalone 4-feature classical model — while
   genuinely incorporating both quantum approaches rather than
   discarding them.

## Why this project still matters

The value here isn't a single quantum model beating classical — it's
a complete, honest, reproducible pipeline that identified its own
limitation (a small, single-site dataset causing quantum overfitting),
fixed it with real additional data from three more verified hospital
sites rather than synthetic augmentation, and measured the exact
improvement. Building and diagnosing both a Quantum Kernel SVM and a
VQC surfaced a genuinely interesting comparative finding — that their
relative strengths depend on how much data is available — and the
hybrid ensemble shows a practical path to recovering most of the
remaining accuracy gap today, without waiting for better quantum
hardware. As quantum hardware and training techniques mature (see
Future Work: Quantum Kernel Training, PegasosQSVC, real IBM hardware
via Qiskit Runtime), this same pipeline is directly reusable to
re-test these findings at greater scale.