# Sanjeevani AI — Model Comparison Summary

## Dataset
UCI Cleveland Heart Disease dataset — 297 patients after removing
rows with missing values, 13 clinical features, binary target
(disease present/absent).

## Feature selection for quantum models
Top 4 features selected via Random Forest importance ranking:
cp, thal, thalach, oldpeak.

## Results

| Model | Features | Accuracy | F1 Score |
|---|---|---|---|
| Classical SVM | 13 (all) | 86.67% | 84.62% |
| Classical SVM | 4 (same as quantum) | 81.67% | 78.43% |
| Quantum Kernel SVM | 4 | 60.00% | 53.85% |
| VQC | 4 | 68.33% | 67.80% |

## Key finding
Classical SVM outperforms both quantum approaches on this dataset,
even when restricted to the same 4 features the quantum models use
(81.67% vs 60-68%). Between the two quantum methods, VQC generalizes
far more reliably than the Quantum Kernel SVM: VQC's train and test
accuracy are nearly identical (69.2% vs 68.3%), while QSVC shows a
persistent ~22-point train-test gap even after cross-validated C
tuning across five values — indicating the gap comes from the
feature map's expressiveness relative to our small dataset, not from
insufficient regularization. This pattern is consistent with current
literature: a 2025 npj Digital Medicine systematic review of 169
studies found no consistent quantum advantage yet in digital health,
and most published QML healthcare work to date relies on simulators
and small datasets much like ours.

## Why this project still matters
The value here isn't a quantum model beating classical — it's a
complete, honest, reproducible pipeline that fairly benchmarks two
distinct quantum ML approaches against a matched classical baseline
on identical features. Building and diagnosing both a Quantum Kernel
SVM and a VQC also let us surface a genuinely interesting comparative
finding: variational approaches may generalize better than kernel
approaches on small clinical datasets at today's qubit scales. As
quantum hardware and training techniques mature, this same pipeline
is directly reusable to re-test that hypothesis on larger feature
sets or real quantum hardware.