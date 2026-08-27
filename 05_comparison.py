import json
import matplotlib.pyplot as plt
import numpy as np
import os

with open("results/classical_results.json") as f:
    classical = json.load(f)

with open("results/quantum_results.json") as f:
    quantum = json.load(f)

vqc = None
if os.path.exists("results/vqc_results.json"):
    with open("results/vqc_results.json") as f:
        vqc = json.load(f)

all_results = classical + [quantum]
if vqc is not None:
    all_results.append(vqc)

def short_name(name):
    if "13 features" in name:
        return "Classical SVM\n(13 features)"
    if "4 features" in name or "same 4" in name:
        return "Classical SVM\n(4 features)"
    if "Quantum Kernel" in name:
        return "Quantum Kernel\nSVM (4 qubits)"
    if "VQC" in name:
        return "VQC\n(4 qubits)"
    return name

names = [short_name(r["model"]) for r in all_results]
accuracies = [r["accuracy"] for r in all_results]
f1s = [r["f1"] for r in all_results]

x = np.arange(len(all_results))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#1f6fb2")
bars2 = ax.bar(x + width/2, f1s, width, label="F1 Score", color="#f5a623")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.legend()
ax.set_title("Sanjeevani AI — Classical vs Quantum Model Comparison\n"
              "Heart Disease Risk Prediction", fontsize=13)
ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=9)
ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("results/comparison_chart.png", dpi=150)
print("Saved results/comparison_chart.png")

print("\n" + "=" * 60)
print("FINAL RESULTS TABLE")
print("=" * 60)
for r in all_results:
    print(f"{r['model']:<45} acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")