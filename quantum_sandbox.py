from qiskit.circuit.library import zz_feature_map

feature_map = zz_feature_map(feature_dimension=4, reps=2, entanglement="linear")
print(feature_map.decompose())
print("\nCircuit depth:", feature_map.decompose().depth())
print("Number of qubits:", feature_map.num_qubits)

# ============================================================
# VISUALIZATION — save a clean image of the circuit diagram
# ============================================================
import os
os.makedirs("results", exist_ok=True)

fig = feature_map.decompose().draw(output="mpl", style="iqp", fold=20)
fig.savefig("results/08_quantum_feature_map_circuit.png", dpi=150, bbox_inches="tight")
print("\nSaved circuit diagram image to results/08_quantum_feature_map_circuit.png")