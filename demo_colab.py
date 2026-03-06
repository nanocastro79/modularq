"""
╔══════════════════════════════════════════════════════════╗
║   MODULAR SOFTWARE STACK — Demo en Google Colab          ║
║   Copiá cada celda en Colab y ejecutala en orden         ║
╚══════════════════════════════════════════════════════════╝

INSTRUCCIONES:
1. Abrí Google Colab (colab.research.google.com)
2. Creá un nuevo notebook
3. Copiá cada sección marcada con # ── CELDA N ──
4. Ejecutalas en orden con Shift+Enter
"""

# ── CELDA 1 ─────────────────────────────────────────────
# Instalá las dependencias necesarias
# (Colab ya tiene numpy y scipy, pero por las dudas)

# !pip install numpy scipy matplotlib qiskit qiskit-ibm-runtime --quiet
# print("✅ Dependencias instaladas")


# ── CELDA 2 ─────────────────────────────────────────────
# Descargá modularq desde tu repo de GitHub
# (reemplazá TU_USUARIO con tu usuario de GitHub)

# !wget https://raw.githubusercontent.com/TU_USUARIO/modularq/main/modularq.py
# print("✅ modularq.py descargado")


# ── CELDA 3 ─────────────────────────────────────────────
# Importaciones

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from modularq import ModularAnalyzer, classify_regime, modular_imbalance_score


# ── CELDA 4 ─────────────────────────────────────────────
# DEMO 1: Estado GHZ de 2 qubits (caso básico)
#
# Simulamos un estado Bell |Φ+> = (|00> + |11>) / sqrt(2)
# con ruido realista de hardware (como IBM Quantum)

print("=" * 55)
print("DEMO 1: Estado Bell con ruido de hardware simulado")
print("=" * 55)

# Estado Bell ideal
psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
rho_ideal = np.outer(psi, psi)

# Agregamos ruido (simula lo que ocurre en hardware real)
# Este ruido reproduce el tipo de desbalance que el paper detectó
# en IBM Quantum (ver Sección 5.2 del paper)
noise_level = 0.08
rho_noisy = (1 - noise_level) * rho_ideal + noise_level * np.eye(4) / 4

# Conteos simulados (lo que vería un detector real)
# Notar el leve sesgo: más '00' que '11' y algo de error de lectura
counts_bell = {
    "00": 5180,
    "01": 45,
    "10": 30,
    "11": 4745,
}

# Baseline Born ideal
p0_bell = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
# Agregamos pequeña prob. no-ideal para estabilidad numérica
p0_bell = {"00": 0.499, "11": 0.499, "01": 0.001, "10": 0.001}

# Ejecutar análisis
analyzer = ModularAnalyzer(eps=1e-6, mode="auto")
result = analyzer.analyze(rho_noisy, counts_bell, p0_bell)

print(result.summary())


# ── CELDA 5 ─────────────────────────────────────────────
# DEMO 2: Comparación Born vs Modular en un gráfico

print("\nDEMO 2: Visualización Born vs Modular vs Empírico")

outcomes = result.outcomes
x = np.arange(len(outcomes))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Gráfico 1: Probabilidades ──
ax = axes[0]
bars1 = ax.bar(x - width, [result.p0[o] for o in outcomes],
               width, label='Born (ideal)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, [result.p_modular[o] for o in outcomes],
               width, label='Modular (corregido)', color='darkorange', alpha=0.8)
bars3 = ax.bar(x + width, [result.freq_empirical[o] for o in outcomes],
               width, label='Empírico (medido)', color='green', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(outcomes, fontsize=11)
ax.set_ylabel('Probabilidad', fontsize=11)
ax.set_title('Born vs Modular vs Empírico\n(Estado Bell con ruido)', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, max([result.p0[o] for o in outcomes]) * 1.3)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

# ── Gráfico 2: Desbalance modular (deltaK) ──
ax2 = axes[1]
deltaK_vals = [result.deltaK[o] for o in outcomes]
colors_dk = ['red' if v > 0 else 'blue' for v in deltaK_vals]
bars_dk = ax2.bar(outcomes, deltaK_vals, color=colors_dk, alpha=0.75)

ax2.axhline(0, color='black', linewidth=1, linestyle='--')
ax2.set_ylabel('δK_i (desbalance modular)', fontsize=11)
ax2.set_title(f'Desbalance Modular por Resultado\nMIS = {result.mis:.4f} → {result.regime}',
              fontsize=12)
ax2.grid(axis='y', alpha=0.3)

red_patch = mpatches.Patch(color='red', alpha=0.75, label='Subrepresentado (δK > 0)')
blue_patch = mpatches.Patch(color='blue', alpha=0.75, label='Sobrerrepresentado (δK < 0)')
ax2.legend(handles=[red_patch, blue_patch], fontsize=9)

for bar, val in zip(bars_dk, deltaK_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + (0.002 if val >= 0 else -0.005),
             f'{val:.4f}', ha='center',
             va='bottom' if val >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('modular_analysis_bell.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado como modular_analysis_bell.png")


# ── CELDA 6 ─────────────────────────────────────────────
# DEMO 3: Heatmap de MIS (como en Figura 2 del paper)
# Simula un procesador de 5 qubits

print("\nDEMO 3: Heatmap MIS en un procesador de 5 qubits")

# MIS simulados para 5 qubits en topología lineal
# En producción, esto vendría de mediciones reales de IBM Quantum
mis_por_qubit = {
    'Q0': 0.03,   # KMS-balanced
    'Q1': 0.11,   # perturbativo
    'Q2': 0.21,   # no-KMS fuerte
    'Q3': 0.07,   # perturbativo leve
    'Q4': 0.02,   # KMS-balanced
}

fig, ax = plt.subplots(figsize=(10, 3))

qubits = list(mis_por_qubit.keys())
mis_vals = list(mis_por_qubit.values())

# Color: verde (bajo MIS) → amarillo → rojo (alto MIS)
norm = plt.Normalize(vmin=0, vmax=0.25)
cmap = plt.cm.RdYlGn_r

scatter = ax.scatter(
    range(len(qubits)), [0] * len(qubits),
    c=mis_vals, cmap=cmap, norm=norm,
    s=2000, zorder=3
)

for i, (q, mis_v) in enumerate(mis_por_qubit.items()):
    regime, _ = classify_regime(mis_v)
    ax.text(i, 0, q, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(i, -0.15, f'MIS={mis_v:.2f}', ha='center',
            va='center', fontsize=9)
    ax.text(i, 0.15, regime.split('_')[0], ha='center',
            va='center', fontsize=8, color='gray')

# Conectar qubits con líneas
for i in range(len(qubits) - 1):
    ax.plot([i, i+1], [0, 0], 'k-', linewidth=2, zorder=1)

cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('MIS (Modular Imbalance Score)', fontsize=10)

ax.set_xlim(-0.5, len(qubits) - 0.5)
ax.set_ylim(-0.4, 0.4)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('Heatmap MIS — 5 qubits en topología lineal\n'
             'Verde = Born válido | Rojo = No-KMS (corrección necesaria)',
             fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('mis_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Heatmap guardado como mis_heatmap.png")


# ── CELDA 7 ─────────────────────────────────────────────
# DEMO 4: Shot-scaling test (valida que MIS es señal real, no ruido)
# Sección 4.3.1 del paper

print("\nDEMO 4: Shot-scaling test — MIS vs ruido estadístico")

shot_budgets = [200, 500, 1000, 3000, 10000, 30000]
mis_values = []
shot_noise = []

# Estado con desbalance modular real (no-equilibrio)
rho_noneq = np.array([
    [0.52, 0.05j, -0.02j, 0.03],
    [-0.05j, 0.03, 0.01, 0.02j],
    [0.02j, 0.01, 0.02, -0.01j],
    [0.03, -0.02j, 0.01j, 0.43]
], dtype=complex)
rho_noneq = rho_noneq / np.trace(rho_noneq)

p0_test = {"00": 0.499, "01": 0.001, "10": 0.001, "11": 0.499}

for N in shot_budgets:
    # Simular conteos a N shots
    true_probs = [0.515, 0.015, 0.010, 0.460]
    counts_sim = np.random.multinomial(N, true_probs)
    counts_dict = {"00": counts_sim[0], "01": counts_sim[1],
                   "10": counts_sim[2], "11": counts_sim[3]}

    res = ModularAnalyzer(eps=1e-6).analyze(rho_noneq, counts_dict, p0_test)
    mis_values.append(res.mis)
    shot_noise.append(1 / np.sqrt(N))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(shot_budgets, mis_values, 'o-', color='darkorange',
        linewidth=2, markersize=8, label='MIS observado (señal real)')
ax.plot(shot_budgets, shot_noise, 's--', color='steelblue',
        linewidth=2, markersize=8, label='1/√N (ruido estadístico)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Shot budget (N)', fontsize=12)
ax.set_ylabel('Magnitud', fontsize=12)
ax.set_title('Shot-scaling test\n'
             'MIS real → estable con N | Ruido estadístico → cae como 1/√N',
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shot_scaling_test.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Shot-scaling test guardado")

print("\n" + "=" * 55)
print("  ✅ Demo completo. Próximo paso:")
print("  Conectar con IBM Quantum real (ver CELDA 8)")
print("=" * 55)


# ── CELDA 8 (OPCIONAL) ──────────────────────────────────
# Conexión con IBM Quantum real
# Necesitás tu API key de quantum.ibm.com

"""
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit
import json

# Pegá tu API key de IBM Quantum aquí
IBM_API_KEY = "TU_API_KEY_AQUI"

# Conectar al servicio
service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_API_KEY)

# Circuito Bell real
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Ejecutar en hardware real
backend = service.least_busy(operational=True, simulator=False)
print(f"Ejecutando en: {backend.name}")

sampler = Sampler(backend=backend)
job = sampler.run([qc], shots=4000)
result_ibm = job.result()

# Extraer conteos
counts_real = dict(result_ibm[0].data.c.get_counts())
print("Conteos reales:", counts_real)

# Analizar con modularq
# (necesitás también reconstruir rho desde tomografía)
# Esto es el siguiente paso del proyecto
"""
