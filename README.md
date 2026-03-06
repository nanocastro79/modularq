# modularq 🔬

**Modular Software Stack for Quantum Computing**

Una librería Python que implementa diagnóstico y corrección de no-equilibrio modular en mediciones cuánticas, basada en:

> Balfagón, C. (2025). *A Modular Software Stack for Quantum Computing: From Born-Rule Equilibrium to Nonequilibrium-Aware Quantum Software*. Universidad de Buenos Aires.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanocastro79/modularq/blob/main/demo_colab.ipynb)

---

## ¿Qué problema resuelve?

Las computadoras cuánticas NISQ operan como sistemas abiertos y ruidosos. Las estadísticas de medición pueden desviarse sistemáticamente de las predicciones ideales (Born rule) cuando el proceso de medición opera fuera del equilibrio termodinámico (KMS).

**modularq** detecta y corrige estas desviaciones automáticamente, sin modificar el hardware ni los circuitos.

---

## Instalación

```bash
pip install numpy scipy matplotlib
# Luego descargá modularq.py desde este repo
```

O en Google Colab:

```python
!wget https://raw.githubusercontent.com/nanocastro79/modularq/main/modularq.py
from modularq import ModularAnalyzer
```

---

## Uso básico

```python
import numpy as np
from modularq import ModularAnalyzer

# Matriz de densidad reconstruida (del subsistema medido)
rho = np.array([[0.52, 0.01], [0.01, 0.48]])

# Conteos experimentales
counts = {"0": 5200, "1": 4800}

# Born baseline
p0 = {"0": 0.5, "1": 0.5}

# Analizar
analyzer = ModularAnalyzer(eps=1e-6, mode="auto")
result = analyzer.analyze(rho, counts, p0)

print(result.summary())
# → MIS, régimen, probabilidades corregidas, comparación AIC
```

---

## Qué hace el stack (capas)

| Capa | Función |
|------|---------|
| 0 | Hardware cuántico (sin modificar) |
| 1 | Reconstrucción de estado (tomografía / shadows) |
| **2** | **Diagnóstico modular: K = -log(ρ), δK_i** ← núcleo |
| 3 | Clasificación: Born-válido / perturbativo / no-KMS |
| 4 | Regla de medición efectiva: reweighting exponencial |
| 5 | Control modular (minimizar imbalance) |
| 6 | API de software |
| 7 | Inferencia estadística (AIC/BIC) |

---

## Modular Imbalance Score (MIS)

Métrica escalar que indica la validez del Born rule:

| MIS | Régimen | Acción |
|-----|---------|--------|
| < 0.02 | ✅ KMS-balanced | Ninguna (Born válido) |
| 0.02 – 0.10 | ⚠️ Perturbativo | Corrección leve |
| > 0.10 | 🔴 No-KMS | Corrección necesaria |

---

## Reproducibilidad

Los experimentos originales están disponibles en Zenodo:
- DOI: [10.5281/zenodo.18066279](https://doi.org/10.5281/zenodo.18066279)
- Hardware: IBM Quantum (ibm_marrakesh, ibm_fez, ibm_torino)
- Circuitos: Bell (2q) y GHZ (3q)

---

## Licencia

**Uso académico y no comercial:** libre y gratuito bajo los términos de esta licencia.

**Uso comercial:** requiere acuerdo de licencia separado.  
Contacto: cb@balfagonresearch.org

© 2025 Christian H. Balfagón and Mariano M. Castro.
