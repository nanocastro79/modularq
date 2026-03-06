"""
modularq.py — Modular Software Stack for Quantum Systems
=========================================================
Implementación basada en:
  Balfagón, C. (2025). "A Modular Software Stack for Quantum Computing"
  Universidad de Buenos Aires.

Uso en Google Colab:
    # Clonar repo y hacer:
    from modularq import ModularAnalyzer

Autores: C. Balfagón (framework teórico) — Implementación open-source
"""

import numpy as np
from scipy.linalg import logm
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────
#  CAPA 2: DIAGNÓSTICO MODULAR (núcleo)
# ─────────────────────────────────────────────

def regularize_state(rho: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Regulariza la matriz de densidad para evitar valores singulares.
    Ecuación (5) del paper: rho_eps = (1-eps)*rho + eps*I/d

    Parámetros
    ----------
    rho : np.ndarray
        Matriz de densidad (debe ser cuadrada, hermítica, traza=1)
    eps : float
        Parámetro de regularización (default: 1e-6)

    Retorna
    -------
    np.ndarray : Matriz de densidad regularizada
    """
    d = rho.shape[0]
    rho_eps = (1 - eps) * rho + eps * np.eye(d) / d
    return rho_eps


def modular_generator(rho: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Calcula el generador modular K = -log(rho_eps).
    Ecuación (4) del paper.

    Este es el objeto central del framework: captura cuánto se desvía
    el estado medido del equilibrio (KMS).

    Parámetros
    ----------
    rho : np.ndarray
        Matriz de densidad del subsistema medido
    eps : float
        Parámetro de regularización

    Retorna
    -------
    np.ndarray : Generador modular K
    """
    rho_eps = regularize_state(rho, eps)
    K = -logm(rho_eps)
    # Forzar que sea real si la parte imaginaria es despreciable
    if np.max(np.abs(K.imag)) < 1e-10:
        K = K.real
    return K


def modular_imbalance(
    K: np.ndarray,
    p0: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcula el desbalance modular por resultado: deltaK_i.
    Ecuación (6) del paper.

    deltaK_i = <i|K|i> - sum_j(p0_j * <j|K|j>)

    Mide cuánto se desvía cada resultado del equilibrio KMS.
    Si todos los deltaK_i son ~0, el Born rule es válido.

    Parámetros
    ----------
    K : np.ndarray
        Generador modular (matriz)
    p0 : dict
        Probabilidades Born baseline {'00': 0.5, '11': 0.5, ...}

    Retorna
    -------
    dict : Desbalance modular por resultado {'00': deltaK_00, ...}
    """
    outcomes = list(p0.keys())
    n = len(outcomes)
    d = K.shape[0]

    if n != d:
        raise ValueError(
            f"Dimensión incompatible: p0 tiene {n} outcomes pero K es {d}x{d}. "
            f"Asegurate de que el subsistema medido tenga {int(np.log2(d))} qubits."
        )

    # Elementos diagonales de K en la base de medición: <i|K|i>
    Ki = {outcomes[i]: K[i, i].real for i in range(n)}

    # Promedio ponderado por Born: sum_j p0_j * K_jj
    K_bar = sum(p0[outcomes[j]] * Ki[outcomes[j]] for j in range(n))

    # Desbalance: deltaK_i = K_ii - K_bar
    deltaK = {outcome: Ki[outcome] - K_bar for outcome in outcomes}

    return deltaK


def modular_imbalance_score(deltaK: Dict[str, float]) -> float:
    """
    Calcula el Modular Imbalance Score (MIS).
    Ecuación (10) del paper.

    MIS = (1/|O|) * sum_i |deltaK_i|

    Interpretación:
    - MIS ~ 0      → régimen KMS/Born válido (medición confiable)
    - MIS > 0.05   → no-equilibrio perturbativo (corrección leve)
    - MIS > 0.15   → no-KMS fuerte (corrección necesaria)

    Parámetros
    ----------
    deltaK : dict
        Desbalance modular por resultado

    Retorna
    -------
    float : MIS (adimensional, comparable entre dispositivos)
    """
    values = list(deltaK.values())
    mis = np.mean(np.abs(values))
    return float(mis)


# ─────────────────────────────────────────────
#  CAPA 3: CLASIFICACIÓN DE RÉGIMEN
# ─────────────────────────────────────────────

def classify_regime(mis: float) -> Tuple[str, str]:
    """
    Clasifica el régimen de medición según el MIS.
    Capa 3 del stack (Sección 3.4 del paper).

    Parámetros
    ----------
    mis : float
        Modular Imbalance Score

    Retorna
    -------
    tuple : (régimen, descripción)
    """
    if mis < 0.02:
        return ("KMS_BALANCED",
                "✅ Born válido: las estadísticas son confiables sin corrección.")
    elif mis < 0.10:
        return ("PERTURBATIVE_NEQ",
                "⚠️  No-equilibrio perturbativo: corrección modular leve recomendada.")
    else:
        return ("NON_KMS",
                "🔴 No-KMS fuerte: corrección modular necesaria para resultados confiables.")


# ─────────────────────────────────────────────
#  CAPA 4: REGLA DE MEDICIÓN EFECTIVA
# ─────────────────────────────────────────────

def modular_probabilities(
    p0: Dict[str, float],
    deltaK: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcula las probabilidades corregidas por no-equilibrio modular.
    Ecuación (7) del paper.

    p_i = p0_i * exp(-deltaK_i) / Z
    Z   = sum_j p0_j * exp(-deltaK_j)

    Propiedades garantizadas:
    - Preserva normalización (suma = 1)
    - Preserva positividad
    - Reduce exactamente a Born cuando deltaK_i = 0

    Parámetros
    ----------
    p0 : dict
        Probabilidades Born baseline
    deltaK : dict
        Desbalance modular por resultado

    Retorna
    -------
    dict : Probabilidades corregidas
    """
    outcomes = list(p0.keys())

    # Pesos exponenciales
    weights = {o: p0[o] * np.exp(-deltaK[o]) for o in outcomes}

    # Factor de normalización
    Z = sum(weights.values())

    # Probabilidades normalizadas
    p_mod = {o: weights[o] / Z for o in outcomes}

    return p_mod


# ─────────────────────────────────────────────
#  CAPA 7: INFERENCIA Y COMPARACIÓN DE MODELOS
# ─────────────────────────────────────────────

def log_likelihood(
    counts: Dict[str, int],
    probs: Dict[str, float],
    eps_clip: float = 1e-12
) -> float:
    """
    Calcula la log-verosimilitud multinomial.
    Ecuación (22) del paper.

    log L = sum_i n_i * log(p_i)

    Parámetros
    ----------
    counts : dict
        Conteos observados por resultado {'00': 5200, '11': 4700, ...}
    probs : dict
        Probabilidades del modelo
    eps_clip : float
        Evita log(0)

    Retorna
    -------
    float : Log-verosimilitud
    """
    ll = 0.0
    for outcome, count in counts.items():
        p = max(probs.get(outcome, eps_clip), eps_clip)
        ll += count * np.log(p)
    return float(ll)


def compute_aic(log_l: float, k_params: int) -> float:
    """
    Criterio de Información de Akaike (AIC).
    Penaliza modelos con más parámetros.

    AIC = 2k - 2*log(L)

    Parámetros
    ----------
    log_l : float
        Log-verosimilitud del modelo
    k_params : int
        Número de parámetros libres del modelo

    Retorna
    -------
    float : AIC (menor = mejor modelo)
    """
    return 2 * k_params - 2 * log_l


def compute_bic(log_l: float, k_params: int, n_shots: int) -> float:
    """
    Criterio de Información Bayesiano (BIC).

    BIC = k*log(N) - 2*log(L)
    """
    return k_params * np.log(n_shots) - 2 * log_l


def model_comparison(
    counts: Dict[str, int],
    p_born: Dict[str, float],
    p_modular: Dict[str, float]
) -> Dict:
    """
    Compara el modelo Born vs modelo Modular usando AIC y BIC.
    Sección 5.3 del paper.

    El modelo Born tiene 0 parámetros libres (es la baseline).
    El modelo Modular tiene 1 parámetro libre (epsilon de regularización).

    Parámetros
    ----------
    counts : dict
        Conteos experimentales
    p_born : dict
        Probabilidades Born (sin corrección)
    p_modular : dict
        Probabilidades con corrección modular

    Retorna
    -------
    dict : Resultados completos de la comparación
    """
    N = sum(counts.values())

    ll_born = log_likelihood(counts, p_born)
    ll_mod = log_likelihood(counts, p_modular)

    aic_born = compute_aic(ll_born, k_params=0)
    aic_mod = compute_aic(ll_mod, k_params=1)

    bic_born = compute_bic(ll_born, k_params=0, n_shots=N)
    bic_mod = compute_bic(ll_mod, k_params=1, n_shots=N)

    delta_aic = aic_born - aic_mod  # positivo = prefiere modular
    delta_bic = bic_born - bic_mod

    if delta_aic > 2:
        preference = "MODULAR"
        strength = "fuerte" if delta_aic > 10 else "moderada"
    elif delta_aic < -2:
        preference = "BORN"
        strength = "fuerte" if delta_aic < -10 else "moderada"
    else:
        preference = "EMPATE"
        strength = "evidencia insuficiente"

    return {
        "N_shots": N,
        "logL_born": round(ll_born, 2),
        "logL_modular": round(ll_mod, 2),
        "AIC_born": round(aic_born, 2),
        "AIC_modular": round(aic_mod, 2),
        "BIC_born": round(bic_born, 2),
        "BIC_modular": round(bic_mod, 2),
        "delta_AIC": round(delta_aic, 2),
        "delta_BIC": round(delta_bic, 2),
        "preferred_model": preference,
        "evidence_strength": strength,
    }


# ─────────────────────────────────────────────
#  CLASE PRINCIPAL: ModularAnalyzer
# ─────────────────────────────────────────────

class ModularAnalyzer:
    """
    Interfaz principal del Modular Software Stack.

    Uso básico
    ----------
    >>> analyzer = ModularAnalyzer(eps=1e-6)
    >>> result = analyzer.analyze(rho, counts, p0)
    >>> print(result.summary())
    """

    def __init__(self, eps: float = 1e-6, mode: str = "auto"):
        """
        Parámetros
        ----------
        eps : float
            Parámetro de regularización (default: 1e-6)
        mode : str
            'auto'  → aplica corrección si AIC lo favorece
            'born'  → fuerza Born (sin corrección)
            'modular' → fuerza corrección modular siempre
        """
        self.eps = eps
        self.mode = mode
        self._results = {}

    def analyze(
        self,
        rho: np.ndarray,
        counts: Dict[str, int],
        p0: Optional[Dict[str, float]] = None
    ) -> "ModularResult":
        """
        Ejecuta el análisis completo del stack modular.

        Parámetros
        ----------
        rho : np.ndarray
            Matriz de densidad reconstruida del subsistema medido
        counts : dict
            Conteos experimentales {'00': 5200, '11': 4700, ...}
        p0 : dict, opcional
            Born baseline. Si None, se calcula desde rho.

        Retorna
        -------
        ModularResult : Objeto con todos los resultados
        """
        outcomes = list(counts.keys())
        N = sum(counts.values())

        # Si no hay p0, calcularlo desde la diagonal de rho
        if p0 is None:
            diag = np.diag(rho).real
            diag = np.abs(diag) / np.sum(np.abs(diag))
            p0 = {outcomes[i]: diag[i] for i in range(len(outcomes))}

        # Paso 1: Generador modular
        K = modular_generator(rho, self.eps)

        # Paso 2: Desbalance modular
        deltaK = modular_imbalance(K, p0)

        # Paso 3: MIS y clasificación de régimen
        mis = modular_imbalance_score(deltaK)
        regime, regime_desc = classify_regime(mis)

        # Paso 4: Probabilidades corregidas
        p_mod = modular_probabilities(p0, deltaK)

        # Paso 5: Comparación de modelos
        comparison = model_comparison(counts, p0, p_mod)

        # Frecuencias empíricas
        freq_empirical = {o: counts[o] / N for o in outcomes}

        return ModularResult(
            outcomes=outcomes,
            N_shots=N,
            p0=p0,
            p_modular=p_mod,
            freq_empirical=freq_empirical,
            K=K,
            deltaK=deltaK,
            mis=mis,
            regime=regime,
            regime_description=regime_desc,
            model_comparison=comparison,
            eps=self.eps,
            mode=self.mode,
        )


class ModularResult:
    """
    Contenedor de resultados del análisis modular.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def summary(self) -> str:
        """Resumen legible de los resultados."""
        lines = [
            "=" * 55,
            "  MODULAR SOFTWARE STACK — Resultados",
            "=" * 55,
            f"  Shots totales     : {self.N_shots:,}",
            f"  Epsilon (reg.)    : {self.eps}",
            f"  MIS               : {self.mis:.4f}",
            f"  Régimen           : {self.regime}",
            f"  {self.regime_description}",
            "-" * 55,
            "  Probabilidades por resultado:",
            f"  {'Outcome':<8} {'Born':>10} {'Modular':>10} {'Empírico':>10}",
        ]
        for o in self.outcomes:
            lines.append(
                f"  {o:<8} "
                f"{self.p0[o]:>10.4f} "
                f"{self.p_modular[o]:>10.4f} "
                f"{self.freq_empirical[o]:>10.4f}"
            )
        lines += [
            "-" * 55,
            "  Comparación de modelos (AIC):",
            f"  AIC Born          : {self.model_comparison['AIC_born']}",
            f"  AIC Modular       : {self.model_comparison['AIC_modular']}",
            f"  ΔAIC (Born-Mod)   : {self.model_comparison['delta_AIC']}",
            f"  Modelo preferido  : {self.model_comparison['preferred_model']}",
            f"  Evidencia         : {self.model_comparison['evidence_strength']}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def best_probabilities(self) -> Dict[str, float]:
        """
        Retorna las mejores probabilidades según el criterio AIC.
        En modo 'auto', elige entre Born y Modular automáticamente.
        """
        if self.mode == "born":
            return self.p0
        elif self.mode == "modular":
            return self.p_modular
        else:  # auto
            if self.model_comparison["preferred_model"] == "MODULAR":
                return self.p_modular
            return self.p0
