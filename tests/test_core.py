import numpy as np
import pytest
from modularq import ModularAnalyzer

# Estado |+> ideal
RHO_IDEAL = np.array([[0.5, 0.5], [0.5, 0.5]])

# Estado con ruido (similar al hardware real)
RHO_NOISY = np.array([[0.505+0.j,    0.494-0.028j],
                       [0.494+0.028j, 0.495+0.j   ]])

def test_mis_ideal_es_cero():
    """Con rho ideal, el MIS debe ser 0"""
    counts = {'0': 1000, '1': 1000}
    p0     = {'0': 0.5,  '1': 0.5}
    result = ModularAnalyzer().analyze(RHO_IDEAL, counts, p0)
    assert result.mis == pytest.approx(0.0, abs=1e-6)

def test_regimen_kms_con_rho_ideal():
    """Con rho ideal el regimen debe ser KMS_BALANCED"""
    counts = {'0': 1000, '1': 1000}
    p0     = {'0': 0.5,  '1': 0.5}
    result = ModularAnalyzer().analyze(RHO_IDEAL, counts, p0)
    assert result.regime == 'KMS_BALANCED'

def test_mis_positivo_con_ruido():
    """Con rho ruidosa el MIS debe ser > 0"""
    counts = {'0': 987, '1': 1013}
    p0     = {'0': 0.5, '1': 0.5}
    result = ModularAnalyzer().analyze(RHO_NOISY, counts, p0)
    assert result.mis > 0.0

def test_probabilidades_normalizadas():
    """Las probabilidades corregidas deben sumar 1"""
    counts = {'0': 987, '1': 1013}
    p0     = {'0': 0.5, '1': 0.5}
    result = ModularAnalyzer().analyze(RHO_NOISY, counts, p0)
    total = sum(result.p_modular.values())
    assert total == pytest.approx(1.0, abs=1e-9)

def test_summary_contiene_mis():
    """El summary debe incluir el MIS"""
    counts = {'0': 987, '1': 1013}
    p0     = {'0': 0.5, '1': 0.5}
    result = ModularAnalyzer().analyze(RHO_NOISY, counts, p0)
    assert 'MIS' in result.summary()

def test_dimension_incompatible():
    """Debe lanzar error si p0 y rho tienen dimensiones distintas"""
    rho_3x3 = np.eye(3) / 3
    counts   = {'0': 500, '1': 500}
    p0       = {'0': 0.5, '1': 0.5}
    with pytest.raises(ValueError):
        ModularAnalyzer().analyze(rho_3x3, counts, p0)
