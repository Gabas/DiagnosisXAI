"""
Testes do detector de perfis atípicos (core/ood_detector.py).

Usam um recorte real da base Wisconsin (via ``breast_cancer_data``, do conftest,
já padronizado). Validam o contrato: a maioria do próprio treino é típica, dados
da mesma distribuição (teste) são majoritariamente típicos, e um ponto claramente
extremo é sinalizado como atípico.
"""

import numpy as np
import pytest

from core.ood_detector import OODDetector


@pytest.fixture(scope="module")
def detector(breast_cancer_data):
    return OODDetector(breast_cancer_data['X_train_scaled'], percentil=99.0)


def test_maioria_do_treino_e_tipica(detector, breast_cancer_data):
    # No percentil 99, ~1% do treino fica acima do limiar por construção.
    flags = detector.flag(breast_cancer_data['X_train_scaled'])
    assert flags.mean() <= 0.03


def test_conjunto_de_teste_majoritariamente_tipico(detector, breast_cancer_data):
    # Dados da mesma distribuição (não vistos) devem ter poucos atípicos.
    flags = detector.flag(breast_cancer_data['X_test_scaled'])
    assert flags.mean() < 0.15


def test_ponto_extremo_e_atipico(detector, breast_cancer_data):
    n_features = np.asarray(breast_cancer_data['X_train_scaled']).shape[1]
    extremo = np.full((1, n_features), 50.0)  # 50 desvios-padrão — claramente fora
    assert bool(detector.flag(extremo)[0])
    assert detector.score(extremo)[0] > detector.limiar


def test_rotulos_tem_forma_e_valores_esperados(detector, breast_cancer_data):
    rotulos = detector.rotulos(breast_cancer_data['X_test_scaled'])
    assert len(rotulos) == len(breast_cancer_data['X_test_scaled'])
    assert set(rotulos) <= {'Típico', 'Atípico'}


def test_limiar_positivo(detector):
    assert detector.limiar > 0
    assert detector.percentil == 99.0
