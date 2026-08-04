"""
Testes das métricas de auditoria (core/metrics.py).

Usam matrizes de confusão construídas à mão — o ponto aqui é a definição das
métricas (o que conta como falso negativo, como cada proporção é formada) e o
comportamento nos casos-limite que a interface pode encontrar: lote com uma só
classe, modelo que nunca acusa malignidade, gabarito sem coluna de diagnóstico.
"""

import pandas as pd
import pytest

from core.metrics import (
    analise_critica,
    avaliar_modelos,
    calcular_metricas,
    ic_wilson,
    ressalvas_do_lote,
)

MALIGNO = 'Maligno'
BENIGNO = 'Benigno'


def _lote(vp: int, fn: int, fp: int, vn: int):
    """Monta (reais, previstos) com exatamente a matriz de confusão pedida."""
    reais = [MALIGNO] * (vp + fn) + [BENIGNO] * (fp + vn)
    previstos = [MALIGNO] * vp + [BENIGNO] * fn + [MALIGNO] * fp + [BENIGNO] * vn
    return reais, previstos


def test_matriz_de_confusao_conta_cada_quadrante():
    m = calcular_metricas(*_lote(vp=7, fn=3, fp=2, vn=8))
    assert (m['vp'], m['fn'], m['fp'], m['vn']) == (7, 3, 2, 8)
    assert m['n'] == 20


def test_metricas_derivadas_da_matriz():
    m = calcular_metricas(*_lote(vp=7, fn=3, fp=2, vn=8))
    assert m['acuracia'] == pytest.approx(75.0)          # (7 + 8) / 20
    assert m['sensibilidade'] == pytest.approx(70.0)     # 7 / 10 malignos
    assert m['especificidade'] == pytest.approx(80.0)    # 8 / 10 benignos
    assert m['precisao'] == pytest.approx(77.78, abs=0.01)   # 7 / 9 apontados
    assert m['vpn'] == pytest.approx(72.73, abs=0.01)        # 8 / 11 liberados
    assert m['f1'] == pytest.approx(73.68, abs=0.01)


def test_classificador_perfeito():
    m = calcular_metricas(*_lote(vp=10, fn=0, fp=0, vn=10))
    assert m['sensibilidade'] == 100.0
    assert m['especificidade'] == 100.0
    assert m['f1'] == 100.0


def test_modelo_que_nunca_acusa_malignidade():
    """Acurácia alta escondendo sensibilidade zero — o caso que motiva a tela."""
    m = calcular_metricas(*_lote(vp=0, fn=5, fp=0, vn=95))
    assert m['acuracia'] == pytest.approx(95.0)
    assert m['sensibilidade'] == 0.0
    assert m['precisao'] is None      # nenhum paciente foi apontado como maligno
    assert m['f1'] == 0.0


def test_metricas_indefinidas_viram_none_em_lote_de_uma_classe():
    m = calcular_metricas(*_lote(vp=4, fn=1, fp=0, vn=0))
    assert m['especificidade'] is None
    assert m['sensibilidade'] == pytest.approx(80.0)
    assert m['auc'] is None           # AUC exige as duas classes no gabarito


def test_auc_usa_as_certezas_e_ignora_o_limiar():
    reais = [MALIGNO, MALIGNO, BENIGNO, BENIGNO]
    previstos = [BENIGNO] * 4            # todos errados no corte de 50%...
    certezas = [40.0, 35.0, 20.0, 10.0]  # ...mas a ordenação por risco é perfeita
    m = calcular_metricas(reais, previstos, certezas)
    assert m['sensibilidade'] == 0.0
    assert m['auc'] == pytest.approx(1.0)


def test_ic_wilson_encolhe_com_mais_amostras():
    estreito = ic_wilson(90, 100)
    largo = ic_wilson(9, 10)
    assert (largo[1] - largo[0]) > (estreito[1] - estreito[0])
    assert estreito[0] < 90.0 < estreito[1]


def test_ic_wilson_nao_extrapola_em_acerto_total():
    baixo, alto = ic_wilson(5, 5)
    assert alto == pytest.approx(100.0, abs=1e-6)
    assert baixo < 100.0    # 5/5 não sustenta afirmar 100% de sensibilidade


def test_ic_wilson_sem_amostras():
    assert ic_wilson(0, 0) is None


def test_avaliar_modelos_modo_comparacao_traduz_siglas():
    df = pd.DataFrame({
        'Diagnóstico_Real': [MALIGNO, MALIGNO, BENIGNO, BENIGNO],
        'IA_RAN': [MALIGNO, MALIGNO, BENIGNO, BENIGNO],
        'IA_ÁRV': [MALIGNO, BENIGNO, BENIGNO, BENIGNO],
    })
    metricas = avaliar_modelos(df)
    assert set(metricas) == {'Random Forest', 'Árvore de Decisão'}
    # Ordenado pela sensibilidade: o que pegou os dois malignos vem primeiro.
    assert list(metricas)[0] == 'Random Forest'


def test_avaliar_modelos_modelo_unico_usa_o_nome_informado():
    df = pd.DataFrame({
        'Diagnóstico_Real': [MALIGNO, BENIGNO],
        'Diagnóstico_IA': [MALIGNO, BENIGNO],
    })
    assert list(avaliar_modelos(df, nome_modelo='SVM')) == ['SVM']
    assert list(avaliar_modelos(df)) == ['Modelo Selecionado']


def test_avaliar_modelos_sem_coluna_de_diagnostico():
    df = pd.DataFrame({'Diagnóstico_Real': [MALIGNO, BENIGNO]})
    assert avaliar_modelos(df) == {}


def test_critica_denuncia_falso_negativo():
    metricas = {'SVM': calcular_metricas(*_lote(vp=8, fn=2, fp=0, vn=10))}
    critica = analise_critica('SVM', metricas['SVM'], metricas)
    assert any('2 caso(s) maligno(s)' in r for r in critica['ressalvas'])
    assert critica['veredito']


def test_critica_reconhece_ausencia_de_falso_negativo():
    metricas = {'Random Forest': calcular_metricas(*_lote(vp=10, fn=0, fp=1, vn=9))}
    critica = analise_critica('Random Forest', metricas['Random Forest'], metricas)
    assert any('nenhum tumor maligno' in f for f in critica['fortes'])
    assert any('alarme(s) falso(s)' in r for r in critica['ressalvas'])


def test_critica_aponta_desempenho_igual_ao_palpite_trivial():
    # Nunca acusa malignidade: acerta os 95 benignos e nada mais.
    metricas = {'KNN': calcular_metricas(*_lote(vp=0, fn=5, fp=0, vn=95))}
    critica = analise_critica('KNN', metricas['KNN'], metricas)
    assert any('palpite' in r for r in critica['ressalvas'])


def test_critica_nao_elogia_metrica_em_que_todos_empatam():
    """Se ninguém é distinguido pela métrica, o 'está entre os melhores' é vazio."""
    metricas = {
        'SVM': calcular_metricas(*_lote(vp=9, fn=1, fp=0, vn=10)),
        'KNN': calcular_metricas(*_lote(vp=7, fn=3, fp=0, vn=10)),
    }
    critica_svm = analise_critica('SVM', metricas['SVM'], metricas)
    # Especificidade 100% nos dois: ninguém leva o crédito.
    assert not any('especificidade' in f for f in critica_svm['fortes'])
    # Sensibilidade separa os dois: o melhor é apontado.
    assert any('sensibilidade' in f for f in critica_svm['fortes'])


def test_veredito_diferencia_faixas_de_sensibilidade():
    perfeito = analise_critica('SVM', calcular_metricas(*_lote(vp=10, fn=0, fp=0, vn=10)), {})
    intermediario = analise_critica('SVM', calcular_metricas(*_lote(vp=92, fn=8, fp=0, vn=100)), {})
    ruim = analise_critica('SVM', calcular_metricas(*_lote(vp=50, fn=50, fp=40, vn=60)), {})
    assert 'triagem assistida' in perfeito['veredito']
    assert 'segunda opinião' in intermediario['veredito']
    assert 'insuficiente' in ruim['veredito']


def test_critica_traz_os_tracos_do_algoritmo():
    metricas = {'Árvore de Decisão': calcular_metricas(*_lote(vp=9, fn=1, fp=1, vn=9))}
    critica = analise_critica('Árvore de Decisão', metricas['Árvore de Decisão'], metricas)
    assert any('regras' in f for f in critica['fortes'])
    assert any('overfitting' in r for r in critica['ressalvas'])


def test_ressalvas_do_lote_avisam_sobre_amostra_pequena():
    metricas = {'SVM': calcular_metricas(*_lote(vp=5, fn=1, fp=1, vn=5))}
    avisos = ressalvas_do_lote(metricas)
    assert any('Lote pequeno' in a for a in avisos)
    assert any('treino' in a for a in avisos)      # aviso de vazamento é sempre exibido


def test_ressalvas_do_lote_sem_metricas():
    assert ressalvas_do_lote({}) == []
