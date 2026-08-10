"""
Testes do PredictorEngine (core/predictor.py).

Assim como o BatchProcessor, exercitam o ``data/wisconsin.pkl`` real via
``ModelLoader`` — validando que os 5 modelos empacotados carregam e preveem
corretamente sobre um lote real, e que o modo "Todos (Comparação)" gera uma
coluna por modelo.
"""

import os

import pandas as pd
import pytest

from core.batch_processor import BatchProcessor
from core.decision import PoliticaDecisao
from core.inference import ModelLoader
from core.predictor import PredictorEngine


@pytest.fixture(scope="module")
def loader():
    return ModelLoader()


@pytest.fixture(scope="module")
def lote_processado(repo_data_dir):
    caminho = os.path.join(repo_data_dir, 'dataTeste_sem_diagnostico.csv')
    df_bruto = pd.read_csv(caminho)
    processor = BatchProcessor()
    return processor.process(df_bruto)  # (df_scaled, df_raw)


def test_loader_carrega_os_5_modelos(loader):
    esperados = {'Árvore de Decisão', 'Random Forest', 'SVM', 'KNN', 'Regressão Logística'}
    assert esperados <= set(loader.models.keys())
    assert all(loader.models[nome] is not None for nome in esperados)


@pytest.mark.parametrize("modelo", [
    'Árvore de Decisão', 'Random Forest', 'SVM', 'KNN', 'Regressão Logística',
])
def test_predict_um_modelo_gera_diagnostico_valido(loader, lote_processado, modelo):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, modelo)

    assert 'Diagnóstico_IA' in resultado.columns
    assert set(resultado['Diagnóstico_IA'].unique()) <= {'Maligno', 'Benigno'}
    assert len(resultado) == len(df_scaled)


def test_predict_arvore_omite_coluna_de_certeza(loader, lote_processado):
    """A árvore tem folhas puras (probabilidade sempre 0% ou 100%) — a coluna
    de certeza é deliberadamente omitida (ver MODELOS_SEM_CERTEZA)."""
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, 'Árvore de Decisão')
    assert 'Certeza_Maligno(%)' not in resultado.columns


def test_predict_svm_inclui_coluna_de_certeza(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, 'SVM')
    assert 'Certeza_Maligno(%)' in resultado.columns
    assert resultado['Certeza_Maligno(%)'].between(0, 100).all()


@pytest.mark.parametrize("certeza, esperado", [
    (15, 'Limítrofe'), (10, 'Limítrofe'), (20, 'Limítrofe'),
    (5, 'Limítrofe'), (25, 'Limítrofe'),           # bordas inclusivas (limiar ±10)
    (4.9, 'Definida'), (25.1, 'Definida'), (2, 'Definida'), (50, 'Definida'),
])
def test_classificar_decisao_gira_em_torno_do_limiar_de_operacao(loader, certeza, esperado):
    """A faixa de revisão acompanha o limiar do modelo, não os 50% do sklearn.

    Com limiar de 15%, uma certeza de 50% é uma decisão folgada (bem acima do
    corte) — e não mais um caso limítrofe, como seria sob a régua antiga."""
    engine = PredictorEngine(loader, PoliticaDecisao({'SVM': 0.15}, banda=0.10))
    assert engine.classificar_decisao(certeza, 'SVM') == esperado


@pytest.mark.parametrize("modelo", ['Random Forest', 'SVM', 'KNN', 'Regressão Logística'])
def test_rotulo_e_certeza_nunca_se_contradizem(loader, lote_processado, modelo):
    """O rótulo e a certeza exibida saem da mesma probabilidade calibrada.

    Antes o rótulo vinha do modelo original e a certeza do calibrado, então a
    tabela chegava a mostrar 'Benigno' com 61% de certeza de malignidade."""
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, modelo)

    limiar = engine.politica.limiar(modelo) * 100
    acima = resultado['Certeza_Maligno(%)'] >= limiar
    marcado_maligno = resultado['Diagnóstico_IA'] == 'Maligno'
    assert (acima == marcado_maligno).all()


def test_limiar_mais_baixo_nunca_perde_maligno(loader, lote_processado):
    """Baixar o limiar só pode acrescentar malignos, nunca retirar."""
    df_scaled, df_raw = lote_processado
    conservador = PredictorEngine(loader, PoliticaDecisao({'SVM': 0.50}))
    sensivel = PredictorEngine(loader, PoliticaDecisao({'SVM': 0.15}))

    marcados_conservador = conservador.predict(df_scaled, df_raw, 'SVM')['Diagnóstico_IA'] == 'Maligno'
    marcados_sensivel = sensivel.predict(df_scaled, df_raw, 'SVM')['Diagnóstico_IA'] == 'Maligno'

    assert (marcados_sensivel | marcados_conservador).equals(marcados_sensivel)
    assert marcados_sensivel.sum() >= marcados_conservador.sum()


def test_comite_esta_disponivel_e_nao_e_um_modelo_do_pkl(loader):
    engine = PredictorEngine(loader)
    assert PredictorEngine.NOME_COMITE in engine.modelos_disponiveis()
    assert PredictorEngine.NOME_COMITE not in loader.models
    # A Árvore não entra: sem probabilidade utilizável, não há o que promediar.
    assert 'Árvore de Decisão' not in engine.membros_comite()


def test_comite_gera_certeza_igual_a_media_dos_membros(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, PredictorEngine.NOME_COMITE)

    colunas_membros = [f'Certeza_{PredictorEngine.sigla(m)}(%)' for m in engine.membros_comite()]
    assert all(c in resultado.columns for c in colunas_membros)

    media = resultado[colunas_membros].mean(axis=1)
    assert (media - resultado['Certeza_Maligno(%)']).abs().max() < 0.02  # só arredondamento


def test_comite_marca_mais_malignos_que_o_limiar_padrao(loader, lote_processado):
    """O ponto da calibração: o mesmo comitê acusa mais casos do que a 0,5."""
    df_scaled, df_raw = lote_processado
    comite = PredictorEngine.NOME_COMITE
    calibrado = PredictorEngine(loader)
    neutro = PredictorEngine(loader, PoliticaDecisao({comite: 0.50}))

    n_calibrado = (calibrado.predict(df_scaled, df_raw, comite)['Diagnóstico_IA'] == 'Maligno').sum()
    n_neutro = (neutro.predict(df_scaled, df_raw, comite)['Diagnóstico_IA'] == 'Maligno').sum()
    assert n_calibrado > n_neutro


def test_predict_svm_inclui_coluna_decisao(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    resultado = PredictorEngine(loader).predict(df_scaled, df_raw, 'SVM')
    assert 'Decisão' in resultado.columns
    assert set(resultado['Decisão'].unique()) <= {'Limítrofe', 'Definida'}


def test_predict_arvore_omite_coluna_decisao(loader, lote_processado):
    """Sem certeza informativa, a Árvore não recebe a marcação de decisão."""
    df_scaled, df_raw = lote_processado
    resultado = PredictorEngine(loader).predict(df_scaled, df_raw, 'Árvore de Decisão')
    assert 'Decisão' not in resultado.columns


def test_predict_todos_gera_uma_coluna_ia_por_modelo(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, 'Todos (Comparação)')
    colunas_ia = [c for c in resultado.columns if c.startswith('IA_')]
    assert len(colunas_ia) == 5


def test_recusa_produz_tres_saidas_e_nao_perde_ninguem(loader, lote_processado):
    """Com a recusa ligada, o lote se divide em decididos e devolvidos."""
    df_scaled, df_raw = lote_processado
    politica = PoliticaDecisao({'SVM': 0.15}, faixas_revisao={'SVM': (0.01, 0.70)})
    politica.adiar_incertos = True
    resultado = PredictorEngine(loader, politica).predict(df_scaled, df_raw, 'SVM')

    diagnosticos = resultado['Diagnóstico_IA']
    assert set(diagnosticos.unique()) <= {'Maligno', 'Benigno', 'Revisar'}
    assert (diagnosticos == 'Revisar').any()
    assert len(diagnosticos) == len(df_scaled)
    # Quem foi devolvido está dentro da faixa; quem foi decidido, fora dela.
    adiado = diagnosticos == 'Revisar'
    certeza = resultado['Certeza_Maligno(%)']
    assert certeza[adiado].between(1.0, 70.0, inclusive="left").all()
    assert not certeza[~adiado].between(1.0, 70.0, inclusive="left").any()


def test_recusa_desligada_nao_muda_nada(loader, lote_processado):
    """A recusa é opcional: desligada, o resultado é o de antes dela existir."""
    df_scaled, df_raw = lote_processado
    politica = PoliticaDecisao({'SVM': 0.15}, faixas_revisao={'SVM': (0.01, 0.70)})
    resultado = PredictorEngine(loader, politica).predict(df_scaled, df_raw, 'SVM')
    assert set(resultado['Diagnóstico_IA'].unique()) <= {'Maligno', 'Benigno'}


def test_predict_modelo_invalido_levanta_erro(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    with pytest.raises(ValueError):
        engine.predict(df_scaled, df_raw, 'Modelo Inexistente')
