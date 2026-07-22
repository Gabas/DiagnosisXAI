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


def test_predict_todos_gera_uma_coluna_ia_por_modelo(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    resultado = engine.predict(df_scaled, df_raw, 'Todos (Comparação)')
    colunas_ia = [c for c in resultado.columns if c.startswith('IA_')]
    assert len(colunas_ia) == 5


def test_predict_modelo_invalido_levanta_erro(loader, lote_processado):
    df_scaled, df_raw = lote_processado
    engine = PredictorEngine(loader)
    with pytest.raises(ValueError):
        engine.predict(df_scaled, df_raw, 'Modelo Inexistente')
