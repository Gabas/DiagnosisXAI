"""
Testes do BatchProcessor (core/batch_processor.py).

``BatchProcessor`` instancia seu próprio ``ModelLoader`` internamente (sem
ponto de injeção), então estes testes são de integração: exercitam o
``data/wisconsin.pkl`` real, versionado no repositório. Isso também serve
como um smoke test do próprio artefato — se o notebook for reexecutado e
gerar um .pkl com um schema diferente, estes testes acusam.
"""

import os

import pandas as pd
import pytest

from core.batch_processor import BatchProcessor


@pytest.fixture(scope="module")
def csv_bruto(repo_data_dir):
    caminho = os.path.join(repo_data_dir, 'dataTeste_sem_diagnostico.csv')
    return pd.read_csv(caminho)


def test_process_produz_30_colunas_padronizadas_e_brutas(csv_bruto):
    processor = BatchProcessor()
    df_scaled, df_raw = processor.process(csv_bruto)

    assert df_scaled.shape[1] == 30
    assert df_raw.shape[1] == 30
    assert list(df_scaled.columns) == list(df_raw.columns)
    assert df_scaled.shape[0] == len(csv_bruto)


def test_process_padroniza_dados_brutos_para_z_score(csv_bruto):
    processor = BatchProcessor()
    df_scaled, df_raw = processor.process(csv_bruto)

    # CSV de entrada está em escala bruta (area_mean na casa das centenas);
    # a saída padronizada deve ter média ~0, e a "limpa" deve manter a escala.
    assert abs(df_scaled['area_mean'].mean()) < 1.0
    assert df_raw['area_mean'].mean() > 10


def test_process_remove_colunas_nao_preditivas(csv_bruto):
    df = csv_bruto.copy()
    df.insert(0, 'id', range(len(df)))
    df['diagnosis'] = 'M'

    processor = BatchProcessor()
    df_scaled, df_raw = processor.process(df)

    assert 'id' not in df_scaled.columns
    assert 'diagnosis' not in df_scaled.columns


def test_process_e_idempotente_em_dados_ja_padronizados(csv_bruto):
    """Reaplicar process() sobre a saída já padronizada não deve tentar
    padronizar de novo (a heurística de detecção é pela escala de area_mean)."""
    processor = BatchProcessor()
    df_scaled, _ = processor.process(csv_bruto)
    df_scaled_2, _ = processor.process(df_scaled)

    pd.testing.assert_frame_equal(df_scaled, df_scaled_2, check_exact=False, atol=1e-8)
