"""
Testes da validação da planilha de entrada (core/validacao.py).

O módulo é puro — recebe DataFrames e a lista de colunas esperadas —, então
estes testes não dependem do ``wisconsin.pkl``. Além de verificar que cada
planilha malformada é recusada, eles checam o *conteúdo* da mensagem: ela
precisa nomear a coluna e a linha, que é a razão de o validador existir (o erro
cru do scikit-learn não diz nem uma coisa nem outra).
"""

import numpy as np
import pandas as pd
import pytest

from core.validacao import (LoteInvalido, validar_colunas,
                            validar_nao_negativos, validar_valores)

COLUNAS = ['radius_mean', 'texture_mean', 'area_mean']


@pytest.fixture
def lote():
    """Lote pequeno e válido, em escala bruta."""
    return pd.DataFrame({
        'radius_mean': [13.5, 14.2, 12.8],
        'texture_mean': [20.1, 18.7, 22.4],
        'area_mean': [560.0, 620.0, 510.0],
    })


# --- validar_colunas -------------------------------------------------------

def test_lote_valido_passa(lote):
    validar_colunas(lote, COLUNAS)
    validar_valores(lote)
    validar_nao_negativos(lote)


def test_planilha_sem_linhas_e_recusada(lote):
    with pytest.raises(LoteInvalido, match="nenhuma linha de dados"):
        validar_colunas(lote.iloc[0:0], COLUNAS)


def test_coluna_faltando_e_nomeada_na_mensagem(lote):
    with pytest.raises(LoteInvalido) as erro:
        validar_colunas(lote.drop(columns=['texture_mean']), COLUNAS)
    assert 'texture_mean' in str(erro.value)
    assert 'Falta 1' in str(erro.value)   # singular


def test_varias_colunas_faltando_usam_plural(lote):
    with pytest.raises(LoteInvalido) as erro:
        validar_colunas(lote[['area_mean']], COLUNAS)
    assert 'Faltam 2' in str(erro.value)


def test_coluna_extra_nao_e_erro(lote):
    lote['id_do_laboratorio'] = [1, 2, 3]
    validar_colunas(lote, COLUNAS)   # não levanta


# --- validar_valores -------------------------------------------------------

def test_celula_vazia_e_recusada_com_coluna_e_linha(lote):
    lote.loc[1, 'texture_mean'] = np.nan
    with pytest.raises(LoteInvalido) as erro:
        validar_valores(lote)
    msg = str(erro.value)
    assert 'texture_mean' in msg
    assert 'linha 3' in msg          # posição 1 -> 3ª linha do arquivo
    assert '1 célula vazia' in msg   # singular


def test_varias_celulas_vazias_usam_plural(lote):
    lote.loc[0, 'texture_mean'] = np.nan
    lote.loc[2, 'area_mean'] = np.nan
    with pytest.raises(LoteInvalido, match="2 células vazias"):
        validar_valores(lote)


def test_texto_em_coluna_numerica_e_recusado(lote):
    lote['radius_mean'] = lote['radius_mean'].astype(object)
    lote.loc[2, 'radius_mean'] = 'doze vírgula oito'
    with pytest.raises(LoteInvalido) as erro:
        validar_valores(lote)
    msg = str(erro.value)
    assert 'radius_mean' in msg
    assert 'linha 4' in msg
    assert 'doze vírgula oito' in msg


def test_texto_e_apontado_antes_de_celula_vazia(lote):
    """
    A ordem importa: uma coluna com texto no meio não tem média, e é da média
    de area_mean que o BatchProcessor deduz a escala do lote.
    """
    lote['radius_mean'] = lote['radius_mean'].astype(object)
    lote.loc[0, 'radius_mean'] = 'abc'
    lote.loc[1, 'texture_mean'] = np.nan
    with pytest.raises(LoteInvalido, match="texto onde era esperado número"):
        validar_valores(lote)


def test_numero_como_texto_nao_e_erro(lote):
    """Planilha exportada com números entre aspas ainda é numérica."""
    lote['radius_mean'] = lote['radius_mean'].astype(str)
    validar_valores(lote)   # não levanta


# --- validar_nao_negativos -------------------------------------------------

def test_negativo_em_escala_bruta_e_recusado(lote):
    lote.loc[1, 'area_mean'] = -3.0
    with pytest.raises(LoteInvalido) as erro:
        validar_nao_negativos(lote)
    msg = str(erro.value)
    assert 'area_mean' in msg
    assert 'linha 3' in msg


def test_zero_nao_e_negativo(lote):
    """O mínimo da base WDBC é 0,0 — zero é um valor legítimo."""
    lote.loc[0, 'area_mean'] = 0.0
    validar_nao_negativos(lote)   # não levanta
