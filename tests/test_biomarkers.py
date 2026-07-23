"""
Testes do glossário de biomarcadores (core/biomarkers.py).

Garantem que a lógica de parsing/descrição cobre exatamente as 30 colunas reais
da base Wisconsin (incluindo os nomes com espaço, como ``concave points_mean``,
e com underscore no meio, como ``fractal_dimension_worst``) e as colunas de
resultado geradas pelo app — o contrato usado pelos tooltips da tabela.
"""

import os

import pandas as pd
import pytest

from core.biomarkers import (
    BIOMARCADORES_BASE,
    ESTATISTICAS,
    GLOSSARIO_INTRO,
    GLOSSARIO_UNIDADES,
    descricao_coluna,
    rotulo_amigavel,
    separar_coluna,
)


@pytest.fixture(scope="module")
def colunas_reais(repo_data_dir):
    """As 30 colunas de biomarcadores, lidas do data.csv real do repositório."""
    caminho = os.path.join(repo_data_dir, 'data.csv')
    cols = pd.read_csv(caminho, nrows=1).columns.tolist()
    return [c for c in cols if c not in ('id', 'Unnamed: 32', 'diagnosis')]


def test_ha_30_colunas_e_todas_tem_descricao(colunas_reais):
    assert len(colunas_reais) == 30
    for col in colunas_reais:
        assert descricao_coluna(col), f'sem descrição para {col!r}'


def test_separar_coluna_lida_com_espaco_e_underscore():
    assert separar_coluna('radius_worst') == ('radius', 'worst')
    assert separar_coluna('concave points_mean') == ('concave points', 'mean')
    assert separar_coluna('fractal_dimension_se') == ('fractal_dimension', 'se')


def test_separar_coluna_sem_sufixo_conhecido():
    assert separar_coluna('Diagnóstico_IA') == ('Diagnóstico_IA', None)


def test_rotulo_amigavel():
    assert rotulo_amigavel('radius_worst') == 'Raio (pior)'
    assert rotulo_amigavel('concave points_mean') == 'Pontos côncavos (média)'
    assert rotulo_amigavel('Certeza_Maligno(%)') is None


def test_descricao_inclui_unidade_e_estatistica():
    texto = descricao_coluna('area_mean')
    assert 'pixels²' in texto
    assert 'média' in texto


def test_descricao_colunas_de_resultado():
    assert 'Maligno' in descricao_coluna('Diagnóstico_IA')
    assert 'calibrada' in descricao_coluna('Certeza_Maligno(%)')
    assert 'RF' in descricao_coluna('IA_RF')
    assert 'Atípico' in descricao_coluna('Perfil')


def test_descricao_none_para_desconhecida():
    assert descricao_coluna('coluna_inexistente') is None
    assert descricao_coluna(None) is None


def test_estrutura_do_glossario():
    # 10 medições-base, cada uma com (rótulo, o que mede, unidade).
    assert len(BIOMARCADORES_BASE) == 10
    assert all(len(v) == 3 and all(v) for v in BIOMARCADORES_BASE.values())
    # 3 estatísticas, cada uma com (rótulo, explicação).
    assert set(ESTATISTICAS) == {'mean', 'se', 'worst'}
    # Textos do card presentes e mencionando pontos-chave (PAAF e Z-score).
    assert 'PAAF' in GLOSSARIO_INTRO
    assert 'Z-score' in GLOSSARIO_UNIDADES
