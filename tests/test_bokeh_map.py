"""
Testes do gerador do mapa populacional interativo (utils/bokeh_map.py).

Verificam o contrato do artefato HTML — autocontido (BokehJS embutido, sem CDN,
para funcionar offline) e com os dados dos pacientes embutidos para o hover —
sem precisar de navegador ou display.
"""

import numpy as np
import pytest

from utils.bokeh_map import gerar_mapa_html


@pytest.fixture
def dados():
    rng = np.random.default_rng(0)
    return {
        'train_2d': rng.normal(size=(200, 2)),
        'train_y': rng.integers(0, 2, 200),
        'batch_2d': rng.normal(size=(3, 2)),
        'pacientes': [
            {'indice': 0, 'classe': 'Maligno', 'certeza': '98.99%', 'perfil': 'Típico', 'decisao': 'Definida'},
            {'indice': 1, 'classe': 'Benigno', 'certeza': '54.13%', 'perfil': 'Típico', 'decisao': 'Limítrofe'},
            {'indice': 2, 'classe': 'Maligno', 'certeza': '99.00%', 'perfil': 'Atípico', 'decisao': 'Definida'},
        ],
    }


def test_gera_html_autocontido(tmp_path, dados):
    saida = tmp_path / "mapa.html"
    caminho = gerar_mapa_html(dados['train_2d'], dados['train_y'], dados['batch_2d'],
                              dados['pacientes'], caminho_saida=str(saida))
    assert caminho == str(saida)
    html = saida.read_text(encoding='utf-8')
    assert 'Bokeh' in html                      # BokehJS presente
    assert 'cdn.bokeh.org' not in html          # embutido (INLINE), funciona offline
    assert len(html) > 100_000                  # BokehJS inline -> arquivo grande


def test_dados_dos_pacientes_estao_no_html(tmp_path, dados):
    saida = tmp_path / "mapa.html"
    gerar_mapa_html(dados['train_2d'], dados['train_y'], dados['batch_2d'],
                    dados['pacientes'], caminho_saida=str(saida))
    html = saida.read_text(encoding='utf-8')
    # Bokeh escapa acentos como \uXXXX no JSON embutido — checamos as duas formas.
    assert '98.99%' in html and '54.13%' in html
    assert 'Definida' in html
    assert 'Atípico' in html or 'At\\u00edpico' in html
    assert 'Limítrofe' in html or 'Lim\\u00edtrofe' in html


def test_funciona_sem_campos_de_confiabilidade(tmp_path):
    # Pacientes só com indice/classe (ex.: modo 'Todos' ou Árvore) — sem quebrar.
    pacientes = [{'indice': 0, 'classe': 'Benigno'}, {'indice': 1, 'classe': 'Maligno'}]
    saida = tmp_path / "mapa.html"
    caminho = gerar_mapa_html(np.zeros((10, 2)), np.zeros(10, dtype=int),
                              np.zeros((2, 2)), pacientes, caminho_saida=str(saida))
    html = open(caminho, encoding='utf-8').read()
    assert 'Bokeh' in html
    # campos ausentes viram travessão no hover
    assert '\\u2014' in html or '—' in html
