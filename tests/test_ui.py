"""
Testes dos utilitários de layout responsivo (utils/ui.py).

Exercitam apenas o cálculo — quanto encolher figuras e listas conforme a altura
da tela —, sem abrir janelas. É a parte que decide se o conteúdo de um relatório
cabe num notebook, e a que dá para verificar sem um servidor gráfico: as funções
aceitam a altura da tela como parâmetro justamente para isso.
"""

import pytest

from utils.ui import (
    ALTURA_REFERENCIA,
    FATOR_MINIMO,
    fator_tela,
    figura_responsiva,
    itens_visiveis,
)


def test_tela_de_referencia_nao_encolhe_nada():
    assert fator_tela(None, ALTURA_REFERENCIA) == 1.0


def test_tela_maior_que_a_referencia_nao_amplia():
    """O layout foi desenhado para caber, não para esticar."""
    assert fator_tela(None, 2160) == 1.0


def test_telas_menores_encolhem_proporcionalmente():
    assert fator_tela(None, 900) == pytest.approx(0.75)
    assert fator_tela(None, 768) == pytest.approx(0.64)


def test_fator_tem_piso():
    """Abaixo do piso os gráficos ficariam ilegíveis; quem resolve é a rolagem."""
    assert fator_tela(None, 400) == FATOR_MINIMO


def test_figura_encolhe_mais_na_altura_do_que_na_largura():
    largura, altura = figura_responsiva(None, 5.0, 4.0, altura_tela=768)
    assert altura == pytest.approx(4.0 * 0.64)
    assert largura > 5.0 * 0.64      # a falta de espaço é sobretudo vertical
    assert largura < 5.0


def test_figura_intacta_na_tela_de_referencia():
    assert figura_responsiva(None, 5.0, 3.8, altura_tela=ALTURA_REFERENCIA) == (5.0, 3.8)


def test_itens_visiveis_reduz_a_lista_em_tela_pequena():
    assert itens_visiveis(None, 10, altura_tela=ALTURA_REFERENCIA) == 10
    assert itens_visiveis(None, 10, altura_tela=768) == 6


def test_itens_visiveis_respeita_o_minimo():
    """Um ranking com menos de 5 biomarcadores deixaria de ser informativo."""
    assert itens_visiveis(None, 10, minimo=5, altura_tela=400) == 6
    assert itens_visiveis(None, 6, minimo=5, altura_tela=400) == 5
