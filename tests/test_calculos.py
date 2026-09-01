"""
Testes do memorial de cálculo (core/calculos.py).

Um módulo de texto tem dois modos de falhar, e os dois são cobertos aqui.

O primeiro é estrutural: a aba "Sobre" percorre estas tuplas para desenhar a
tabela, então um item com o número errado de campos derruba a tela inteira — e
derruba na abertura da aba, longe de qualquer teste de comportamento.

O segundo é mais silencioso e mais grave: o texto envelhecer. Se alguém
recalibrar os limiares com outro piso de especificidade, ou mudar a banda, o
card continuaria afirmando os números antigos com toda a autoridade de uma
seção chamada "Como os números são calculados". Os testes de sincronia abaixo
comparam o que o texto afirma com o que o código e o ``data/limiares.json``
realmente usam.
"""

import inspect
import json
import os

import pytest

import numpy as np

from core.calculos import (
    CORTES_BRUTOS,
    DUAS_PORCENTAGENS,
    DUAS_PORCENTAGENS_FECHO,
    DUAS_PORCENTAGENS_INTRO,
    ONDE_NO_CODIGO,
    SECOES,
)
from core.decision import BANDA_PADRAO, PoliticaDecisao
from core.inference import ModelLoader
from core.ood_detector import OODDetector


def _todo_o_texto() -> str:
    """Todo o conteúdo do memorial concatenado, para buscas de sincronia."""
    partes = [DUAS_PORCENTAGENS_INTRO, DUAS_PORCENTAGENS_FECHO, ONDE_NO_CODIGO]
    partes += [campo for entrada in DUAS_PORCENTAGENS for campo in entrada]
    for secao in SECOES:
        partes += [secao['titulo'], secao['resumo']]
        partes += [campo for item in secao['itens'] for campo in item]
    return "\n".join(partes)


# --- Estrutura: o que a aba "Sobre" percorre para desenhar ---

def test_todas_as_secoes_tem_a_forma_esperada():
    assert SECOES, "o memorial não pode ficar vazio"
    for secao in SECOES:
        assert set(secao) == {'titulo', 'resumo', 'itens'}
        assert secao['titulo'].strip() and secao['resumo'].strip()
        assert secao['itens'], f"seção sem itens: {secao['titulo']}"


def test_todo_item_tem_nome_formula_e_explicacao():
    """A fórmula pode ser vazia (nem todo cálculo cabe numa linha); as outras não."""
    for secao in SECOES:
        for item in secao['itens']:
            assert len(item) == 3, f"item malformado em {secao['titulo']}: {item}"
            nome, formula, explicacao = item
            assert all(isinstance(campo, str) for campo in item)
            assert nome.strip(), f"item sem nome em {secao['titulo']}"
            assert explicacao.strip(), f"item '{nome}' sem explicação"


def test_bloco_de_abertura_compara_exatamente_duas_porcentagens():
    """São duas colunas lado a lado — nem uma, nem três."""
    assert len(DUAS_PORCENTAGENS) == 2
    for titulo, natureza, exemplo, texto in DUAS_PORCENTAGENS:
        assert titulo.strip() and natureza.strip() and exemplo.strip() and texto.strip()


def test_nomes_de_item_nao_se_repetem_dentro_da_secao():
    """Nome repetido numa mesma tabela é sintoma de item duplicado por engano."""
    for secao in SECOES:
        nomes = [nome for nome, _, _ in secao['itens']]
        assert len(nomes) == len(set(nomes)), f"nome repetido em {secao['titulo']}"


# --- Sincronia: o texto ainda descreve o programa que existe ---

def test_a_banda_citada_e_a_banda_em_vigor():
    texto = _todo_o_texto()
    pontos = f"±{BANDA_PADRAO * 100:.0f} pontos"
    assert pontos in texto, f"o memorial não cita a banda atual ({pontos})"


def test_o_piso_de_especificidade_citado_e_o_do_arquivo_de_limiares(repo_data_dir):
    """O critério do limiar é a pergunta que a banca faz — não pode envelhecer."""
    caminho = os.path.join(repo_data_dir, 'limiares.json')
    with open(caminho, encoding='utf-8') as arquivo:
        piso = json.load(arquivo)['piso_especificidade']

    assert f"especificidade ≥ {piso:.0f}%" in _todo_o_texto()


def test_o_percentil_de_atipicidade_citado_e_o_padrao_do_detector():
    padrao = inspect.signature(OODDetector.__init__).parameters['percentil'].default
    assert f"percentil {padrao:.0f}" in _todo_o_texto()


# --- Os cortes na escala nativa: a seção 7 contra os artefatos reais ---

@pytest.fixture(scope="module")
def loader():
    return ModelLoader()


def _corte_bruto_real(loader, modelo: str, limiar: float) -> float:
    """
    Refaz a inversão da sigmoide de Platt a partir do modelo empacotado.

    De ``P = 1 / (1 + e^(a·s + b))`` sai ``s = ( ln((1−P)/P) − b ) / a``. É a
    mesma conta que produziu os números da seção 7 do memorial; aqui ela é
    repetida sobre o ``wisconsin.pkl`` atual para flagrar o texto envelhecendo.
    """
    calibrador = loader.calibrated_models[modelo].calibrated_classifiers_[0].calibrators[0]
    return float((np.log((1 - limiar) / limiar) - calibrador.b_) / calibrador.a_)


@pytest.mark.parametrize("modelo", sorted(CORTES_BRUTOS))
def test_o_limiar_citado_e_o_do_arquivo_de_limiares(modelo):
    limiar_citado, _ = CORTES_BRUTOS[modelo]
    assert PoliticaDecisao.carregar().limiar(modelo) == pytest.approx(limiar_citado, abs=1e-4)


@pytest.mark.parametrize("modelo", sorted(CORTES_BRUTOS))
def test_o_corte_bruto_citado_bate_com_o_modelo_empacotado(loader, modelo):
    """
    O número que a aba "Sobre" anuncia precisa ser o que a inversão devolve.

    É a afirmação mais frágil do memorial — depende do .pkl E do JSON de
    limiares —, e também a mais visada: é ela que responde "o limiar de 12% do
    Random Forest são 12% das árvores?". Se qualquer um dos dois artefatos for
    regerado, este teste falha antes de a tela passar a mentir.
    """
    limiar_citado, bruto_citado = CORTES_BRUTOS[modelo]
    real = _corte_bruto_real(loader, modelo, limiar_citado)
    assert real == pytest.approx(bruto_citado, abs=0.01)


def test_a_arvore_nao_tem_corte_bruto_para_citar():
    """Folhas puras: não há escore contínuo em que um limiar pudesse cair."""
    assert 'Árvore de Decisão' not in CORTES_BRUTOS


@pytest.mark.parametrize("termo", [
    "Certeza_Maligno(%)",   # a coluna que a dúvida original provocou
    "out-of-fold",          # como os cortes foram medidos sem tocar no teste
    "Platt",                # o que transforma escore em porcentagem
    "Mahalanobis",          # o perfil atípico
    "Wilson",               # os intervalos de confiança
    "Shapley",              # o SHAP
])
def test_o_memorial_cobre_os_calculos_centrais(termo):
    """Cada termo aqui é um cálculo que a defesa vai cobrar explicação."""
    assert termo in _todo_o_texto()
