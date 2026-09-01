"""
Testes da explicabilidade do comitê (core/committee.py).

O foco é a classificação do motivo: é o que o relatório do comitê acrescenta à
média das probabilidades. Dois pacientes com a mesma probabilidade final podem
ter chegado lá por caminhos opostos — quatro modelos de acordo, ou dois pares
se cancelando —, e um adiamento por desacordo pede conduta diferente de um
adiamento causado pela largura da faixa de recusa.
"""

import pytest

from core.committee import (
    MOTIVO_CAUTELA,
    MOTIVO_CONSENSO,
    MOTIVO_DISCORDANCIA,
    MOTIVO_FRONTEIRA,
    MOTIVO_MAIORIA,
    explicar,
)
from core.decision import PoliticaDecisao

COMITE = 'Comitê (voto suave)'
MEMBROS = ('Regressão Logística', 'Random Forest', 'SVM', 'KNN')


@pytest.fixture
def politica():
    """Membros e comitê com limiar 0,15 e recusa em [0,01; 0,70) — ligada, como no app."""
    return PoliticaDecisao(
        {m: 0.15 for m in MEMBROS} | {COMITE: 0.15}, banda=0.10,
        faixas_revisao={COMITE: (0.01, 0.70)},
    )


@pytest.fixture
def sem_recusa(politica):
    """A mesma política com a recusa desligada — o comitê volta a decidir tudo."""
    politica.adiar_incertos = False
    return politica


def _explicar(politica, *colunas):
    """Explica um lote em que cada tupla é a leitura dos 4 membros de um paciente."""
    probabilidades = {m: [linha[i] for linha in colunas] for i, m in enumerate(MEMBROS)}
    return explicar(probabilidades, range(len(colunas)), politica, COMITE)['explicacoes']


def test_media_dos_membros_e_a_decisao_do_comite(sem_recusa):
    (e,) = _explicar(sem_recusa, (0.10, 0.20, 0.30, 0.40))
    assert e['probabilidade'] == 25.0
    assert e['classe'] == 'Maligno'
    assert e['votos_maligno'] == 3      # 0,20 / 0,30 / 0,40 acima de 0,15


def test_consenso_quando_todos_apontam_o_mesmo_lado(politica):
    (e,) = _explicar(politica, (0.90, 0.95, 0.88, 0.99))
    assert e['classe'] == 'Maligno'
    assert e['motivo'] == MOTIVO_CONSENSO
    assert e['discordantes'] == []


def test_maioria_quando_um_membro_diverge(politica):
    (e,) = _explicar(politica, (0.90, 0.95, 0.05, 0.99))
    assert e['motivo'] == MOTIVO_MAIORIA
    assert e['discordantes'] == ['SVM']


def test_discordancia_quando_os_membros_se_dividem(politica):
    """A média cai na faixa por cancelamento — não porque alguém esteja em dúvida."""
    (e,) = _explicar(politica, (0.95, 0.02, 0.90, 0.03))
    assert e['classe'] == 'Revisar'
    assert e['motivo'] == MOTIVO_DISCORDANCIA
    assert e['amplitude'] == pytest.approx(93.0)


def test_fronteira_quando_a_media_fica_em_cima_do_limiar(politica):
    (e,) = _explicar(politica, (0.16, 0.17, 0.15, 0.18))
    assert e['classe'] == 'Revisar'
    assert e['motivo'] == MOTIVO_FRONTEIRA


def test_cautela_quando_os_membros_concordam_longe_do_limiar(politica):
    """O caso mais comum no lote real: quem adiou foi a política, não o modelo."""
    (e,) = _explicar(politica, (0.02, 0.03, 0.02, 0.04))
    assert e['classe'] == 'Revisar'
    assert e['motivo'] == MOTIVO_CAUTELA
    assert e['votos_maligno'] == 0          # unanimidade...
    assert all(m['convicto'] for m in e['membros'])   # ...e com convicção


def test_sem_recusa_nao_ha_motivo_de_adiamento(sem_recusa):
    """Desligada a recusa, o mesmo paciente é decidido normalmente."""
    (e,) = _explicar(sem_recusa, (0.02, 0.03, 0.02, 0.04))
    assert e['classe'] == 'Benigno'
    assert e['motivo'] == MOTIVO_CONSENSO


def test_margem_e_medida_contra_o_limiar_do_proprio_membro():
    """Comparar probabilidades cruas entre membros seria injusto: os cortes diferem."""
    politica = PoliticaDecisao(
        {'Regressão Logística': 0.10, 'Random Forest': 0.50,
         'SVM': 0.15, 'KNN': 0.15, COMITE: 0.15})
    (e,) = _explicar(politica, (0.30, 0.30, 0.30, 0.30))
    por_nome = {m['nome']: m for m in e['membros']}
    assert por_nome['Regressão Logística']['classe'] == 'Maligno'
    assert por_nome['Random Forest']['classe'] == 'Benigno'   # mesma prob., outro corte
    assert por_nome['Regressão Logística']['margem'] == pytest.approx(20.0)
    assert por_nome['Random Forest']['margem'] == pytest.approx(-20.0)


def test_concordancia_e_medida_no_lote_inteiro(politica):
    """
    Medir só sobre os decididos devolveria 100% para todos.

    Decidir é justamente o que o comitê faz quando ninguém discorda, então a
    taxa precisa ser contra a inclinação da média, inclusive nos adiados.
    """
    relatorio = explicar(
        {m: [0.95, 0.02, 0.30] for m in MEMBROS} | {'SVM': [0.02, 0.02, 0.30]},
        range(3), politica, COMITE)

    svm = relatorio['resumo']['por_membro']['SVM']
    assert svm['total'] == 3
    assert svm['acompanhou'] == 2          # discorda apenas no primeiro paciente
    assert svm['taxa'] == pytest.approx(66.7, abs=0.1)
    assert relatorio['resumo']['por_membro']['KNN']['taxa'] == 100.0


def test_relatorio_traz_as_referencias_para_o_grafico(politica):
    relatorio = explicar({m: [0.5] for m in MEMBROS}, [0], politica, COMITE)
    assert relatorio['membros'] == list(MEMBROS)
    assert relatorio['limiar_comite'] == 15.0
    assert relatorio['faixa'] == [1.0, 70.0]
    assert relatorio['limiares']['SVM'] == 15.0


def test_faixa_ausente_quando_a_recusa_esta_desligada(sem_recusa):
    assert explicar({m: [0.5] for m in MEMBROS}, [0], sem_recusa, COMITE)['faixa'] is None