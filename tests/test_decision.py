"""
Testes da política de decisão (core/decision.py).

Cobrem as três garantias que o módulo existe para dar: o rótulo sai do limiar
de operação (e não dos 50% implícitos do sklearn), a faixa de revisão acompanha
esse limiar, e as explicações do Passo 5 passam a repetir exatamente a decisão
que a tabela do Passo 3 mostra.
"""

import json

import pytest

from core.decision import (
    BANDA_PADRAO,
    LIMIAR_NEUTRO,
    PoliticaDecisao,
    ROTULO_REVISAR,
    ZONA_DEFINIDA,
    ZONA_LIMITROFE,
    ZONA_REVISAO,
    aplicar_a_explicacoes,
    limitrofe_padrao,
)


@pytest.fixture
def politica():
    """Política com limiares calibrados e sem faixa de recusa (nada a adiar)."""
    return PoliticaDecisao({'SVM': 0.15, 'KNN': 0.30}, banda=0.10,
                           metadados={'piso_especificidade': 92.0})


def test_limiar_conhecido_e_desconhecido(politica):
    assert politica.limiar('SVM') == 0.15
    assert politica.limiar('Modelo Inexistente') == LIMIAR_NEUTRO


def test_rotulo_usa_o_limiar_do_modelo(politica):
    # 0,20 é Maligno para o SVM (limiar 0,15) e Benigno para o KNN (limiar 0,30).
    assert politica.rotular(0.20, 'SVM') == 'Maligno'
    assert politica.rotular(0.20, 'KNN') == 'Benigno'
    # A borda pertence à classe positiva: em rastreio, empate vai para o lado seguro.
    assert politica.rotular(0.15, 'SVM') == 'Maligno'


def test_zona_de_revisao_acompanha_o_limiar(politica):
    assert politica.zona(0.15, 'SVM') == ZONA_LIMITROFE
    assert politica.zona(0.25, 'SVM') == ZONA_LIMITROFE       # borda inclusiva
    assert politica.zona(0.26, 'SVM') == ZONA_DEFINIDA
    # 0,50 seria "limítrofe" sob a régua antiga; com limiar 0,15 é decisão folgada.
    assert politica.zona(0.50, 'SVM') == ZONA_DEFINIDA


def test_politica_neutra_reproduz_o_comportamento_do_sklearn():
    neutra = PoliticaDecisao()
    assert not neutra.calibrada
    assert neutra.rotular(0.49, 'SVM') == 'Benigno'
    assert neutra.rotular(0.50, 'SVM') == 'Maligno'


def test_calibrada_so_quando_algum_limiar_sai_de_050():
    assert not PoliticaDecisao({'SVM': 0.5}).calibrada
    assert PoliticaDecisao({'SVM': 0.2}).calibrada


def test_carregar_arquivo(tmp_path):
    caminho = tmp_path / 'limiares.json'
    caminho.write_text(json.dumps({
        'limiares': {'SVM': 0.2}, 'banda_revisao': 0.05, 'piso_especificidade': 92.0,
    }), encoding='utf-8')

    politica = PoliticaDecisao.carregar(str(caminho))
    assert politica.limiar('SVM') == 0.2
    assert politica.banda == 0.05
    assert politica.metadados['piso_especificidade'] == 92.0


def test_carregar_sem_arquivo_recai_na_politica_neutra(tmp_path):
    """Sem data/limiares.json o app precisa continuar funcionando como antes."""
    politica = PoliticaDecisao.carregar(str(tmp_path / 'inexistente.json'))
    assert not politica.calibrada
    assert politica.limiar('SVM') == LIMIAR_NEUTRO


def test_carregar_arquivo_corrompido_nao_derruba_o_app(tmp_path):
    caminho = tmp_path / 'limiares.json'
    caminho.write_text('{ isto não é json', encoding='utf-8')
    assert not PoliticaDecisao.carregar(str(caminho)).calibrada


def test_justificativa_traz_o_desempenho_medido_no_treino():
    politica = PoliticaDecisao(
        {'SVM': 0.15},
        metadados={'piso_especificidade': 92.0,
                   'desempenho_treino': {'SVM': {'sensibilidade': 97.5, 'especificidade': 92.1}}},
    )
    texto = politica.justificativa('SVM')
    assert 'sensibilidade 97.5%' in texto
    assert 'especificidade 92.1%' in texto


def test_justificativa_da_politica_neutra_admite_o_corte_de_50(politica):
    """Sem calibração, o texto não pode fingir que houve escolha de ponto."""
    texto = PoliticaDecisao().justificativa('SVM')
    assert '50%' in texto and 'scikit-learn' in texto


def test_regra_condensa_as_faixas_numa_linha(politica):
    linha = politica.regra('SVM')
    assert 'Benigno: certeza < 5.0%' in linha
    assert 'Limítrofe: 5.0% a 25.0%' in linha
    assert 'Maligno: certeza > 25.0%' in linha


# --- Régua de decisão: as faixas e o porquê dos cortes ---

def test_regua_cobre_de_ponta_a_ponta_sem_buracos(politica):
    """As faixas precisam ser contíguas: todo paciente cai em exatamente uma."""
    faixas = politica.regua('SVM')
    assert faixas[0]['inferior'] == 0.0
    assert faixas[-1]['superior'] == 1.0
    for anterior, seguinte in zip(faixas, faixas[1:]):
        assert anterior['superior'] == seguinte['inferior']


def test_regua_sem_recusa_tem_a_faixa_limitrofe_no_meio(politica):
    benigno, incerta, maligno = politica.regua('SVM')
    assert benigno['rotulo'] == 'Benigno'
    assert maligno['rotulo'] == 'Maligno'
    # A faixa incerta é o limiar ± banda: 0,15 ± 0,10.
    assert incerta['rotulo'] == ZONA_LIMITROFE
    assert (incerta['inferior'], incerta['superior']) == pytest.approx((0.05, 0.25))
    assert '5.0% a 25.0%' in incerta['faixa']


def test_regua_com_recusa_tem_a_faixa_de_recusa_no_meio(com_recusa):
    benigno, incerta, maligno = com_recusa.regua('SVM')
    assert incerta['rotulo'] == ROTULO_REVISAR
    assert (incerta['inferior'], incerta['superior']) == pytest.approx((0.01, 0.70))
    # Os cortes da régua são os mesmos que rotular() aplica.
    assert com_recusa.rotular(benigno['superior'] - 1e-6, 'SVM') == 'Benigno'
    assert com_recusa.rotular(maligno['inferior'], 'SVM') == 'Maligno'


def test_regua_omite_faixa_vazia():
    """Limiar baixo demais para caber uma faixa Benigno abaixo dele."""
    faixas = PoliticaDecisao({'SVM': 0.05}, banda=0.10).regua('SVM')
    assert [f['rotulo'] for f in faixas] == [ZONA_LIMITROFE, 'Maligno']
    assert faixas[0]['inferior'] == 0.0


def test_faixa_incerta_muda_com_a_recusa(politica, com_recusa):
    assert politica.faixa_incerta('SVM') == pytest.approx((0.05, 0.25))
    assert com_recusa.faixa_incerta('SVM') == pytest.approx((0.01, 0.70))


def test_justificativa_do_limiar_responde_por_que_nao_50(politica):
    texto = politica.justificativa('SVM')
    assert '15.0%' in texto and '50%' in texto
    assert 'falso negativo' in texto
    assert 'especificidade ≥ 92%' in texto
    assert 'teste não participou' in texto      # a escolha não vazou o teste


def test_justificativa_da_recusa_explica_os_dois_cortes(com_recusa):
    texto = com_recusa.justificativa('SVM')
    assert '1.0%' in texto and '70.0%' in texto
    assert 'menor certeza que um paciente maligno' in texto
    # O limiar cai dentro da faixa: quem decide passam a ser os dois cortes.
    assert '15.0%' in texto


def test_justificativa_da_recusa_com_tolerancia_nao_promete_erro_zero():
    politica = PoliticaDecisao(
        {'SVM': 0.15}, faixas_revisao={'SVM': (0.05, 0.60)},
        metadados={'tolerancia_recusa': 0.02})
    politica.adiar_incertos = True
    texto = politica.justificativa('SVM')
    assert '2% mais extremos' in texto
    assert 'menor certeza que um paciente maligno' not in texto


# --- Opção de recusa (coluna "Revisar") ---

@pytest.fixture
def com_recusa():
    """Política com faixa de recusa calibrada para o SVM (ligada, como no padrão)."""
    return PoliticaDecisao(
        {'SVM': 0.15, 'KNN': 0.30}, banda=0.10,
        faixas_revisao={'SVM': (0.01, 0.70)},
        metadados={'cobertura_treino': {'SVM': 64.8}},
    )


def test_recusa_comeca_ligada(com_recusa):
    """
    O padrão seguro é não decidir o que não dá para decidir.

    Entre devolver um caso incerto e arriscar um palpite que pode liberar um
    tumor maligno, o programa devolve — e quem quiser o palpite desliga a
    recusa explicitamente.
    """
    assert PoliticaDecisao().adiar_incertos is True
    assert com_recusa.rotular(0.20, 'SVM') == ROTULO_REVISAR


def test_recusa_desligada_volta_a_decidir_tudo(com_recusa):
    com_recusa.adiar_incertos = False
    assert com_recusa.rotular(0.20, 'SVM') == 'Maligno'   # decide, mesmo incerto


def test_recusa_divide_em_tres_saidas(com_recusa):
    assert com_recusa.rotular(0.005, 'SVM') == 'Benigno'      # abaixo da faixa
    assert com_recusa.rotular(0.80, 'SVM') == 'Maligno'       # acima da faixa
    assert com_recusa.rotular(0.70, 'SVM') == 'Maligno'       # limite superior decide
    assert com_recusa.rotular(0.35, 'SVM') == ROTULO_REVISAR  # dentro: não decide
    assert com_recusa.rotular(0.01, 'SVM') == ROTULO_REVISAR  # limite inferior não decide


def test_recusa_so_vale_para_modelo_com_faixa(com_recusa):
    """O KNN não tem faixa: continua decidindo tudo, mesmo com a recusa ligada."""
    assert not com_recusa.pode_adiar('KNN')
    assert com_recusa.rotular(0.35, 'KNN') == 'Maligno'
    assert com_recusa.pode_adiar('SVM')


def test_zona_de_um_caso_adiado(com_recusa):
    assert com_recusa.zona(0.35, 'SVM') == ZONA_REVISAO
    assert com_recusa.zona(0.005, 'SVM') == ZONA_DEFINIDA


def test_justificativa_da_recusa_descreve_a_faixa_e_a_cobertura(com_recusa):
    texto = com_recusa.justificativa('SVM')
    assert '1.0%' in texto and '70.0%' in texto
    assert 'decidiu 65% dos casos' in texto


def test_aplicar_marca_explicacoes_adiadas(com_recusa):
    explicacoes = aplicar_a_explicacoes(_explicacoes(), [0.35, 0.005, 0.99], com_recusa, 'SVM')
    assert explicacoes[0]['classe'] == ROTULO_REVISAR
    assert explicacoes[1]['classe'] == 'Benigno'
    assert explicacoes[2]['classe'] == 'Maligno'
    # Sem classe decidida, exibe-se a confiança no lado para o qual pendeu.
    assert explicacoes[0]['confianca'] == 65.0


def test_carregar_le_a_faixa_de_recusa(tmp_path):
    caminho = tmp_path / 'limiares.json'
    caminho.write_text(json.dumps({
        'limiares': {'SVM': 0.2}, 'faixas_recusa': {'SVM': [0.01, 0.7]},
    }), encoding='utf-8')
    politica = PoliticaDecisao.carregar(str(caminho))
    assert politica.faixa_recusa('SVM') == (0.01, 0.7)
    assert politica.faixa_recusa('KNN') is None


def test_limitrofe_padrao_gira_em_torno_de_meio():
    assert limitrofe_padrao(LIMIAR_NEUTRO)
    assert limitrofe_padrao(LIMIAR_NEUTRO + BANDA_PADRAO)
    assert not limitrofe_padrao(LIMIAR_NEUTRO + BANDA_PADRAO + 0.01)


# --- Alinhamento das explicações à decisão exibida ---

def _explicacoes():
    """Três pacientes no formato do RandomForestExplainer (o mais completo)."""
    return [
        {'indice': 0, 'classe': 'Benigno', 'probabilidade': 30.0, 'confianca': 70.0,
         'limitrofe': False, 'votos_maligno': 120},
        {'indice': 1, 'classe': 'Benigno', 'probabilidade': 10.0, 'confianca': 90.0,
         'limitrofe': False, 'votos_maligno': 40},
        {'indice': 2, 'classe': 'Maligno', 'probabilidade': 95.0, 'confianca': 95.0,
         'limitrofe': False, 'votos_maligno': 480},
    ]


def test_aplicar_reescreve_classe_probabilidade_e_limitrofe(politica):
    explicacoes = aplicar_a_explicacoes(_explicacoes(), [0.30, 0.02, 0.99], politica, 'SVM')

    # 0,30 está acima do limiar de 0,15: o relatório passa a dizer Maligno,
    # como a tabela — antes diria Benigno, por decidir em 0,5.
    assert explicacoes[0]['classe'] == 'Maligno'
    assert explicacoes[0]['probabilidade'] == 30.0
    assert explicacoes[0]['confianca'] == 30.0     # confiança na classe predita
    assert explicacoes[1]['classe'] == 'Benigno'
    assert explicacoes[1]['confianca'] == 98.0
    assert explicacoes[2]['classe'] == 'Maligno'


def test_aplicar_marca_limitrofe_pela_distancia_ao_limiar(politica):
    explicacoes = aplicar_a_explicacoes(_explicacoes(), [0.18, 0.02, 0.99], politica, 'SVM')
    assert explicacoes[0]['limitrofe'] is True      # 0,18 dista 0,03 do limiar
    assert explicacoes[1]['limitrofe'] is False
    assert explicacoes[2]['limitrofe'] is False


def test_aplicar_preserva_o_conteudo_explicativo(politica):
    """Só a decisão é reescrita; o raciocínio do modelo permanece intacto."""
    explicacoes = aplicar_a_explicacoes(_explicacoes(), [0.30, 0.02, 0.99], politica, 'SVM')
    assert [e['votos_maligno'] for e in explicacoes] == [120, 40, 480]


def test_aplicar_atualiza_campo_certeza_quando_e_esse_o_nome(politica):
    """A Árvore nomeia o campo 'certeza'; os demais, 'confianca'/'probabilidade'."""
    explicacoes = [{'indice': 0, 'classe': 'Benigno', 'certeza': 100.0}]
    aplicar_a_explicacoes(explicacoes, [0.80], politica, 'SVM')
    assert explicacoes[0]['classe'] == 'Maligno'
    assert explicacoes[0]['certeza'] == 80.0


def test_aplicar_sem_probabilidades_devolve_intacto(politica):
    original = _explicacoes()
    assert aplicar_a_explicacoes(original, None, politica, 'SVM') == original


def test_aplicar_ignora_tamanho_incompativel(politica):
    """Cenário impossível no fluxo normal, mas não deve derrubar o relatório."""
    original = _explicacoes()
    assert aplicar_a_explicacoes(original, [0.5], politica, 'SVM') == original
