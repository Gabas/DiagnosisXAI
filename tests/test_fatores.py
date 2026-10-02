"""
Testes do bloco "Fatores que mais pesaram" (core/fatores.py).

A maior parte é isolada: verifica as propriedades que fazem do valor de
Shapley a escolha certa — a matriz bate com a definição por permutações, a soma
fecha a certeza (eficiência), um fator ignorado recebe zero (jogador nulo), o
caso linear tem forma fechada, e o comitê é a média dos membros (linearidade).

O teste de integração, ao final, usa o ``wisconsin.pkl`` real para garantir o
que o médico vê: a certeza no fim do bloco é a MESMA da tabela do Passo 3.
"""

import itertools
import math
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.biomarkers import BIOMARCADORES_BASE
from core.decision import NOME_COMITE
from core.fatores import (FatoresDoLote, ShapleyPorFator, _matriz_shapley, agrupar_colunas,
                          combinar, linhas_fatores)

# As 30 colunas na ordem do CSV do WDBC: as 10 médias, os 10 erros padrão, os 10 piores.
COLUNAS_WDBC = [f"{base}_{suf}" for suf in ('mean', 'se', 'worst') for base in BIOMARCADORES_BASE]


# --- agrupamento ----------------------------------------------------------

def test_30_colunas_viram_8_fatores_com_tamanho_fundido():
    grupos = agrupar_colunas(COLUNAS_WDBC)
    nomes = [g[0] for g in grupos]
    assert len(grupos) == 8
    assert nomes[0] == 'tamanho'
    assert not {'radius', 'perimeter', 'area'} & set(nomes)

    tamanho = dict((g[0], g[1]) for g in grupos)['tamanho']
    assert sorted(COLUNAS_WDBC[j] for j in tamanho) == sorted(
        f"{b}_{s}" for b in ('radius', 'perimeter', 'area') for s in ('mean', 'se', 'worst'))
    # Cada coluna em exatamente um fator.
    todas = np.concatenate([g[1] for g in grupos])
    assert sorted(todas.tolist()) == list(range(30))


def test_coluna_estranha_e_recusada():
    with pytest.raises(ValueError, match="biomarcadores"):
        agrupar_colunas(COLUNAS_WDBC[:-1] + ['idade_do_paciente'])


# --- propriedades do valor de Shapley ------------------------------------

def test_matriz_bate_com_a_definicao_por_permutacoes():
    n = 5
    coalizoes, M = _matriz_shapley(n)
    v = np.random.default_rng(1).normal(size=len(coalizoes))
    indice = {tuple(c): i for i, c in enumerate(coalizoes)}

    esperado = np.zeros(n)
    for ordem in itertools.permutations(range(n)):
        presentes = np.zeros(n, dtype=bool)
        for jogador in ordem:
            antes = v[indice[tuple(presentes)]]
            presentes[jogador] = True
            esperado[jogador] += v[indice[tuple(presentes)]] - antes
    esperado /= math.factorial(n)

    np.testing.assert_allclose(M @ v, esperado, atol=1e-12)


@pytest.fixture
def jogo():
    """Fundo aleatório e uma função não linear com interação entre fatores."""
    rng = np.random.default_rng(7)
    fundo = rng.normal(size=(40, 30))
    w = rng.normal(size=30)

    def f(X):
        X = np.asarray(X)
        return 1 / (1 + np.exp(-(X @ w * 0.3 + X[:, 0] * X[:, 1] - X[:, 4] ** 2 * 0.2)))

    return fundo, f, rng.normal(size=30)


def test_eficiencia_a_soma_fecha_a_certeza(jogo):
    fundo, f, x = jogo
    r = ShapleyPorFator(f, fundo, COLUNAS_WDBC).explicar(x)
    assert r['prob'] == pytest.approx(float(f(x[None, :])[0]))
    assert r['base'] == pytest.approx(float(np.mean(f(fundo))))
    assert r['base'] + sum(item['contribuicao'] for item in r['fatores']) == pytest.approx(r['prob'])


def test_fatores_saem_ordenados_por_impacto(jogo):
    fundo, f, x = jogo
    r = ShapleyPorFator(f, fundo, COLUNAS_WDBC).explicar(x)
    impactos = [abs(item['contribuicao']) for item in r['fatores']]
    assert impactos == sorted(impactos, reverse=True)


def test_caso_linear_tem_forma_fechada():
    """f = w·x + c  =>  φ_fator = Σ_{j no fator} w_j (x_j − média do fundo_j)."""
    rng = np.random.default_rng(3)
    fundo, w, x = rng.normal(size=(25, 30)), rng.normal(size=30), rng.normal(size=30)
    explicador = ShapleyPorFator(lambda X: np.asarray(X) @ w + 0.5, fundo, COLUNAS_WDBC)
    r = {item['fator']: item['contribuicao'] for item in explicador.explicar(x)['fatores']}
    for fator, idx, _ in agrupar_colunas(COLUNAS_WDBC):
        esperado = float(np.sum(w[idx] * (x[idx] - fundo[:, idx].mean(axis=0))))
        assert r[fator] == pytest.approx(esperado, abs=1e-10)


def test_fator_que_o_modelo_ignora_recebe_zero():
    rng = np.random.default_rng(4)
    fundo, x = rng.normal(size=(30, 30)), rng.normal(size=30)
    textura = [j for j, c in enumerate(COLUNAS_WDBC) if c.startswith('texture')]
    w = rng.normal(size=30)
    w[textura] = 0.0
    r = ShapleyPorFator(lambda X: np.tanh(np.asarray(X) @ w), fundo, COLUNAS_WDBC).explicar(x)
    textura_phi = next(i['contribuicao'] for i in r['fatores'] if i['fator'] == 'texture')
    assert textura_phi == pytest.approx(0.0, abs=1e-12)


def test_comite_e_a_media_dos_membros_por_linearidade(jogo):
    fundo, f, x = jogo
    g = lambda X: 1 / (1 + np.exp(-np.asarray(X)[:, 5] * 2))  # segundo "membro"
    membros = [ShapleyPorFator(h, fundo, COLUNAS_WDBC).explicar(x) for h in (f, g)]
    media = ShapleyPorFator(lambda X: (f(X) + g(X)) / 2, fundo, COLUNAS_WDBC).explicar(x)

    combinado = combinar(membros)
    assert combinado['prob'] == pytest.approx(media['prob'])
    assert combinado['base'] == pytest.approx(media['base'])
    esperado = {i['fator']: i['contribuicao'] for i in media['fatores']}
    for item in combinado['fatores']:
        assert item['contribuicao'] == pytest.approx(esperado[item['fator']], abs=1e-12)


# --- texto do bloco -------------------------------------------------------

def _resultado(base, contribuicoes):
    fatores = [{'fator': f'f{i}', 'rotulo': f'Fator {i}', 'contribuicao': c, 'z': 0.5}
               for i, c in enumerate(contribuicoes)]
    fatores.sort(key=lambda item: abs(item['contribuicao']), reverse=True)
    return {'base': base, 'prob': base + sum(contribuicoes), 'fatores': fatores}


def test_bloco_lista_seis_e_soma_o_resto_numa_linha():
    r = _resultado(0.375, [0.40, 0.05, -0.02, 0.01, 0.008, -0.005, 0.002, 0.001])
    texto = "\n".join(linhas_fatores(r, 'SVM'))
    assert texto.count(" pp  →") == 6
    assert "Demais 2 fatores" in texto
    assert "37.5%" in texto and f"{r['prob'] * 100:.1f}%" in texto


def test_bloco_avisa_quando_fatores_e_diagnostico_parecem_se_contradizer():
    # Certeza 22% (acima de um corte de ~15%), mas abaixo da média do treino.
    contra = _resultado(0.375, [-0.10, -0.05, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0])
    assert "Atenção" in "\n".join(linhas_fatores(contra, 'KNN', rotulo='Maligno'))
    assert "Atenção" not in "\n".join(linhas_fatores(contra, 'KNN', rotulo='Benigno'))

    a_favor = _resultado(0.375, [0.40, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert "Atenção" not in "\n".join(linhas_fatores(a_favor, 'SVM', rotulo='Maligno'))


def test_aviso_cita_o_corte_que_decidiu():
    contra = _resultado(0.384, [-0.158, -0.058, 0.021, 0.0, 0.0, 0.0, 0.0, 0.0])  # 18.3%
    texto = " ".join(linhas_fatores(contra, 'KNN', rotulo='Maligno', corte=0.17))
    assert "(17.0%)" in " ".join(texto.split())
    # Um corte incoerente com o diagnóstico não é citado (seria uma afirmação falsa).
    texto = " ".join(linhas_fatores(contra, 'KNN', rotulo='Maligno', corte=0.30))
    assert "30.0%" not in texto


def test_corte_do_lote_segue_a_recusa():
    from core.decision import PoliticaDecisao

    politica = PoliticaDecisao(limiares={NOME_COMITE: 0.146, 'KNN': 0.17},
                               faixas_revisao={NOME_COMITE: (0.011, 0.682)})

    class Loader:
        calibrated_models, models, X_train_scaled = {}, {}, None

    fatores = FatoresDoLote(Loader(), np.zeros((1, 30)), np.zeros((1, 30)), [0], politica=politica)
    assert fatores.corte(NOME_COMITE, 'Maligno') == pytest.approx(0.682)
    assert fatores.corte(NOME_COMITE, 'Benigno') == pytest.approx(0.011)
    assert fatores.corte('KNN', 'Maligno') == pytest.approx(0.17)      # sem faixa: o limiar
    politica.adiar_incertos = False
    assert fatores.corte(NOME_COMITE, 'Maligno') == pytest.approx(0.146)
    assert fatores.corte('Árvore de Decisão', 'Maligno') is None        # a árvore decide pela folha


def test_arvore_fala_em_folha_e_nao_mostra_regua():
    r = _resultado(0.376, [0.5, 0.1, 0.01, 0.005, 0.004, 0.003, 0.001, 0.001])
    regra = "Benigno: certeza < 50%  ·  Maligno: certeza ≥ 50%"
    texto = "\n".join(linhas_fatores(r, 'Árvore de Decisão', regra=regra))
    assert "Saída (folha)" in texto and "na árvore" in texto
    assert "Régua" not in texto
    svm = linhas_fatores(r, 'SVM', regra=regra)
    assert "Régua do modelo:" in "\n".join(svm)
    assert "   Maligno: certeza ≥ 50%" in svm          # uma faixa por linha


def test_fator_sem_efeito_nao_e_listado_com_sentido():
    # Como na árvore: só três fatores atuam, os demais têm efeito exatamente zero.
    r = _resultado(0.376, [0.43, 0.16, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    linhas = linhas_fatores(r, 'Árvore de Decisão')
    assert sum(" pp  →" in l for l in linhas) == 3
    assert any("Demais 5 fatores" in l for l in linhas)


def test_bloco_cabe_na_caixa_de_detalhe():
    """As caixas de detalhe comportam ~47 colunas: linha maior quebra a cascata."""
    from core.fatores import LARGURA_BLOCO
    r = _resultado(0.384, [-0.158, -0.058, 0.021, 0.012, -0.012, -0.008, 0.0001, 0.0])
    regra = "Benigno: certeza < 7.0%  ·  Limítrofe: 7.0% a 27.0%  ·  Maligno: certeza > 27.0%"
    linhas = linhas_fatores(r, 'KNN', rotulo='Maligno', regra=regra, corte=0.17)
    assert max(len(l) for l in linhas) <= LARGURA_BLOCO


def test_bloco_sobrevive_a_fonte_do_pdf():
    """O PDF usa Courier básica (cp1252): nada pode virar quadradinho."""
    from utils.pdf_report import _sanitizar
    r = _resultado(0.375, [0.40, -0.05, 0.02, 0.01, 0.008, -0.005, 0.002, 0.001])
    texto = "\n".join(linhas_fatores(r, NOME_COMITE, rotulo='Maligno',
                                     regra="Benigno: certeza < 1.1%  ·  Maligno: certeza ≥ 68.2%"))
    _sanitizar(texto).encode('cp1252')   # levanta UnicodeEncodeError se sobrar glifo


# --- provedor do lote (com modelos de verdade, sem o .pkl) ------------------

@pytest.fixture(scope="module")
def lote_falso(breast_cancer_data):
    """Um 'loader' mínimo, com o treino reduzido para o teste ser rápido."""
    d = breast_cancer_data
    renomear = dict(zip(d['feature_names'], COLUNAS_WDBC))   # mesma ordem do WDBC
    X_tr = d['X_train'].rename(columns=renomear).iloc[:60]
    y_tr = d['y_train'].iloc[:60]
    X_te = d['X_test'].rename(columns=renomear).iloc[:5]

    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    lr = LogisticRegression(max_iter=2000).fit(Z_tr, y_tr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Z_tr, y_tr)
    dt = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr.values, y_tr)

    class Loader:
        calibrated_models = {'Regressão Logística': lr, 'KNN': knn}
        models = {'Regressão Logística': lr, 'KNN': knn, 'Árvore de Decisão': dt}
        X_train_scaled = Z_tr
        feature_names = COLUNAS_WDBC

    Loader.scaler = scaler
    indices = [int(i) for i in X_te.index]
    fatores = FatoresDoLote(Loader(), Z_te, X_te.values, indices)
    return fatores, {'lr': lr, 'knn': knn, 'dt': dt}, Z_te, X_te.values, indices


def test_provedor_explica_a_mesma_probabilidade_que_o_modelo(lote_falso):
    fatores, m, Z, X_raw, indices = lote_falso
    r = fatores.explicar('Regressão Logística', indices[0])
    assert r['prob'] == pytest.approx(m['lr'].predict_proba(Z[:1])[0, 1])


def test_provedor_usa_dados_brutos_na_arvore(lote_falso):
    fatores, m, Z, X_raw, indices = lote_falso
    r = fatores.explicar('Árvore de Decisão', indices[1])
    assert r['prob'] == pytest.approx(m['dt'].predict_proba(X_raw[1:2])[0, 1])


def test_comite_do_provedor_e_a_media_dos_membros(lote_falso):
    fatores, m, Z, X_raw, indices = lote_falso
    comite = fatores.explicar(NOME_COMITE, indices[2])
    membros = [fatores.explicar(n, indices[2]) for n in ('Regressão Logística', 'KNN')]
    assert comite['prob'] == pytest.approx(np.mean([r['prob'] for r in membros]))
    assert comite['prob'] == pytest.approx(np.mean(
        [m['lr'].predict_proba(Z[2:3])[0, 1], m['knn'].predict_proba(Z[2:3])[0, 1]]))


def test_provedor_guarda_em_cache_e_agenda_em_segundo_plano(lote_falso):
    fatores, _, _, _, indices = lote_falso
    assert fatores.em_cache('KNN', indices[3]) is None
    futuro = fatores.agendar('KNN', indices[3])
    resultado = futuro.result(timeout=30)
    assert fatores.em_cache('KNN', indices[3]) is resultado
    assert fatores.agendar('KNN', indices[3]).result() is resultado   # já pronto


def test_provedor_recusa_paciente_fora_do_lote(lote_falso):
    fatores = lote_falso[0]
    assert fatores.explicar('KNN', 999_999) is None


# --- o bloco dentro do detalhe das janelas ----------------------------------

def test_bloco_entra_logo_abaixo_do_diagnostico():
    from views.report_common import FatoresPacienteMixin

    class Janela(FatoresPacienteMixin):
        def _formatar_detalhe(self, e):
            return "PACIENTE 3\nDiagnóstico da IA: Maligno\n\nResto do relatório"

    texto = Janela()._texto_com_fatores({'indice': 3}, "BLOCO")
    assert texto == "PACIENTE 3\nDiagnóstico da IA: Maligno\n\nBLOCO\n\nResto do relatório"


# --- integração com o .pkl real ---------------------------------------------

@pytest.fixture(scope="module")
def lote_real(repo_data_dir):
    from core.batch_processor import BatchProcessor
    from core.predictor import PredictorEngine

    proc = BatchProcessor()
    bruto = pd.read_csv(os.path.join(repo_data_dir, 'dataTeste_sem_diagnostico.csv')).iloc[:3]
    Z, X = proc.process(bruto)
    fn = proc.loader.feature_names
    fatores = FatoresDoLote(proc.loader, Z[fn].values, X[fn].values, list(Z.index))
    certezas = PredictorEngine(proc.loader).probabilidades_calibradas(
        Z, X, ['Regressão Logística', NOME_COMITE])
    return fatores, certezas, list(Z.index)


def test_certeza_do_bloco_e_a_da_tabela_do_passo3(lote_real):
    """O que o médico lê no fim da cascata tem de ser o número da tabela."""
    fatores, certezas, indices = lote_real
    for modelo in ('Regressão Logística', NOME_COMITE):
        r = fatores.explicar(modelo, indices[0])
        assert r['prob'] == pytest.approx(float(certezas[modelo][0]), abs=1e-9)
        assert r['base'] + sum(i['contribuicao'] for i in r['fatores']) == pytest.approx(r['prob'])
        # Fundo = treino inteiro: a base é a prevalência de malignos do treino.
        assert 0.30 < r['base'] < 0.45
