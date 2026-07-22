"""
Testes dos explicadores (core/explainers.py).

O objetivo central destes testes é validar um invariante simples mas fácil
de violar silenciosamente: o que o explicador EXIBE (classe, confiança,
votos) precisa bater com o que o modelo de fato DECIDE (predict/predict_proba/
decision_function). Dois bugs reais já apareceram por essa via — o SVM (o
balanço de forças não reconstituía a decisão) e o KNN (a contagem bruta de
vizinhos contradizia a classe final quando ``weights='distance'``) — por
isso os testes abaixo replicam exatamente as equações usadas para corrigi-los.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.explainers import (
    DecisionTreeExplainer,
    KNNExplainer,
    LogisticRegressionExplainer,
    RandomForestExplainer,
    SVMExplainer,
)


class TestDecisionTreeExplainer:
    def test_classe_bate_com_predict(self, breast_cancer_data):
        d = breast_cancer_data
        model = DecisionTreeClassifier(criterion='entropy', random_state=42)
        model.fit(d['X_train'], d['y_train'])
        exp = DecisionTreeExplainer(model, d['feature_names'])

        pred = model.predict(d['X_test'])
        explicacoes = exp.explain(d['X_test'])

        for e, p in zip(explicacoes, pred):
            assert e['classe'] == ('Maligno' if p == 1 else 'Benigno')

    def test_certeza_bate_com_a_folha(self, breast_cancer_data):
        d = breast_cancer_data
        model = DecisionTreeClassifier(criterion='entropy', random_state=42)
        model.fit(d['X_train'], d['y_train'])
        exp = DecisionTreeExplainer(model, d['feature_names'])

        mal_idx = list(model.classes_).index(1)
        proba = model.predict_proba(d['X_test'])
        explicacoes = exp.explain(d['X_test'])

        for e, p in zip(explicacoes, proba):
            p_mal = p[mal_idx]
            esperada = p_mal if e['classe'] == 'Maligno' else (1 - p_mal)
            assert e['certeza'] == pytest.approx(round(esperada * 100, 1), abs=0.15)

    def test_indices_preservados(self, breast_cancer_data):
        d = breast_cancer_data
        model = DecisionTreeClassifier(random_state=42).fit(d['X_train'], d['y_train'])
        exp = DecisionTreeExplainer(model, d['feature_names'])
        explicacoes = exp.explain(d['X_test'])
        assert [e['indice'] for e in explicacoes] == list(d['X_test'].index)


class TestLogisticRegressionExplainer:
    def _build(self, d):
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(d['X_train_scaled'], d['y_train'])
        exp = LogisticRegressionExplainer(model, d['feature_names'])
        return model, exp

    def test_classe_e_probabilidade_batem_com_o_modelo(self, breast_cancer_data):
        d = breast_cancer_data
        model, exp = self._build(d)

        pred = model.predict(d['X_test_scaled'])
        mal_idx = list(model.classes_).index(1)
        proba = model.predict_proba(d['X_test_scaled'])[:, mal_idx]
        explicacoes = exp.explain(d['X_test_scaled'], d['X_test'])

        for e, p, pr in zip(explicacoes, pred, proba):
            assert e['classe'] == ('Maligno' if p == 1 else 'Benigno')
            assert e['probabilidade'] == pytest.approx(round(pr * 100, 1), abs=0.15)

    def test_distancia_reconstitui_a_decision_function(self, breast_cancer_data):
        """distancia = decision_function / ||w|| — a projeção exata usada
        para posicionar o paciente em relação à fronteira real do modelo."""
        d = breast_cancer_data
        model, exp = self._build(d)

        decisao = model.decision_function(d['X_test_scaled'])
        norm_w = np.linalg.norm(model.coef_[0])
        explicacoes = exp.explain(d['X_test_scaled'], d['X_test'])

        for e, dist in zip(explicacoes, decisao / norm_w):
            assert e['distancia'] == pytest.approx(dist, abs=1e-8)


class TestKNNExplainer:
    def _build(self, d, n_neighbors, weights, metric='minkowski'):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        model.fit(d['X_train_scaled'], d['y_train'])
        pca = PCA(n_components=2, random_state=42).fit(d['X_train_scaled'])
        train_2d = pca.transform(d['X_train_scaled'])
        exp = KNNExplainer(model, d['feature_names'], d['y_train'].values, pca, train_2d, [])
        return model, exp

    def test_classe_bate_com_predict_uniform(self, breast_cancer_data):
        d = breast_cancer_data
        model, exp = self._build(d, n_neighbors=5, weights='uniform')
        pred = model.predict(d['X_test_scaled'])
        explicacoes = exp.explain(d['X_test_scaled'])
        for e, p in zip(explicacoes, pred):
            assert e['classe'] == ('Maligno' if p == 1 else 'Benigno')

    def test_confianca_vem_de_predict_proba_com_peso_por_distancia(self, breast_cancer_data):
        """Regressão do bug corrigido: com weights='distance' a confiança
        exibida precisa vir de predict_proba (a decisão real ponderada),
        não de uma fração ingênua da contagem bruta de vizinhos."""
        d = breast_cancer_data
        model, exp = self._build(d, n_neighbors=4, weights='distance', metric='manhattan')

        pred = model.predict(d['X_test_scaled'])
        mal_idx = list(model.classes_).index(1)
        proba = model.predict_proba(d['X_test_scaled'])[:, mal_idx]
        explicacoes = exp.explain(d['X_test_scaled'])

        for e, p, pr in zip(explicacoes, pred, proba):
            esperada = pr if p == 1 else (1 - pr)
            assert e['confianca'] == pytest.approx(round(esperada * 100, 1), abs=0.15)

    def test_peso_ponderado_do_lado_vencedor_e_sempre_maioria(self, breast_cancer_data):
        """Com K par e weights='distance', a contagem bruta de vizinhos pode
        empatar (ou a minoria 'vencer') — mas o peso ponderado do lado que a
        classe final representa tem que ser sempre >= 50%, pois é ele quem
        de fato decide."""
        d = breast_cancer_data
        model, exp = self._build(d, n_neighbors=4, weights='distance', metric='manhattan')
        explicacoes = exp.explain(d['X_test_scaled'])

        algum_empate_bruto = False
        for e in explicacoes:
            peso_vencedor = e['peso_maligno'] if e['classe'] == 'Maligno' else e['peso_benigno']
            assert peso_vencedor >= 50.0 - 1e-6
            if e['votos_maligno'] == e['votos_benigno']:
                algum_empate_bruto = True
        # Confirma que o cenário do bug (empate na contagem bruta) realmente
        # ocorre neste recorte de dados — senão o teste acima não cobriria nada.
        assert algum_empate_bruto

    def test_votos_somam_k(self, breast_cancer_data):
        d = breast_cancer_data
        model, exp = self._build(d, n_neighbors=6, weights='uniform')
        explicacoes = exp.explain(d['X_test_scaled'])
        for e in explicacoes:
            assert e['votos_maligno'] + e['votos_benigno'] == 6


class TestRandomForestExplainer:
    def test_classe_e_probabilidade_batem_com_o_voto_suave(self, breast_cancer_data):
        d = breast_cancer_data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(d['X_train_scaled'], d['y_train'])
        exp = RandomForestExplainer(model, d['feature_names'])

        pred = model.predict(d['X_test_scaled'])
        mal_idx = list(model.classes_).index(1)
        proba = model.predict_proba(d['X_test_scaled'])[:, mal_idx]
        explicacoes = exp.explain(d['X_test_scaled'])

        for e, p, pr in zip(explicacoes, pred, proba):
            assert e['classe'] == ('Maligno' if p == 1 else 'Benigno')
            assert e['probabilidade'] == pytest.approx(round(pr * 100, 1), abs=0.15)

    def test_votos_somam_n_arvores(self, breast_cancer_data):
        d = breast_cancer_data
        model = RandomForestClassifier(n_estimators=37, random_state=42)
        model.fit(d['X_train_scaled'], d['y_train'])
        exp = RandomForestExplainer(model, d['feature_names'])
        explicacoes = exp.explain(d['X_test_scaled'])
        for e in explicacoes:
            assert e['votos_maligno'] + e['votos_benigno'] == 37


class TestSVMExplainer:
    def _build(self, d):
        model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        model.fit(d['X_train_scaled'], d['y_train'])
        exp = SVMExplainer(
            model, d['feature_names'], d['y_train'].values,
            d['X_train_scaled'].values, [])
        return model, exp

    def test_classe_bate_com_predict(self, breast_cancer_data):
        d = breast_cancer_data
        model, exp = self._build(d)
        pred = model.predict(d['X_test_scaled'])
        explicacoes = exp.explain(d['X_test_scaled'])
        for e, p in zip(explicacoes, pred):
            assert e['classe'] == ('Maligno' if p == 1 else 'Benigno')

    def test_balanco_de_forcas_reconstitui_a_decision_function(self, breast_cancer_data):
        """Regressão do bug corrigido: forca_maligno - forca_benigno + vies
        precisa reconstituir exatamente a decision_function do SVM — é essa
        soma completa (não um top-N por magnitude) que decide de fato."""
        d = breast_cancer_data
        model, exp = self._build(d)

        decisao_real = model.decision_function(d['X_test_scaled'])
        explicacoes = exp.explain(d['X_test_scaled'])

        for e, dist in zip(explicacoes, decisao_real):
            reconstituida = e['forca_maligno'] - e['forca_benigno'] + e['vies']
            assert reconstituida == pytest.approx(dist, abs=1e-6)
            assert e['distancia'] == pytest.approx(dist, abs=1e-8)
