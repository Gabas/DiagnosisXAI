"""
Configuração compartilhada dos testes.

Coloca ``app/`` no ``sys.path`` (os módulos do projeto são importados como
``core.xxx``/``utils.xxx``, exatamente como o app faz) e disponibiliza um
recorte pequeno e determinístico de dados no mesmo formato da base Wisconsin
(30 biomarcadores, diagnóstico binário) para os testes dos explicadores —
via ``sklearn.datasets.load_breast_cancer``, sem depender do ``wisconsin.pkl``
real nem do notebook. Isso testa o *contrato* dos explicadores (a decisão
exibida bate com a do modelo), não os hiperparâmetros específicos escolhidos
no notebook.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(REPO_ROOT, 'app')
DATA_DIR = os.path.join(REPO_ROOT, 'data')

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="session")
def repo_data_dir():
    """Caminho absoluto da pasta ``data/`` do repositório."""
    return DATA_DIR


@pytest.fixture(scope="session")
def breast_cancer_data():
    """
    Recorte de treino/teste da base Wisconsin (via sklearn), padronizado.

    O sklearn usa a convenção 0=maligno/1=benigno — invertida em relação ao
    projeto (1=Maligno, ver notebook: ``{'B': 0, 'M': 1}``). Aqui o alvo já
    vem alinhado à convenção do projeto, para os explicadores (que assumem
    ``CLASSE_MALIGNO = 1``) funcionarem sem adaptação.

    Returns
    -------
    dict
        {'feature_names', 'X_train', 'X_test', 'X_train_scaled',
         'X_test_scaled', 'y_train', 'y_test'} — os dois primeiros em escala
        bruta (para explicadores como a Árvore, treinada sem Z-score), os
        demais padronizados.
    """
    dados = load_breast_cancer()
    colunas = [c.replace(' ', '_') for c in dados.feature_names]
    X = pd.DataFrame(dados.data, columns=colunas)
    y = pd.Series(1 - dados.target, name='diagnosis')  # alinha ao 1=Maligno do projeto

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=colunas, index=X_train.index)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=colunas, index=X_test.index)

    return {
        'feature_names': colunas,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
    }
