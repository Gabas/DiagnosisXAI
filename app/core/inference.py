"""
Módulo responsável pelo carregamento de artefactos de Machine Learning.
"""

import cloudpickle
import os

class ModelLoader:
    """
    Classe utilitária para carregar os artefactos e modelos treinados.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler ou None
        O transformador Z-score ajustado durante o treino.
    feature_names : list ou None
        Lista com os nomes das colunas (features) esperadas pelo modelo.
    models : dict
        Dicionário que armazena os modelos de Machine Learning.
    explainers : dict
        Explicadores de explicabilidade (XAI) gerados e serializados pelo
        notebook: {'arvore': ..., 'logistica': ...}. Valores None quando o
        .pkl foi gerado por uma versão do notebook sem essa etapa.
    """

    def __init__(self):
        """Inicializa o ModelLoader e extrai os artefactos salvos em disco."""
        self.scaler = None
        self.feature_names = None
        self.models = {}
        self.explainers = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Abre o arquivo .pkl e carrega o scaler, features e modelos na memória."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, 'data', 'wisconsin.pkl')

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        with open(filepath, 'rb') as f:
            data = cloudpickle.load(f)

            if isinstance(data, dict):
                self.scaler = data.get('scaler')
                self.feature_names = data.get('feature_names')

                # Mapeamento dinâmico dos seus modelos do Jupyter Notebook
                self.models['Árvore de Decisão'] = data.get('model_dt')
                self.models['Random Forest'] = data.get('model_rf')
                self.models['SVM'] = data.get('model_svm')
                self.models['KNN'] = data.get('model_knn')
                self.models['Regressão Logística'] = data.get('model_lr')

                # Explicadores (XAI) definidos e serializados pelo notebook.
                # O app apenas os utiliza — a lógica de explicabilidade vive no notebook.
                self.explainers['arvore'] = data.get('explainer_dt')
                self.explainers['logistica'] = data.get('explainer_lr')
                self.explainers['knn'] = data.get('explainer_knn')
            else:
                raise ValueError("O arquivo .pkl está num formato antigo. Execute o notebook novamente.")