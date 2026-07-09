"""
Módulo responsável pelo carregamento de artefactos de Machine Learning.
"""

import cloudpickle
import os
import numpy as np

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
        self.shap_background = None
        self.shap_importances = {}
        self.X_train_scaled = None
        self.y_train = None
        self.umap_train_2d = None
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

                # Explicadores (XAI) definidos em core.explainers e serializados
                # por referência no .pkl. O app apenas os utiliza.
                self.explainers['arvore'] = data.get('explainer_dt')
                self.explainers['logistica'] = data.get('explainer_lr')
                self.explainers['knn'] = data.get('explainer_knn')
                self.explainers['randomforest'] = data.get('explainer_rf')
                self.explainers['svm'] = data.get('explainer_svm')

                # Artefatos do SHAP (o wrapper é construído sob demanda no app).
                self.shap_background = data.get('shap_background')
                self.shap_importances = data.get('shap_importances') or {}

                # Dados de treino e embedding UMAP para o mapa populacional.
                self.X_train_scaled = data.get('X_train_scaled')
                self.y_train = data.get('y_train')
                self.umap_train_2d = data.get('umap_train_2d')

                # Fallback: embedding UMAP em arquivo próprio (robusto quando o
                # .pkl é regenerado sem ele). Alinhado com X_train_scaled.
                if self.umap_train_2d is None:
                    umap_path = os.path.join(base_dir, 'data', 'umap_train_2d.npy')
                    if os.path.exists(umap_path):
                        self.umap_train_2d = np.load(umap_path)

                # Explicador do Random Forest: usa apenas o modelo + feature_names
                # (nada pré-calculado), então é construído no app quando o .pkl
                # ainda não o traz — mantendo o app funcional sem regenerar o .pkl.
                if self.explainers.get('randomforest') is None \
                        and self.models.get('Random Forest') is not None \
                        and self.feature_names is not None:
                    from core.explainers import RandomForestExplainer
                    self.explainers['randomforest'] = RandomForestExplainer(
                        self.models['Random Forest'], self.feature_names)
            else:
                raise ValueError("O arquivo .pkl está num formato antigo. Execute o notebook novamente.")