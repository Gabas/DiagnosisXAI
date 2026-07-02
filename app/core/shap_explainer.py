"""
Módulo que encapsula o cálculo de valores SHAP para os modelos do sistema.

Diferente dos explicadores exatos (árvore, regressão logística e KNN), que são
definidos no notebook e serializados no .pkl, o SHAP é uma biblioteca padrão —
por isso este wrapper vive no app e é construído em tempo de execução a partir
do modelo e de um background resumido (calculado e salvo pelo notebook).

Estratégia (validada por medição de tempo):
- Árvore de Decisão e Random Forest usam TreeExplainer (exato e instantâneo).
- Regressão Logística, SVM e KNN usam KernelExplainer sobre predict_proba
  (aproximado); é rápido por paciente (~0,2–0,4 s), mas lento para o lote todo —
  por isso a explicação individual é calculada sob demanda (lazy) no app.

Em todos os casos os valores estão em espaço de probabilidade:
``base + Σ(shap) = P(Maligno)``.
"""

import numpy as np


class ShapExplainer:
    """
    Calcula contribuições SHAP (por paciente) para um modelo específico.

    Attributes
    ----------
    model_key : str
        Identificador do modelo ('dt', 'rf', 'lr', 'svm' ou 'knn').
    feature_names : list[str]
        Nomes das características, na ordem usada no treino.
    """

    CLASSE_MALIGNO = 1
    # Tipo de explicador por modelo: 'tree' é exato/instantâneo; 'kernel' é lazy.
    TIPO = {'dt': 'tree', 'rf': 'tree', 'lr': 'kernel', 'svm': 'kernel', 'knn': 'kernel'}

    def __init__(self, model, model_key, feature_names, background):
        """
        Constrói o explicador SHAP apropriado para o modelo.

        Parameters
        ----------
        model : sklearn estimator
            Modelo já treinado.
        model_key : str
            'dt', 'rf', 'lr', 'svm' ou 'knn'.
        feature_names : list[str]
            Nomes das colunas, na ordem do treino.
        background : array-like
            Amostra de referência (centroides do treino) usada pelo KernelExplainer.
        """
        import shap  # import tardio: só quando um relatório SHAP é solicitado

        self.model = model
        self.model_key = model_key
        self.feature_names = list(feature_names)
        self._tipo = self.TIPO[model_key]
        self._mal_idx = list(model.classes_).index(self.CLASSE_MALIGNO)

        if self._tipo == 'tree':
            self._explainer = shap.TreeExplainer(model)
        else:
            self._explainer = shap.KernelExplainer(model.predict_proba, np.asarray(background))

    @property
    def is_lazy(self) -> bool:
        """True quando a explicação individual é custosa (KernelExplainer)."""
        return self._tipo == 'kernel'

    def _valores_maligno(self, shap_values):
        """
        Extrai o vetor SHAP e o valor base referentes à classe Maligno.

        Lida com os diferentes formatos retornados pelo SHAP conforme o
        explicador (lista por classe, array 3D ou 2D).
        """
        base = np.asarray(self._explainer.expected_value)
        if isinstance(shap_values, list):
            return float(base[self._mal_idx]), np.asarray(shap_values[self._mal_idx])[0]
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return float(base[self._mal_idx]), arr[0, :, self._mal_idx]
        base_val = float(base) if base.ndim == 0 else float(base[self._mal_idx])
        return base_val, arr[0]

    def explain_one(self, x_scaled_row, valores_reais=None) -> dict:
        """
        Calcula as contribuições SHAP de um único paciente.

        Parameters
        ----------
        x_scaled_row : array-like
            Linha de dados padronizados (Z-score) do paciente.
        valores_reais : array-like ou None
            Valores brutos das características, apenas para exibição.

        Returns
        -------
        dict
            {'base', 'prob', 'contribuicoes'} onde 'contribuicoes' é uma lista
            ordenada por impacto, cada item com {'feature', 'valor', 'shap',
            'direcao'}. Vale base + Σ(shap) = prob.
        """
        x = np.asarray(x_scaled_row, dtype=float).reshape(1, -1)
        if self._tipo == 'tree':
            shap_values = self._explainer.shap_values(x)
        else:
            shap_values = self._explainer.shap_values(x, silent=True)

        base, vetor = self._valores_maligno(shap_values)
        contribs = [
            {
                'feature': self.feature_names[j],
                'valor': (float(valores_reais[j]) if valores_reais is not None else None),
                'shap': float(s),
                'direcao': 'Maligno' if s > 0 else 'Benigno',
            }
            for j, s in enumerate(vetor)
        ]
        contribs.sort(key=lambda c: abs(c['shap']), reverse=True)
        return {
            'base': float(base),
            'prob': float(base + float(np.sum(vetor))),
            'contribuicoes': contribs,
        }
