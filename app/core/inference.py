"""
Módulo responsável pelo carregamento de artefactos de Machine Learning.
"""

import pickle
import os

class ModelLoader:
    """
    Classe utilitária para carregar os artefactos do modelo treinados.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler ou None
        O transformador Z-score ajustado durante o treino.
    feature_names : list ou None
        Lista com os nomes das colunas (features) esperadas pelo modelo.
    """

    def __init__(self):
        """
        Inicializa o ModelLoader e extrai os artefactos salvos em disco.
        """
        self.scaler = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Abre o ficheiro .pkl e carrega o scaler e as features na memória.

        Raises
        ------
        FileNotFoundError
            Se o ficheiro wisconsin.pkl não for encontrado na pasta data.
        ValueError
            Se o formato do ficheiro .pkl for inválido ou desatualizado.
        """
        # Resolve o caminho dinamicamente para encontrar a pasta 'data'
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, 'data', 'wisconsin.pkl')

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ficheiro não encontrado: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
            if isinstance(data, dict):
                self.scaler = data.get('scaler')
                self.feature_names = data.get('feature_names')
            else:
                raise ValueError("O ficheiro .pkl está num formato antigo. Execute o notebook novamente.")