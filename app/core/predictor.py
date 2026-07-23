"""
Módulo para geração de diagnósticos utilizando modelos de Inteligência Artificial.
"""

import pandas as pd
from core.inference import ModelLoader

class PredictorEngine:
    """
    Motor de inferência que aplica algoritmos de Machine Learning 
    aos dados dos pacientes para gerar diagnósticos.

    Attributes
    ----------
    loader : ModelLoader
        Instância do carregador contendo os modelos salvos.
    """

    def __init__(self, loader: ModelLoader):
        """Recebe o ModelLoader para aceder aos algoritmos disponíveis."""
        self.loader = loader

    # Modelos que foram treinados em dados brutos (sem Z-score) e devem receber df_limpo
    MODELOS_SEM_ESCALA = {'Árvore de Decisão'}

    # Modelos cuja "certeza" não é informativa: a Árvore de Decisão tem folhas
    # puras, portanto a probabilidade é sempre 100% ou 0% — omitimos a coluna.
    MODELOS_SEM_CERTEZA = {'Árvore de Decisão'}

    def predict(self, df_padronizado: pd.DataFrame, df_limpo: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Gera as previsões para um lote utilizando um ou todos os modelos.

        Parameters
        ----------
        df_padronizado : pandas.DataFrame
            DataFrame higienizado e em escala Z-score (para RF, SVM, KNN, LR).
        df_limpo : pandas.DataFrame
            DataFrame higienizado sem escalonamento (para modelos invariantes à escala,
            como a Árvore de Decisão, que foi treinada em dados brutos).
        model_name : str
            O nome do modelo ('Random Forest', 'SVM', etc) ou 'Todos (Comparação)'.

        Returns
        -------
        pandas.DataFrame
            O DataFrame original acrescido das colunas de diagnóstico.
        """
        df_resultado = df_padronizado.copy()

        # --- LÓGICA 1: TODOS OS MODELOS (COMPARAÇÃO) ---
        if model_name == "Todos (Comparação)":
            print("A executar comparação entre todas as IAs...")

            for nome, modelo in self.loader.models.items():
                entrada = df_limpo if nome in self.MODELOS_SEM_ESCALA else df_padronizado
                previsoes = modelo.predict(entrada)
                sigla = nome[:3].upper()
                df_resultado[f'IA_{sigla}'] = ['Maligno' if p == 1 else 'Benigno' for p in previsoes]

        # --- LÓGICA 2: UM ÚNICO MODELO ---
        else:
            print(f"A executar o modelo: {model_name}...")
            modelo = self.loader.models.get(model_name)

            if modelo is None:
                raise ValueError(f"Modelo '{model_name}' não carregado corretamente.")

            entrada = df_limpo if model_name in self.MODELOS_SEM_ESCALA else df_padronizado
            previsoes = modelo.predict(entrada)
            df_resultado['Diagnóstico_IA'] = ['Maligno' if p == 1 else 'Benigno' for p in previsoes]

            if model_name not in self.MODELOS_SEM_CERTEZA:
                # A "certeza" vem, de preferência, do modelo CALIBRADO (probabilidades
                # confiáveis — ver Seção 10.3/14 do notebook); recai no modelo original
                # quando o .pkl não traz a versão calibrada. O rótulo acima permanece o
                # do modelo original, preservando o desempenho validado.
                calibrados = getattr(self.loader, 'calibrated_models', {})
                modelo_proba = calibrados.get(model_name, modelo)
                if hasattr(modelo_proba, 'predict_proba'):
                    probabilidades = modelo_proba.predict_proba(entrada)
                    df_resultado['Certeza_Maligno(%)'] = [round(prob[1] * 100, 2) for prob in probabilidades]

        print("Processamento concluído!")
        return df_resultado