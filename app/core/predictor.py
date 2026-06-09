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

    def predict(self, df_padronizado: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Gera as previsões para um lote utilizando um ou todos os modelos.

        Parameters
        ----------
        df_padronizado : pandas.DataFrame
            DataFrame contendo os dados já higienizados e em escala Z-score.
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
            
            # Pede a opinião a cada modelo disponível e cria uma coluna para cada
            for nome, modelo in self.loader.models.items():
                previsoes = modelo.predict(df_padronizado)
                # Cria uma sigla com as 3 primeiras letras (ex: IA_Ran, IA_SVM)
                sigla = nome[:3].upper()
                df_resultado[f'IA_{sigla}'] = ['Maligno' if p == 1 else 'Benigno' for p in previsoes]

        # --- LÓGICA 2: UM ÚNICO MODELO ---
        else:
            print(f"A executar o modelo: {model_name}...")
            modelo = self.loader.models.get(model_name)
            
            if modelo is None:
                raise ValueError(f"Modelo '{model_name}' não carregado corretamente.")

            previsoes = modelo.predict(df_padronizado)
            df_resultado['Diagnóstico_IA'] = ['Maligno' if p == 1 else 'Benigno' for p in previsoes]

            if hasattr(modelo, 'predict_proba'):
                probabilidades = modelo.predict_proba(df_padronizado)
                df_resultado['Certeza_Maligno(%)'] = [round(prob[1] * 100, 2) for prob in probabilidades]

        print("Processamento concluído!")
        return df_resultado