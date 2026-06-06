"""
Módulo para processamento e padronização de lotes de dados clínicos.
"""

import pandas as pd
from core.inference import ModelLoader

class BatchProcessor:
    """
    Recebe um lote de dados, higieniza e aplica a escala Z-score.

    Attributes
    ----------
    loader : ModelLoader
        Instância do carregador que contém o scaler e as features do modelo.
    """

    def __init__(self):
        """
        Inicializa o processador de lotes instanciando o ModelLoader.
        """
        self.loader = ModelLoader()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa o pipeline de limpeza e padronização Z-score.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame contendo os dados brutos importados da interface.

        Returns
        -------
        pandas.DataFrame
            DataFrame higienizado e matematicamente padronizado.

        Raises
        ------
        ValueError
            Se o objeto scaler não estiver disponível na memória.
        """
        print("Iniciando processamento de lote...")

        # 1. Higienização: Remover colunas inúteis
        colunas_remover = ['id', 'Unnamed: 32', 'diagnosis']
        df_clean = df.drop(columns=[col for col in colunas_remover if col in df.columns], errors='ignore')

        # Filtra e ordena apenas as colunas que o modelo espera
        if self.loader.feature_names:
            df_clean = df_clean[[col for col in self.loader.feature_names if col in df_clean.columns]]

        # 2. Validação: Detetar se os dados estão brutos
        # Ex: A média bruta de 'area_mean' costuma ser > 100, enquanto no Z-score aproxima-se de 0.
        is_raw = False
        if 'area_mean' in df_clean.columns and df_clean['area_mean'].mean() > 10:
            is_raw = True

        # 3. Aplicação do Scaler (Z-score)
        if is_raw:
            print("Detetados dados brutos. A aplicar padronização Z-Score...")
            if self.loader.scaler is None:
                raise ValueError("Erro: O objeto Scaler não foi carregado corretamente.")
            
            dados_transformados = self.loader.scaler.transform(df_clean)
            df_final = pd.DataFrame(dados_transformados, columns=df_clean.columns)
        else:
            print("Os dados já se encontram padronizados.")
            df_final = df_clean

        return df_final