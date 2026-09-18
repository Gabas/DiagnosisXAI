"""
Módulo para processamento e padronização de lotes de dados clínicos.
"""

import pandas as pd
from core.inference import ModelLoader
from core.validacao import validar_colunas, validar_nao_negativos, validar_valores

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

    def process(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executa o pipeline de limpeza e padronização.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame contendo os dados brutos importados da interface.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (df_scaled, df_limpo) — dados com Z-score e dados limpos sem escalonamento.
            Modelos invariantes à escala (ex: Árvore de Decisão) devem usar df_limpo.
        """
        colunas_remover = ['id', 'Unnamed: 32', 'diagnosis']
        df_clean = df.drop(columns=[col for col in colunas_remover if col in df.columns], errors='ignore')

        if self.loader.feature_names:
            validar_colunas(df_clean, self.loader.feature_names)
            df_clean = df_clean[list(self.loader.feature_names)]

        # Vem antes da heurística de escala: uma coluna com texto no meio nem
        # tem média para comparar com o limiar de area_mean.
        validar_valores(df_clean)

        is_raw = False
        if 'area_mean' in df_clean.columns and df_clean['area_mean'].mean() > 10:
            is_raw = True

        if is_raw:
            # Só agora dá para recusar negativos: em Z-score eles são normais.
            validar_nao_negativos(df_clean)

        if is_raw:
            if self.loader.scaler is None:
                raise ValueError("O objeto Scaler não foi carregado corretamente.")

            dados_transformados = self.loader.scaler.transform(df_clean)
            df_scaled = pd.DataFrame(dados_transformados, columns=df_clean.columns)
        else:
            df_scaled = df_clean

        return df_scaled, df_clean