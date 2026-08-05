"""
Módulo para geração de diagnósticos utilizando modelos de Inteligência Artificial.

A classe prevista, a certeza exibida e a marcação de caso limítrofe saem todas
da mesma probabilidade calibrada, aplicada à política de decisão de
``core.decision`` — ver a discussão de coerência naquele módulo.
"""

import numpy as np
import pandas as pd

from core.decision import NOME_COMITE, PoliticaDecisao
from core.inference import ModelLoader

class PredictorEngine:
    """
    Motor de inferência que aplica algoritmos de Machine Learning
    aos dados dos pacientes para gerar diagnósticos.

    Attributes
    ----------
    loader : ModelLoader
        Instância do carregador contendo os modelos salvos.
    politica : core.decision.PoliticaDecisao
        Limiar de operação e faixa de revisão em vigor.
    """

    def __init__(self, loader: ModelLoader, politica: PoliticaDecisao = None):
        """
        Recebe o ModelLoader para aceder aos algoritmos disponíveis.

        Parameters
        ----------
        loader : ModelLoader
            Carregador com os modelos, calibrados e artefatos do .pkl.
        politica : PoliticaDecisao ou None, optional
            Política de decisão. Quando None, usa a do ``ModelLoader`` (lida de
            ``data/limiares.json``), recaindo na política neutra de 0,5.
        """
        self.loader = loader
        self.politica = politica or getattr(loader, 'politica', None) or PoliticaDecisao()

    # Modelos que foram treinados em dados brutos (sem Z-score) e devem receber df_limpo
    MODELOS_SEM_ESCALA = {'Árvore de Decisão'}

    # Modelos cuja "certeza" não é informativa: a Árvore de Decisão tem folhas
    # puras, portanto a probabilidade é sempre 100% ou 0% — omitimos a coluna.
    # Sem probabilidade utilizável, esses modelos também não participam do
    # comitê nem recebem limiar de operação: continuam decidindo por predict().
    MODELOS_SEM_CERTEZA = {'Árvore de Decisão'}

    # Nome do comitê no seletor (reexportado para as views).
    NOME_COMITE = NOME_COMITE

    # Ordem fixa dos membros do comitê — mantém a média (e as colunas por
    # membro) determinísticas, independentemente da ordem do dicionário.
    MEMBROS_COMITE = ('Regressão Logística', 'Random Forest', 'SVM', 'KNN')

    def modelos_disponiveis(self) -> list:
        """
        Nomes dos modelos oferecidos ao usuário, na ordem do seletor.

        Returns
        -------
        list[str]
            Os modelos carregados e, quando há membros calibrados suficientes,
            o comitê de voto suave.
        """
        nomes = list(self.loader.models.keys())
        if len(self.membros_comite()) >= 2:
            nomes.append(NOME_COMITE)
        return nomes

    def membros_comite(self) -> list:
        """Membros do comitê efetivamente disponíveis, na ordem canônica."""
        return [nome for nome in self.MEMBROS_COMITE
                if self.loader.calibrated_models.get(nome) is not None]

    @staticmethod
    def sigla(nome_modelo: str) -> str:
        """Sigla de 3 letras usada nas colunas por modelo (ex.: 'Random Forest' -> 'RAN')."""
        return nome_modelo[:3].upper()

    def classificar_decisao(self, certeza_maligno: float, modelo: str) -> str:
        """
        Classifica a decisão como 'Limítrofe' ou 'Definida'.

        Parameters
        ----------
        certeza_maligno : float
            Probabilidade calibrada de malignidade, em porcentagem (0–100).
        modelo : str
            Nome do modelo, para resolver o limiar de operação.

        Returns
        -------
        str
            'Limítrofe' quando a certeza está a até ``politica.banda`` do limiar
            de operação — perto o bastante para que a decisão vire com uma
            variação pequena; 'Definida' caso contrário.
        """
        return self.politica.zona(certeza_maligno / 100.0, modelo)

    def probabilidades_calibradas(self, df_padronizado: pd.DataFrame, df_limpo: pd.DataFrame,
                                  modelos) -> dict:
        """
        Calcula P(Maligno) calibrada para cada modelo pedido.

        É o que alinha os relatórios de explicabilidade à decisão exibida: as
        janelas do Passo 5 recebem estas probabilidades e reescrevem nelas a
        classe e a marcação de limítrofe (ver ``core.decision``).

        Parameters
        ----------
        df_padronizado, df_limpo : pandas.DataFrame
            Lote em escala Z-score e sem escalonamento.
        modelos : iterable of str
            Nomes dos modelos desejados.

        Returns
        -------
        dict[str, numpy.ndarray]
            {modelo: vetor de P(Maligno) em 0–1}. Modelos sem probabilidade
            utilizável (ex.: Árvore de Decisão) ficam de fora.
        """
        probabilidades = {}
        for nome in modelos:
            entrada = df_limpo if nome in self.MODELOS_SEM_ESCALA else df_padronizado
            if nome == NOME_COMITE:
                prob = self._probabilidade_comite(df_padronizado)
            else:
                prob = self._probabilidade_maligno(entrada, nome)
            if prob is not None:
                probabilidades[nome] = prob
        return probabilidades

    def _probabilidade_maligno(self, entrada, nome_modelo: str):
        """
        P(Maligno) calibrada de um modelo, ou None quando ele não a fornece.

        Prefere sempre o modelo CALIBRADO (escalonamento de Platt, Seção 14 do
        notebook), pois é essa a probabilidade que o usuário vê e sobre a qual o
        limiar de operação foi escolhido. Recai no modelo original quando o .pkl
        não traz a versão calibrada.
        """
        if nome_modelo in self.MODELOS_SEM_CERTEZA:
            return None

        modelo = (self.loader.calibrated_models.get(nome_modelo)
                  or self.loader.models.get(nome_modelo))
        if modelo is None or not hasattr(modelo, 'predict_proba'):
            return None

        classes = list(getattr(modelo, 'classes_', [0, 1]))
        idx_maligno = classes.index(1) if 1 in classes else -1
        return modelo.predict_proba(entrada)[:, idx_maligno]

    def _probabilidade_comite(self, df_padronizado: pd.DataFrame):
        """
        Média das probabilidades calibradas dos membros (voto suave).

        A média das probabilidades — e não a contagem de votos — é o que faz o
        comitê ganhar sensibilidade sem perder especificidade: um membro muito
        confiante em malignidade puxa a média mesmo quando os demais discordam
        por pouco, e é justamente nesses casos que mora o falso negativo.
        """
        membros = self.membros_comite()
        if len(membros) < 2:
            raise ValueError(
                "O comitê exige ao menos dois modelos calibrados no wisconsin.pkl.")
        return np.mean([self._probabilidade_maligno(df_padronizado, nome)
                        for nome in membros], axis=0)

    def predict(self, df_padronizado: pd.DataFrame, df_limpo: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Gera as previsões para um lote utilizando um, todos ou o comitê dos modelos.

        Parameters
        ----------
        df_padronizado : pandas.DataFrame
            DataFrame higienizado e em escala Z-score (para RF, SVM, KNN, LR).
        df_limpo : pandas.DataFrame
            DataFrame higienizado sem escalonamento (para modelos invariantes à escala,
            como a Árvore de Decisão, que foi treinada em dados brutos).
        model_name : str
            O nome do modelo ('Random Forest', 'SVM', etc), 'Todos (Comparação)'
            ou 'Comitê (voto suave)'.

        Returns
        -------
        pandas.DataFrame
            O DataFrame original acrescido das colunas de diagnóstico.
        """
        df_resultado = df_padronizado.copy()

        # --- LÓGICA 1: TODOS OS MODELOS (COMPARAÇÃO) ---
        if model_name == "Todos (Comparação)":
            print("A executar comparação entre todas as IAs...")

            for nome in self.loader.models:
                df_resultado[f'IA_{self.sigla(nome)}'] = self._rotular(
                    df_padronizado, df_limpo, nome)

        # --- LÓGICA 2: COMITÊ (VOTO SUAVE) ---
        elif model_name == NOME_COMITE:
            membros = self.membros_comite()
            print(f"A executar o comitê ({', '.join(membros)})...")

            certezas = self._probabilidade_comite(df_padronizado) * 100
            df_resultado['Diagnóstico_IA'] = [
                self.politica.rotular(c / 100.0, model_name) for c in certezas]
            df_resultado['Certeza_Maligno(%)'] = certezas.round(2)
            df_resultado['Decisão'] = [
                self.classificar_decisao(c, model_name) for c in certezas]

            # A certeza de cada membro fica visível: é a explicação do comitê —
            # mostra se a média veio de consenso ou de um único membro alarmado.
            for nome in membros:
                p = self._probabilidade_maligno(df_padronizado, nome)
                df_resultado[f'Certeza_{self.sigla(nome)}(%)'] = (p * 100).round(2)

        # --- LÓGICA 3: UM ÚNICO MODELO ---
        else:
            print(f"A executar o modelo: {model_name}...")
            if self.loader.models.get(model_name) is None:
                raise ValueError(f"Modelo '{model_name}' não carregado corretamente.")

            entrada = df_limpo if model_name in self.MODELOS_SEM_ESCALA else df_padronizado
            prob = self._probabilidade_maligno(entrada, model_name)

            if prob is None:
                # Árvore de Decisão: sem probabilidade utilizável, decide por predict().
                previsoes = self.loader.models[model_name].predict(entrada)
                df_resultado['Diagnóstico_IA'] = [
                    'Maligno' if p == 1 else 'Benigno' for p in previsoes]
            else:
                certezas = prob * 100
                df_resultado['Diagnóstico_IA'] = [
                    self.politica.rotular(p, model_name) for p in prob]
                df_resultado['Certeza_Maligno(%)'] = certezas.round(2)
                # Marca decisões limítrofes (perto do limiar de operação) para revisão.
                df_resultado['Decisão'] = [
                    self.classificar_decisao(c, model_name) for c in certezas]

        # Aviso de perfil atípico (fora da distribuição de treino) — independe do
        # modelo escolhido e sempre usa os atributos padronizados. Sinaliza casos em
        # que a previsão extrapola para uma região pouco vista no treino.
        detector = getattr(self.loader, 'ood_detector', None)
        if detector is not None:
            features = self.loader.feature_names
            entrada_ood = df_padronizado[features] if features else df_padronizado
            df_resultado['Perfil'] = detector.rotulos(entrada_ood)

        print("Processamento concluído!")
        return df_resultado

    def _rotular(self, df_padronizado: pd.DataFrame, df_limpo: pd.DataFrame,
                 nome_modelo: str) -> list:
        """
        Classe prevista por um modelo, segundo o limiar de operação em vigor.

        Quando o modelo fornece probabilidade calibrada, o rótulo sai dela — a
        mesma que alimenta a coluna de certeza, de modo que as duas nunca se
        contradigam. Sem probabilidade utilizável (Árvore de Decisão), recai em
        ``predict()``, que decide implicitamente por 0,5.
        """
        entrada = df_limpo if nome_modelo in self.MODELOS_SEM_ESCALA else df_padronizado
        prob = self._probabilidade_maligno(entrada, nome_modelo)
        if prob is not None:
            return [self.politica.rotular(p, nome_modelo) for p in prob]

        previsoes = self.loader.models[nome_modelo].predict(entrada)
        return ['Maligno' if p == 1 else 'Benigno' for p in previsoes]
