"""
Detector de amostras fora da distribuição de treino (out-of-distribution, OOD).

Sinaliza pacientes cujo perfil morfológico é atípico em relação aos dados de
treino — casos em que a previsão dos modelos é menos confiável, pois extrapola
para uma região do espaço de atributos pouco (ou não) observada no aprendizado.

Método: distância de Mahalanobis no espaço completo dos 30 atributos JÁ
padronizados (Z-score). A covariância é estimada com encolhimento de Ledoit-Wolf,
necessário porque vários atributos da base Wisconsin são fortemente correlacionados
(raio, perímetro e área são quase colineares) — o que tornaria a covariância
empírica mal-condicionada e a distância instável.

O limiar de atipicidade é empírico: o percentil ``percentil`` das distâncias dos
próprios pacientes de treino. Assim não se assume normalidade multivariada para o
corte — apenas que um paciente típico deve ficar tão perto do centro quanto a
maioria do treino. Um paciente além desse limiar é marcado como "Atípico".

Observação: NÃO se usa a projeção UMAP/PCA 2D para isso — a redução a 2D descarta
justamente a informação necessária para julgar atipicidade, e pontos novos nem
poderiam ser projetados sem o redutor ajustado. A visualização 2D permanece uma
ferramenta de inspeção, não de detecção.
"""

import numpy as np
from sklearn.covariance import LedoitWolf


class OODDetector:
    """
    Marca amostras atípicas por distância de Mahalanobis ao conjunto de treino.

    Parameters
    ----------
    X_train : array-like (n_amostras, n_atributos)
        Conjunto de treino JÁ padronizado (Z-score) — o mesmo espaço em que os
        dados do app são transformados antes da inferência.
    percentil : float, optional
        Percentil das distâncias de treino usado como limiar de atipicidade
        (padrão 99.0 — conservador, para não alarmar perfis apenas um pouco
        incomuns).

    Attributes
    ----------
    limiar : float
        Distância² de Mahalanobis a partir da qual uma amostra é "Atípica".
    percentil : float
        Percentil que definiu o limiar.
    """

    TIPICO = "Típico"
    ATIPICO = "Atípico"

    def __init__(self, X_train, percentil: float = 99.0):
        """Ajusta a covariância regularizada e calibra o limiar pelo treino."""
        X = np.asarray(X_train, dtype=float)
        self._cov = LedoitWolf().fit(X)
        d2_treino = self._cov.mahalanobis(X)
        self.percentil = float(percentil)
        self.limiar = float(np.percentile(d2_treino, percentil))

    def score(self, X):
        """Distância² de Mahalanobis de cada amostra ao centro do treino."""
        return self._cov.mahalanobis(np.asarray(X, dtype=float))

    def flag(self, X):
        """Vetor booleano — ``True`` onde a amostra é atípica (além do limiar)."""
        return self.score(X) > self.limiar

    def rotulos(self, X):
        """Lista de rótulos ``'Típico'``/``'Atípico'``, um por amostra."""
        return [self.ATIPICO if atipico else self.TIPICO for atipico in self.flag(X)]
