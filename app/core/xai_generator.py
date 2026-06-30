"""
Módulo responsável pela geração de explicações (XAI) dos diagnósticos.

Implementa a interpretação da Árvore de Decisão: para cada paciente,
percorre o caminho real de regras da raiz até a folha e decompõe a
probabilidade final de malignidade na soma das contribuições marginais
de cada característica morfológica. Diferente de métodos aproximados,
esta explicação é exata — reflete a própria lógica de decisão da árvore.
"""


class DecisionTreeExplainer:
    """
    Gera explicações interpretáveis para um classificador Árvore de Decisão.

    Para cada amostra, segue o caminho percorrido na árvore (do nó raiz até a
    folha) e atribui a cada divisão (split) a variação que ela provocou na
    probabilidade de malignidade. A soma das contribuições mais a proporção
    inicial da raiz reconstitui exatamente a probabilidade da folha.

    Attributes
    ----------
    model : sklearn.tree.DecisionTreeClassifier
        Modelo de árvore já treinado.
    feature_names : list[str]
        Nomes das características morfológicas, na ordem usada no treino.
    """

    CLASSE_MALIGNO = 1

    def __init__(self, model, feature_names):
        """
        Inicializa o explicador com o modelo treinado e os nomes das features.

        Parameters
        ----------
        model : sklearn.tree.DecisionTreeClassifier
            Árvore de decisão treinada a ser explicada.
        feature_names : list[str]
            Nomes das colunas (features), na mesma ordem usada no treino.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self._tree = model.tree_
        self._mal_idx = list(model.classes_).index(self.CLASSE_MALIGNO)

    def _proba_maligno(self, node: int) -> float:
        """
        Retorna a proporção de malignos — P(Maligno) — registrada em um nó.

        Parameters
        ----------
        node : int
            Índice do nó na estrutura interna da árvore.

        Returns
        -------
        float
            Fração das amostras de treino daquele nó pertencentes à classe Maligno.
        """
        valores = self._tree.value[node][0]
        return valores[self._mal_idx] / valores.sum()

    def _percorrer(self, linha) -> list[int]:
        """
        Caminha da raiz até a folha para uma amostra, reproduzindo a decisão.

        Parameters
        ----------
        linha : sequence[float]
            Valores das características da amostra, na ordem de feature_names.

        Returns
        -------
        list[int]
            Sequência de índices de nós visitados, da raiz (0) até a folha.
        """
        no = 0
        caminho = [0]
        while self._tree.children_left[no] != -1:  # enquanto não for folha
            idx_feat = self._tree.feature[no]
            if linha[idx_feat] <= self._tree.threshold[no]:
                no = self._tree.children_left[no]
            else:
                no = self._tree.children_right[no]
            caminho.append(no)
        return caminho

    def global_importances(self, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Retorna o ranking global de importância das características na árvore.

        Parameters
        ----------
        top_n : int, optional
            Número máximo de características a retornar (padrão 10).

        Returns
        -------
        list[tuple[str, float]]
            Pares (característica, importância) com importância > 0, ordenados
            do maior para o menor. A importância é a redução total de impureza
            atribuída à característica (soma 1 entre todas as features).
        """
        importancias = self.model.feature_importances_
        pares = [
            (self.feature_names[i], float(imp))
            for i, imp in enumerate(importancias) if imp > 0
        ]
        pares.sort(key=lambda p: p[1], reverse=True)
        return pares[:top_n]

    def explain(self, X) -> list[dict]:
        """
        Explica a decisão da árvore para cada amostra do lote.

        Parameters
        ----------
        X : pandas.DataFrame
            Dados sem escalonamento (a árvore foi treinada em dados brutos),
            contendo as colunas de feature_names.

        Returns
        -------
        list[dict]
            Uma explicação por amostra, contendo:
            - 'indice'        : rótulo da linha no DataFrame.
            - 'classe'        : 'Maligno' ou 'Benigno'.
            - 'certeza'       : confiança (%) da classe predita na folha.
            - 'contribuicoes' : lista ordenada por impacto, cada item com
                                {'feature', 'valor', 'contribuicao', 'direcao'}.
            - 'caminho'       : lista de regras (str) da raiz até a folha.
        """
        X = X[self.feature_names]
        valores = X.values
        indices = list(X.index)
        predicoes = self.model.predict(valores)

        children_left = self._tree.children_left
        feature = self._tree.feature
        threshold = self._tree.threshold

        explicacoes = []
        for i in range(valores.shape[0]):
            linha = valores[i]
            caminho_nos = self._percorrer(linha)

            contribuicoes = {}
            regras = []
            for k in range(len(caminho_nos) - 1):
                no_pai = caminho_nos[k]
                no_filho = caminho_nos[k + 1]
                idx_feat = feature[no_pai]
                valor_feat = linha[idx_feat]
                limiar = threshold[no_pai]

                delta = self._proba_maligno(no_filho) - self._proba_maligno(no_pai)
                contribuicoes[idx_feat] = contribuicoes.get(idx_feat, 0.0) + delta

                operador = "≤" if no_filho == children_left[no_pai] else ">"
                regras.append(
                    f"{self.feature_names[idx_feat]} = {valor_feat:.3f}  "
                    f"{operador}  {limiar:.3f}"
                )

            lista_contrib = sorted(
                (
                    {
                        'feature': self.feature_names[idx],
                        'valor': float(linha[idx]),
                        'contribuicao': float(c),
                        'direcao': 'Maligno' if c > 0 else 'Benigno',
                    }
                    for idx, c in contribuicoes.items()
                ),
                key=lambda d: abs(d['contribuicao']),
                reverse=True,
            )

            classe = 'Maligno' if predicoes[i] == self.CLASSE_MALIGNO else 'Benigno'
            p_maligno = self._proba_maligno(caminho_nos[-1])
            certeza = p_maligno if classe == 'Maligno' else (1 - p_maligno)

            explicacoes.append({
                'indice': indices[i],
                'classe': classe,
                'certeza': round(certeza * 100, 1),
                'contribuicoes': lista_contrib,
                'caminho': regras,
            })

        return explicacoes
