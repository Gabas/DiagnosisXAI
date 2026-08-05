"""
Explicadores de interpretabilidade (XAI) dos modelos do DiagnosisXAI.

Este módulo é a **fonte única** das classes explicadoras. Ele é importado
tanto pelo notebook (`notebooks/Wisconsin.ipynb`, ao construir e serializar os
explicadores) quanto pelo aplicativo (ao carregar o `data/wisconsin.pkl`).

Por que viver num módulo importável — e não ser definido no notebook?
--------------------------------------------------------------------
O `cloudpickle` serializa classes definidas no notebook (`__main__`) **por
valor**, embutindo o *bytecode* delas no `.pkl`. O bytecode do CPython não é
compatível entre versões (ex.: 3.11 → 3.12), então um `.pkl` gerado por um
kernel Jupyter numa versão de Python diferente da que roda o app faz os métodos
executarem bytecode inválido — resultando em falhas não determinísticas
(`AttributeError`, `Type does not define the tp_name field`, `Segmentation
fault`, `Illegal instruction`).

Como estas classes vivem aqui, o `cloudpickle` as serializa **por referência**
(apenas `core.explainers` + o nome da classe). O `.pkl` deixa de conter
bytecode: ao carregá-lo, cada ambiente recompila estas classes com o seu
próprio Python. O artefato passa a ser independente da versão do interpretador.
"""

import numpy as np

from core.decision import limitrofe_padrao


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
                'indice': int(indices[i]),
                'classe': classe,
                'certeza': round(certeza * 100, 1),
                'contribuicoes': lista_contrib,
                'caminho': regras,
            })

        return explicacoes


class LogisticRegressionExplainer:
    """
    Gera explicações interpretáveis para um classificador de Regressão Logística.

    Como o modelo é linear sobre os dados padronizados (z-score), a decisão é
    decomposta exatamente: o logito de malignidade vale ``w·x + b``, e a
    contribuição de cada característica é ``wⱼ·xⱼ`` — positiva empurra para
    Maligno, negativa para Benigno. Além disso, projeta cada paciente em
    relação à fronteira de decisão real (o hiperplano ``w·x + b = 0``) para
    visualização honesta em 2D, sem retreinar nenhum modelo auxiliar.

    Attributes
    ----------
    model : sklearn.linear_model.LogisticRegression
        Modelo linear já treinado sobre dados padronizados.
    feature_names : list[str]
        Nomes das características, na ordem usada no treino.
    """

    CLASSE_MALIGNO = 1
    # Máximo de contribuições guardadas por paciente (o relatório exibe as 6 maiores).
    MAX_CONTRIBUICOES = 10

    def __init__(self, model, feature_names):
        """
        Inicializa o explicador a partir do modelo linear treinado.

        Parameters
        ----------
        model : sklearn.linear_model.LogisticRegression
            Regressão logística treinada (sobre dados em escala Z-score).
        feature_names : list[str]
            Nomes das colunas, na mesma ordem usada no treino.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self._w = np.asarray(model.coef_[0], dtype=float)
        self._b = float(model.intercept_[0])
        self._norm_w = float(np.linalg.norm(self._w)) or 1.0
        self._mal_idx = list(model.classes_).index(self.CLASSE_MALIGNO)

    def global_importances(self, top_n: int = 10) -> list[dict]:
        """
        Retorna as características de maior peso na decisão do modelo.

        Como a entrada é padronizada, a magnitude do coeficiente mede a
        importância e o sinal indica a direção.

        Parameters
        ----------
        top_n : int, optional
            Número máximo de características a retornar (padrão 10).

        Returns
        -------
        list[dict]
            Itens {'feature', 'coeficiente', 'direcao'} ordenados pelo módulo
            do coeficiente, do maior para o menor. 'direcao' é 'Maligno' quando
            valores altos da característica indicam malignidade, 'Benigno' caso
            contrário.
        """
        itens = [
            {
                'feature': self.feature_names[j],
                'coeficiente': float(self._w[j]),
                'direcao': 'Maligno' if self._w[j] > 0 else 'Benigno',
            }
            for j in range(len(self._w))
        ]
        itens.sort(key=lambda d: abs(d['coeficiente']), reverse=True)
        return itens[:top_n]

    def explain(self, X_scaled, X_raw) -> list[dict]:
        """
        Explica a decisão da regressão logística para cada amostra do lote.

        Parameters
        ----------
        X_scaled : pandas.DataFrame
            Dados padronizados (Z-score) — a entrada efetiva do modelo.
        X_raw : pandas.DataFrame
            Mesmos dados sem escalonamento, usados apenas para exibir os
            valores reais das características ao usuário.

        Returns
        -------
        list[dict]
            Uma explicação por amostra, contendo:
            - 'indice'        : rótulo da linha.
            - 'classe'        : 'Maligno' ou 'Benigno'.
            - 'probabilidade' : P(Maligno) em %, segundo o modelo.
            - 'distancia'     : distância (com sinal) até a fronteira de decisão.
            - 'limitrofe'     : True se o modelo está pouco decidido.
            - 'contribuicoes' : lista ordenada por impacto, cada item com
                                {'feature', 'valor', 'acima_media',
                                 'contribuicao', 'direcao'}.
            - 'x_plot'/'y_plot': coordenadas para o gráfico da fronteira.
        """
        Xs = X_scaled[self.feature_names]
        Xr = X_raw[self.feature_names]
        z = Xs.values
        bruto = Xr.values
        indices = list(Xs.index)

        decisao = self.model.decision_function(z)          # w·x + b (logito)
        prob = self.model.predict_proba(z)[:, self._mal_idx]
        predicoes = self.model.predict(z)
        distancia = decisao / self._norm_w                 # distância ao hiperplano
        y_plot = self._eixo_ortogonal(z)

        explicacoes = []
        for i in range(z.shape[0]):
            contribs = self._w * z[i]                       # wⱼ·zⱼ
            total_abs = float(np.abs(contribs).sum()) or 1.0
            itens = [
                {
                    'feature': self.feature_names[j],
                    'valor': float(bruto[i, j]),
                    'acima_media': bool(z[i, j] >= 0),
                    'contribuicao': float(contribs[j]),
                    'direcao': 'Maligno' if contribs[j] > 0 else 'Benigno',
                    'peso_pct': round(abs(float(contribs[j])) / total_abs * 100, 1),
                }
                for j in range(len(self._w))
            ]
            itens.sort(key=lambda d: abs(d['contribuicao']), reverse=True)
            lista = itens[:self.MAX_CONTRIBUICOES]

            p = float(prob[i])
            explicacoes.append({
                'indice': int(indices[i]),
                'classe': 'Maligno' if predicoes[i] == self.CLASSE_MALIGNO else 'Benigno',
                'probabilidade': round(p * 100, 1),
                'distancia': float(distancia[i]),
                'limitrofe': limitrofe_padrao(p),
                'contribuicoes': lista,
                'x_plot': float(distancia[i]),
                'y_plot': float(y_plot[i]),
            })

        return explicacoes

    def _eixo_ortogonal(self, z):
        """
        Calcula, para cada amostra, a coordenada no eixo ortogonal à fronteira.

        Remove de cada amostra a componente na direção de ``w`` (a direção que
        decide o diagnóstico) e projeta o resíduo na direção de maior variância.
        Esse eixo não altera a decisão, servindo apenas para espalhar os
        pacientes no gráfico sem distorcer a posição da fronteira.

        Parameters
        ----------
        z : numpy.ndarray
            Matriz (n_amostras × n_features) dos dados padronizados.

        Returns
        -------
        numpy.ndarray
            Vetor com a coordenada do eixo ortogonal de cada amostra.
        """
        w_unit = self._w / self._norm_w
        residuo = z - np.outer(z @ w_unit, w_unit)
        centrado = residuo - residuo.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centrado, full_matrices=False)
            direcao = vt[0]
        except np.linalg.LinAlgError:
            return np.zeros(z.shape[0])
        return residuo @ direcao


class KNNExplainer:
    """
    Explica o K-Nearest Neighbors pela vizinhança que fundamenta cada decisão.

    Para cada paciente, identifica os K pacientes de treino mais próximos (no
    espaço padronizado de 30 biomarcadores) e decompõe a decisão no voto deles.
    Quando o modelo pondera o voto pela distância (``weights='distance'``), a
    contagem bruta de vizinhos por classe pode não bater com a decisão real —
    com K par, por exemplo, um empate de 2×2 ainda é desempatado pelo peso do
    vizinho mais próximo. Por isso a confiança exibida vem sempre de
    ``predict_proba`` (a mesma conta do modelo) e o voto é decomposto também
    em peso ponderado por classe, não só na contagem bruta. Fornece ainda a
    importância global por permutação e a projeção 2D usada para situar o
    paciente entre os vizinhos.
    """

    CLASSE_MALIGNO = 1

    def __init__(self, model, feature_names, y_train, pca, train_2d, importancias):
        self.model = model
        self.feature_names = list(feature_names)
        self.k = int(model.n_neighbors)
        self._pondera_distancia = model.weights == 'distance'
        self._y_train = np.asarray(y_train)
        self._pca = pca
        self._train_2d = np.asarray(train_2d, dtype=float)
        self._importancias = [(f, float(v)) for f, v in importancias]

    def global_importances(self, top_n=10):
        pares = [(f, v) for f, v in self._importancias if v > 0]
        pares.sort(key=lambda p: p[1], reverse=True)
        return pares[:top_n]

    def contexto(self):
        """Mapa 2D dos pacientes de treino — fundo do gráfico de vizinhança."""
        return {
            'k': self.k,
            'train_2d': self._train_2d.tolist(),
            'train_y': self._y_train.tolist(),
        }

    def _pesos_vizinhos(self, vd: np.ndarray) -> np.ndarray:
        """
        Peso de cada vizinho na votação, replicando a fórmula do sklearn.

        Com ``weights='uniform'`` todo vizinho pesa igual. Com
        ``weights='distance'`` o peso é ``1/distância`` — e, se algum vizinho
        coincide exatamente com o paciente (distância 0), o sklearn dá peso
        total a esse(s) vizinho(s) e ignora os demais, em vez de dividir por
        zero.

        Parameters
        ----------
        vd : numpy.ndarray
            Distâncias do paciente aos seus K vizinhos.

        Returns
        -------
        numpy.ndarray
            Pesos (não normalizados) de cada vizinho, na mesma ordem de ``vd``.
        """
        if not self._pondera_distancia:
            return np.ones_like(vd)
        with np.errstate(divide='ignore'):
            pesos = 1.0 / vd
        empatados_em_zero = np.isinf(pesos)
        if empatados_em_zero.any():
            return empatados_em_zero.astype(float)
        return pesos

    def explain(self, X_scaled):
        Xs = X_scaled[self.feature_names]
        z = Xs.values
        indices = list(Xs.index)
        dist, viz_idx = self.model.kneighbors(z)
        pred = self.model.predict(z)
        mal_idx = list(self.model.classes_).index(self.CLASSE_MALIGNO)
        proba = self.model.predict_proba(z)[:, mal_idx]
        query_2d = self._pca.transform(z)

        explicacoes = []
        for i in range(z.shape[0]):
            vz, vd = viz_idx[i], dist[i]
            classes_viz = self._y_train[vz]
            votos_mal = int((classes_viz == self.CLASSE_MALIGNO).sum())
            votos_ben = self.k - votos_mal

            pesos = self._pesos_vizinhos(vd)
            peso_total = pesos.sum()
            peso_mal = float(pesos[classes_viz == self.CLASSE_MALIGNO].sum() / peso_total)

            classe = 'Maligno' if pred[i] == self.CLASSE_MALIGNO else 'Benigno'
            p = float(proba[i])
            confianca = p if classe == 'Maligno' else (1.0 - p)
            vizinhos = [
                {'indice': int(vz[j]),
                 'classe': 'Maligno' if classes_viz[j] == self.CLASSE_MALIGNO else 'Benigno',
                 'distancia': float(vd[j])}
                for j in range(self.k)
            ]
            explicacoes.append({
                'indice': int(indices[i]),
                'classe': classe,
                'confianca': round(confianca * 100, 1),
                'votos_maligno': votos_mal,
                'votos_benigno': votos_ben,
                'peso_maligno': round(peso_mal * 100, 1),
                'peso_benigno': round((1.0 - peso_mal) * 100, 1),
                'pondera_distancia': self._pondera_distancia,
                'limitrofe': limitrofe_padrao(p),
                'vizinhos': vizinhos,
                'x_plot': float(query_2d[i, 0]),
                'y_plot': float(query_2d[i, 1]),
            })
        return explicacoes


class RandomForestExplainer:
    """
    Explica um Random Forest pelo consenso das árvores que fundamenta a decisão.

    A floresta decide por média das probabilidades das N árvores (voto suave):
    a probabilidade final é a média e a classe é a de maior probabilidade. Este
    explicador expõe, por paciente, quantas árvores votaram em cada classe (voto
    duro) e a distribuição das probabilidades das árvores — o análogo, para o
    ensemble, do voto dos vizinhos no KNN: mostra se a decisão é um consenso
    forte ou uma maioria apertada. Fornece ainda a importância global (redução
    de impureza / Gini agregada pela floresta).

    Complementa o relatório SHAP do RF (que detalha a contribuição de cada
    atributo), focando na estrutura de votação do ensemble.

    Attributes
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Floresta já treinada.
    feature_names : list[str]
        Nomes das características, na ordem usada no treino.
    n_arvores : int
        Número de árvores (estimadores) da floresta.
    """

    CLASSE_MALIGNO = 1
    N_BINS = 10  # bins do histograma de probabilidades das árvores (consenso)

    def __init__(self, model, feature_names):
        """
        Inicializa o explicador a partir da floresta treinada.

        Parameters
        ----------
        model : sklearn.ensemble.RandomForestClassifier
            Random Forest treinado (sobre dados padronizados).
        feature_names : list[str]
            Nomes das colunas, na mesma ordem usada no treino.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.n_arvores = int(len(model.estimators_))
        self._mal_idx = list(model.classes_).index(self.CLASSE_MALIGNO)

    def global_importances(self, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Retorna o ranking global de importância (redução de impureza / Gini).

        Parameters
        ----------
        top_n : int, optional
            Número máximo de características a retornar (padrão 10).

        Returns
        -------
        list[tuple[str, float]]
            Pares (característica, importância) com importância > 0, do maior
            para o menor. A importância soma 1 entre todas as features.
        """
        importancias = self.model.feature_importances_
        pares = [
            (self.feature_names[i], float(imp))
            for i, imp in enumerate(importancias) if imp > 0
        ]
        pares.sort(key=lambda p: p[1], reverse=True)
        return pares[:top_n]

    def contexto(self) -> dict:
        """Metadados da floresta (cabeçalho e eixo do histograma do relatório)."""
        return {'n_arvores': self.n_arvores, 'n_bins': self.N_BINS}

    def explain(self, X_scaled) -> list[dict]:
        """
        Explica a decisão da floresta para cada amostra do lote.

        Parameters
        ----------
        X_scaled : pandas.DataFrame
            Dados padronizados (Z-score) — a entrada efetiva do modelo.

        Returns
        -------
        list[dict]
            Uma explicação por amostra, contendo:
            - 'indice'        : rótulo da linha.
            - 'classe'        : 'Maligno' ou 'Benigno' (predição da floresta).
            - 'probabilidade' : P(Maligno) em %, média das árvores.
            - 'confianca'     : probabilidade da classe predita, em %.
            - 'votos_maligno' / 'votos_benigno' : nº de árvores por classe (voto duro).
            - 'limitrofe'     : True se a floresta está pouco decidida.
            - 'hist'          : contagem de árvores por faixa de P(Maligno)
                                (N_BINS faixas de 0 a 1) — o consenso do ensemble.
        """
        Xs = X_scaled[self.feature_names]
        valores = Xs.values
        indices = list(Xs.index)

        # P(Maligno) de cada árvore para cada paciente: (n_arvores, n_pacientes).
        probs_arvores = np.stack([
            est.predict_proba(valores)[:, self._mal_idx]
            for est in self.model.estimators_
        ])
        prob_media = probs_arvores.mean(axis=0)          # == RandomForest.predict_proba
        votos_mal = (probs_arvores >= 0.5).sum(axis=0)   # voto duro por paciente
        predicoes = self.model.predict(valores)

        explicacoes = []
        for i in range(valores.shape[0]):
            classe = 'Maligno' if predicoes[i] == self.CLASSE_MALIGNO else 'Benigno'
            p = float(prob_media[i])
            confianca = p if classe == 'Maligno' else (1.0 - p)
            vm = int(votos_mal[i])
            hist, _ = np.histogram(probs_arvores[:, i], bins=self.N_BINS, range=(0.0, 1.0))
            explicacoes.append({
                'indice': int(indices[i]),
                'classe': classe,
                'probabilidade': round(p * 100, 1),
                'confianca': round(confianca * 100, 1),
                'votos_maligno': vm,
                'votos_benigno': self.n_arvores - vm,
                'limitrofe': limitrofe_padrao(p),
                'hist': [int(h) for h in hist],
            })
        return explicacoes


class SVMExplainer:
    """
    Explica um SVM de kernel RBF pelos vetores de suporte que fundamentam a decisão.

    Diferente de um SVM linear, o kernel RBF não define um hiperplano no espaço
    original das características: a decisão de cada paciente é, na verdade, uma
    soma ponderada da sua similaridade (kernel gaussiano) com os vetores de
    suporte do treino::

        decision_function(x) = Σᵢ dual_coefᵢ · K(x, svᵢ) + b
        K(x, sv) = exp(-γ · ‖x - sv‖²)

    Este explicador reproduz exatamente essa soma (validado contra
    ``model.decision_function`` — diferença de ponto flutuante apenas) e expõe,
    por paciente, quais pacientes de treino (vetores de suporte) mais pesaram na
    decisão — o análogo, para o SVM, do voto dos vizinhos no KNN, mas fiel à
    matemática exata do kernel, sem qualquer modelo auxiliar ou projeção
    aproximada. A posição 2D de cada paciente (para o mapa populacional) é
    calculada fora deste explicador, pela mesma projeção UMAP usada no restante
    do app.

    Attributes
    ----------
    model : sklearn.svm.SVC
        SVM (kernel RBF) já treinado sobre dados padronizados.
    feature_names : list[str]
        Nomes das características, na ordem usada no treino.
    n_support : int
        Número total de vetores de suporte do modelo.
    """

    CLASSE_MALIGNO = 1
    # Nº de vetores de suporte guardados por lado (Maligno/Benigno) no relatório.
    MAX_POR_LADO = 5

    def __init__(self, model, feature_names, y_train, train_2d, importancias):
        """
        Inicializa o explicador a partir do SVM treinado.

        Parameters
        ----------
        model : sklearn.svm.SVC
            SVM de kernel RBF treinado (sobre dados padronizados), com
            ``probability=True``.
        feature_names : list[str]
            Nomes das colunas, na mesma ordem usada no treino.
        y_train : array-like
            Rótulos reais dos pacientes de treino (0/1) — usados para rotular
            a classe de cada vetor de suporte.
        train_2d : array-like
            Embedding UMAP 2D dos pacientes de treino (o mesmo usado pelo Mapa
            Populacional), alinhado por índice com ``y_train``.
        importancias : list[tuple[str, float]]
            Importância global por permutação, pré-calculada no notebook.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self._mal_idx = list(model.classes_).index(self.CLASSE_MALIGNO)

        # Vetores de suporte e coeficientes duais (alpha_i · y_i, já com sinal) —
        # tudo o que é necessário para reconstruir exatamente a decision_function.
        self._sv = np.asarray(model.support_vectors_, dtype=float)
        self._dual_coef = np.asarray(model.dual_coef_[0], dtype=float)
        self._intercept = float(model.intercept_[0])
        self._gamma = float(model._gamma)  # valor resolvido de gamma='scale'/'auto'
        self._sv_train_idx = np.asarray(model.support_, dtype=int)

        self._y_train = np.asarray(y_train)
        self._sv_y = self._y_train[self._sv_train_idx]
        self._sv_mal_mask = self._sv_y == self.CLASSE_MALIGNO
        self._train_2d = np.asarray(train_2d, dtype=float)
        self.n_support = int(len(self._sv))
        self._importancias = [(f, float(v)) for f, v in importancias]

    def global_importances(self, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Retorna as características de maior impacto na decisão, por permutação.

        Parameters
        ----------
        top_n : int, optional
            Número máximo de características a retornar (padrão 10).

        Returns
        -------
        list[tuple[str, float]]
            Pares (característica, importância) com importância > 0, do maior
            para o menor.
        """
        pares = [(f, v) for f, v in self._importancias if v > 0]
        pares.sort(key=lambda p: p[1], reverse=True)
        return pares[:top_n]

    def contexto(self) -> dict:
        """
        Mapa 2D dos pacientes de treino e dos vetores de suporte (fundo do mapa).

        Returns
        -------
        dict
            {'train_2d', 'train_y', 'sv_indices', 'n_support',
             'n_support_maligno', 'n_support_benigno'}.
        """
        return {
            'train_2d': self._train_2d.tolist(),
            'train_y': self._y_train.tolist(),
            'sv_indices': self._sv_train_idx.tolist(),
            'n_support': self.n_support,
            'n_support_maligno': int((self._sv_y == self.CLASSE_MALIGNO).sum()),
            'n_support_benigno': int((self._sv_y != self.CLASSE_MALIGNO).sum()),
        }

    def explain(self, X_scaled) -> list[dict]:
        """
        Explica a decisão do SVM para cada amostra do lote.

        Como a soma é feita sobre todos os vetores de suporte (dezenas a
        centenas), listar apenas os "top-N por magnitude" pode enganar: o
        sinal de cada vetor é fixo (positivo se ele for Maligno, negativo se
        Benigno — o kernel RBF nunca é negativo), então os termos maiores de
        um único lado não revelam se esse lado de fato venceu a soma. Por
        isso a explicação é um balanço de forças (soma total a favor de cada
        classe, mais o viés do modelo) e não um ranking único.

        Parameters
        ----------
        X_scaled : pandas.DataFrame
            Dados padronizados (Z-score) — a entrada efetiva do modelo.

        Returns
        -------
        list[dict]
            Uma explicação por amostra, contendo:
            - 'indice'        : rótulo da linha.
            - 'classe'        : 'Maligno' ou 'Benigno'.
            - 'probabilidade' : P(Maligno) em %, segundo o modelo.
            - 'confianca'     : probabilidade da classe predita, em %.
            - 'distancia'     : valor (com sinal) da decision_function —
                                distância à margem no espaço do kernel.
            - 'limitrofe'     : True se o modelo está pouco decidido.
            - 'vies'          : termo independente (b) do modelo.
            - 'forca_maligno' : soma das contribuições dos vetores de suporte
                                Maligno (sempre ≥ 0 — pull total para Maligno).
            - 'forca_benigno' : soma (em módulo) das contribuições dos vetores
                                de suporte Benigno — pull total para Benigno.
                                Vale: forca_maligno - forca_benigno + vies ==
                                distancia.
            - 'top_maligno' / 'top_benigno' : os vetores de suporte mais
                                influentes de cada lado, cada item com
                                {'indice_treino', 'classe', 'contribuicao',
                                 'similaridade'}.
        """
        Xs = X_scaled[self.feature_names]
        z = Xs.values
        indices = list(Xs.index)

        decisao = self.model.decision_function(z)
        prob = self.model.predict_proba(z)[:, self._mal_idx]
        predicoes = self.model.predict(z)

        # Kernel RBF entre cada paciente do lote e cada vetor de suporte:
        # K(x, sv) = exp(-gamma * ||x - sv||^2) — mesma fórmula usada pelo SVM.
        sq_dists = ((z[:, None, :] - self._sv[None, :, :]) ** 2).sum(axis=2)
        K = np.exp(-self._gamma * sq_dists)
        contrib_matrix = K * self._dual_coef[None, :]  # contribuição de cada SV

        idx_mal = np.where(self._sv_mal_mask)[0]
        idx_ben = np.where(~self._sv_mal_mask)[0]

        def _top(indices_lado, contribs, n):
            ordem = indices_lado[np.argsort(-np.abs(contribs[indices_lado]))][:n]
            return [
                {
                    'indice_treino': int(self._sv_train_idx[j]),
                    'classe': 'Maligno' if self._sv_mal_mask[j] else 'Benigno',
                    'contribuicao': float(contribs[j]),
                    'similaridade': float(K_row[j]),
                }
                for j in ordem
            ]

        explicacoes = []
        for i in range(z.shape[0]):
            contribs = contrib_matrix[i]
            K_row = K[i]
            forca_maligno = float(contribs[idx_mal].sum())
            forca_benigno = float(-contribs[idx_ben].sum())

            p = float(prob[i])
            classe = 'Maligno' if predicoes[i] == self.CLASSE_MALIGNO else 'Benigno'
            confianca = p if classe == 'Maligno' else (1.0 - p)
            explicacoes.append({
                'indice': int(indices[i]),
                'classe': classe,
                'probabilidade': round(p * 100, 1),
                'confianca': round(confianca * 100, 1),
                'distancia': float(decisao[i]),
                'limitrofe': limitrofe_padrao(p),
                'vies': self._intercept,
                'forca_maligno': forca_maligno,
                'forca_benigno': forca_benigno,
                'top_maligno': _top(idx_mal, contribs, self.MAX_POR_LADO),
                'top_benigno': _top(idx_ben, contribs, self.MAX_POR_LADO),
            })

        return explicacoes
