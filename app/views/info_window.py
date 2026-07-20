"""
Módulo contendo a janela de detalhes dos modelos e tecnologias (aba "Sobre").

Ao clicar num modelo de IA ou numa tecnologia de explicabilidade, abre-se uma
janela explicando o que é, como funciona, os parâmetros usados e — no caso dos
modelos — um exemplo de código com a inicialização.
"""

import customtkinter as ctk

from utils.ui import ScrollableFrame, responsive_geometry


# Conteúdo didático de cada modelo/tecnologia, indexado por uma chave curta.
CONTEUDO = {
    "dt": {
        "titulo": "Árvore de Decisão",
        "subtitulo": "Decision Tree — modelo baseado em regras",
        "secoes": [
            ("O que é",
             "A Árvore de Decisão classifica um paciente respondendo a uma sequência "
             "de perguntas sobre seus biomarcadores, organizadas em forma de árvore. "
             "Cada nó interno testa um atributo contra um limiar; cada folha atribui "
             "um diagnóstico (Benigno ou Maligno)."),
            ("Como funciona",
             "No treino, o algoritmo escolhe em cada nó o atributo e o limiar que melhor "
             "separam as classes, medindo a pureza pela entropia (ganho de informação). "
             "O processo se repete recursivamente até as folhas ficarem puras. Na predição, "
             "o paciente percorre a árvore da raiz à folha — e esse caminho de regras é a "
             "própria explicação da decisão."),
            ("Parâmetros usados",
             "criterion='entropy' — usa o ganho de informação para escolher as divisões.\n"
             "random_state=42 — garante reprodutibilidade.\n"
             "É o único modelo treinado nos dados brutos, pois é invariante à escala "
             "(não depende da padronização Z-score)."),
            ("Vantagens e limitações",
             "Totalmente interpretável — dá para seguir exatamente a lógica da decisão. "
             "Em contrapartida, uma única árvore tende a decorar o treino (overfitting), "
             "o que motiva o uso do Random Forest."),
        ],
        "codigo":
            "from sklearn.tree import DecisionTreeClassifier\n\n"
            "# A árvore é invariante à escala: usa os dados brutos (sem Z-score)\n"
            "modelo = DecisionTreeClassifier(\n"
            "    criterion='entropy',   # mede a pureza dos nós pelo ganho de informação\n"
            "    random_state=42,       # reprodutibilidade\n"
            ")\n"
            "modelo.fit(X_train, y_train)\n"
            "predicao = modelo.predict(X_novo)",
    },
    "knn": {
        "titulo": "K-Nearest Neighbors (KNN)",
        "subtitulo": "Classificação por proximidade entre pacientes",
        "secoes": [
            ("O que é",
             "O KNN não constrói um modelo no sentido tradicional: ele memoriza os "
             "pacientes de treino e classifica cada novo paciente pela maioria entre "
             "seus K vizinhos mais próximos no espaço dos 30 biomarcadores."),
            ("Como funciona",
             "Para um novo paciente, calcula a distância (Manhattan) até todos os "
             "pacientes de treino, seleciona os K=4 mais próximos e adota o diagnóstico "
             "da maioria — mas o voto de cada vizinho é ponderado pelo inverso da "
             "distância, então os mais próximos pesam mais. Por depender de distância, "
             "exige dados padronizados (Z-score), para que todos os atributos pesem "
             "igualmente."),
            ("Parâmetros usados",
             "n_neighbors=4 — número de vizinhos que votam.\n"
             "weights='distance' — vizinhos mais próximos têm peso maior no voto.\n"
             "metric='manhattan' — distância como soma das diferenças absolutas, em vez "
             "da euclidiana.\n"
             "Combinação escolhida por validação cruzada (5-fold) no treino — não só "
             "pela acurácia no teste — por generalizar melhor que k=20 com voto uniforme."),
            ("Vantagens e limitações",
             "Simples e sem suposições sobre a forma dos dados; a explicação é intuitiva "
             "(os vizinhos que decidiram). Fica mais lento com muitos dados, pois compara "
             "com todo o conjunto de treino a cada predição. Um K pequeno como 4 é mais "
             "sensível a ruído local, por isso pondera por distância em vez de voto igual."),
        ],
        "codigo":
            "from sklearn.neighbors import KNeighborsClassifier\n\n"
            "# Exige dados padronizados (Z-score), pois classifica por distância\n"
            "modelo = KNeighborsClassifier(\n"
            "    n_neighbors=4,          # nº de vizinhos que votam\n"
            "    weights='distance',    # vizinhos mais próximos pesam mais\n"
            "    metric='manhattan',    # soma das diferenças absolutas\n"
            ")\n"
            "modelo.fit(X_train_scaled, y_train)\n"
            "predicao = modelo.predict(X_novo_scaled)",
    },
    "lr": {
        "titulo": "Regressão Logística",
        "subtitulo": "Logistic Regression — modelo linear probabilístico",
        "secoes": [
            ("O que é",
             "A Regressão Logística é um modelo linear que estima a probabilidade de um "
             "tumor ser maligno. Combina os 30 biomarcadores com pesos e passa o resultado "
             "por uma função sigmoide, que o comprime para o intervalo 0–1 (probabilidade)."),
            ("Como funciona",
             "Calcula o logito = w·x + b (soma ponderada dos atributos) e aplica a sigmoide "
             "para obter P(Maligno); se P ≥ 0,5, classifica como Maligno. O treino ajusta os "
             "pesos w para maximizar a verossimilhança, com regularização L2 controlada por "
             "C (solver liblinear). Por usar combinação linear, requer dados padronizados."),
            ("Parâmetros usados",
             "C=0.1 — força da regularização L2 (inverso; quanto menor, mais forte). "
             "Escolhido por validação cruzada no treino — generaliza melhor que o padrão "
             "(C=1.0) neste dataset.\n"
             "solver='liblinear' — otimizador eficiente para bases pequenas/médias.\n"
             "class_weight='balanced' — dá mais peso à classe minoritária, compensando o "
             "desbalanceamento.\n"
             "max_iter=2000 — teto de iterações para garantir convergência.\n"
             "random_state=42."),
            ("Vantagens e limitações",
             "Rápida, estável e interpretável — o peso de cada atributo indica sua influência "
             "e direção. Assume uma fronteira de decisão essencialmente linear, o que limita "
             "padrões muito complexos."),
        ],
        "codigo":
            "from sklearn.linear_model import LogisticRegression\n\n"
            "# Modelo linear; usa dados padronizados (Z-score)\n"
            "modelo = LogisticRegression(\n"
            "    C=0.1,                     # regularização L2 (validada por cross-validation)\n"
            "    solver='liblinear',        # otimizador para bases pequenas/médias\n"
            "    class_weight='balanced',   # compensa o desbalanceamento entre classes\n"
            "    max_iter=2000,             # iterações suficientes para convergir\n"
            "    random_state=42,\n"
            ")\n"
            "modelo.fit(X_train_scaled, y_train)\n"
            "prob_maligno = modelo.predict_proba(X_novo_scaled)[:, 1]",
    },
    "rf": {
        "titulo": "Random Forest",
        "subtitulo": "Floresta Aleatória — ensemble de árvores (bagging)",
        "secoes": [
            ("O que é",
             "O Random Forest é um ensemble (comitê) de muitas Árvores de Decisão. Em vez "
             "de confiar em uma única árvore, combina o voto de 500 árvores para uma decisão "
             "mais robusta e estável."),
            ("Como funciona",
             "Cada árvore é treinada em uma amostra aleatória dos dados (bootstrap) e "
             "considera apenas um subconjunto aleatório de atributos em cada divisão. Essa "
             "aleatoriedade (bagging) faz as árvores errarem de formas diferentes; na "
             "combinação, os erros se cancelam e o overfitting diminui. A predição final é a "
             "média das probabilidades das árvores (voto suave), não uma contagem de votos."),
            ("Parâmetros usados",
             "n_estimators=500 — número de árvores no comitê.\n"
             "max_depth=10, min_samples_split=4, min_samples_leaf=2, max_features='sqrt' — "
             "limitam o crescimento de cada árvore; escolhidos por validação cruzada no "
             "treino, para generalizar melhor que os parâmetros padrão.\n"
             "class_weight='balanced_subsample' — compensa o desbalanceamento entre classes, "
             "recalculado a cada amostra bootstrap.\n"
             "n_jobs=-1 — treina as árvores em paralelo, usando todos os núcleos da CPU.\n"
             "random_state=42."),
            ("Vantagens e limitações",
             "Alta acurácia e robustez, com pouca necessidade de ajuste. É menos interpretável "
             "que uma árvore única — por isso o notebook usa SHAP para explicá-lo."),
        ],
        "codigo":
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "modelo = RandomForestClassifier(\n"
            "    n_estimators=500,          # nº de árvores no ensemble\n"
            "    max_depth=10,              # profundidade máxima de cada árvore\n"
            "    min_samples_split=4,       # mín. de amostras para dividir um nó\n"
            "    min_samples_leaf=2,        # mín. de amostras em cada folha\n"
            "    max_features='sqrt',       # nº de atributos considerados por divisão\n"
            "    class_weight='balanced_subsample',  # compensa o desbalanceamento\n"
            "    n_jobs=-1,                 # usa todos os núcleos da CPU\n"
            "    random_state=42,\n"
            ")\n"
            "modelo.fit(X_train_scaled, y_train)\n"
            "predicao = modelo.predict(X_novo_scaled)",
    },
    "svm": {
        "titulo": "Support Vector Machine (SVM)",
        "subtitulo": "Máquina de Vetores de Suporte — kernel RBF",
        "secoes": [
            ("O que é",
             "A SVM procura o hiperplano que separa as classes (benigno × maligno) com a "
             "maior margem possível — ou seja, a maior 'faixa de segurança' entre os grupos."),
            ("Como funciona",
             "Quando as classes não são separáveis por uma reta, a SVM usa um kernel (aqui, "
             "o RBF/gaussiano) para projetar os dados em um espaço de dimensão maior, onde a "
             "separação por um hiperplano se torna possível. A decisão é guiada pelos pontos "
             "mais próximos da fronteira (os vetores de suporte). Usa dados padronizados."),
            ("Parâmetros usados",
             "kernel='rbf' — fronteira não-linear via kernel gaussiano.\n"
             "class_weight='balanced' — compensa o desbalanceamento entre classes.\n"
             "probability=True — habilita a estimativa de probabilidade (predict_proba).\n"
             "random_state=42."),
            ("Vantagens e limitações",
             "Muito eficaz em espaços de alta dimensão e com fronteiras complexas. É pouco "
             "interpretável e mais sensível à escolha de hiperparâmetros."),
        ],
        "codigo":
            "from sklearn.svm import SVC\n\n"
            "modelo = SVC(\n"
            "    kernel='rbf',              # fronteira não-linear (kernel gaussiano)\n"
            "    class_weight='balanced',   # compensa o desbalanceamento entre classes\n"
            "    probability=True,          # habilita predict_proba\n"
            "    random_state=42,\n"
            ")\n"
            "modelo.fit(X_train_scaled, y_train)\n"
            "predicao = modelo.predict(X_novo_scaled)",
    },
    "shap": {
        "titulo": "SHAP — SHapley Additive exPlanations",
        "subtitulo": "Explicabilidade baseada na teoria dos jogos",
        "secoes": [
            ("O que é",
             "SHAP é um método de explicabilidade que atribui a cada atributo uma "
             "'contribuição justa' para a predição de um modelo, usando os valores de "
             "Shapley — um conceito da teoria dos jogos cooperativos."),
            ("Como funciona",
             "A ideia é medir quanto cada biomarcador contribui para afastar a predição do "
             "valor médio (base), considerando todas as combinações possíveis de atributos. "
             "O resultado é aditivo: valor base + soma das contribuições SHAP = predição do "
             "modelo. Assim, dá para ver quais biomarcadores empurraram o diagnóstico para "
             "Maligno ou Benigno, tanto globalmente quanto para um paciente específico."),
            ("No projeto",
             "No notebook, o SHAP é aplicado sobre o Random Forest: o summary plot mostra a "
             "importância global dos atributos, e o waterfall plot explica a decisão de um "
             "paciente individual."),
        ],
        "codigo":
            "import shap\n\n"
            "# Explicador sobre o Random Forest treinado\n"
            "explainer   = shap.Explainer(rf, X_train_scaled)\n"
            "shap_values = explainer(X_scaled)\n\n"
            "shap.summary_plot(shap_values, X_scaled)   # importância global\n"
            "shap.plots.waterfall(shap_values[0])       # explicação de 1 paciente",
    },
    "umap": {
        "titulo": "UMAP — Uniform Manifold Approximation and Projection",
        "subtitulo": "Redução de dimensionalidade não-linear",
        "secoes": [
            ("O que é",
             "UMAP é uma técnica de redução de dimensionalidade que projeta dados de muitas "
             "dimensões (aqui, 30 biomarcadores) para 2D, permitindo visualizá-los em um "
             "gráfico."),
            ("Como funciona",
             "Ele modela a estrutura (o 'manifold') dos dados em alta dimensão construindo um "
             "grafo de vizinhanças e, em seguida, encontra uma disposição em 2D que preserva "
             "ao máximo essas vizinhanças — mantendo próximos os pontos que eram próximos no "
             "espaço original. Diferente do PCA (linear), o UMAP captura relações não-lineares."),
            ("No projeto",
             "No notebook, o UMAP projeta os pacientes em 2D para visualizar a separabilidade "
             "entre tumores benignos e malignos — quanto mais separados os grupos, mais "
             "'aprendível' é o problema."),
        ],
        "codigo":
            "import umap\n\n"
            "reducer   = umap.UMAP(n_components=2, random_state=42)\n"
            "embedding = reducer.fit_transform(X_scaled)   # 30 dimensões -> 2\n\n"
            "plt.scatter(embedding[:, 0], embedding[:, 1], c=y)",
    },
}


class InfoWindow(ctk.CTkToplevel):
    """
    Janela secundária com a explicação detalhada de um modelo ou tecnologia.

    Renderiza o conteúdo de CONTEUDO[chave] em uma área rolável: título,
    subtítulo, seções explicativas e, quando houver, um bloco com exemplo de
    código.
    """

    def __init__(self, master, chave: str, **kwargs):
        """
        Inicializa a janela de detalhes.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou a abertura.
        chave : str
            Identificador do conteúdo em CONTEUDO (ex.: 'rf', 'shap').
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        dados = CONTEUDO.get(chave, {"titulo": "Detalhes", "secoes": []})
        self.title(dados.get("titulo", "Detalhes"))
        responsive_geometry(self, 740, 700, min_width=420, min_height=380)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build(dados)
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build(self, dados: dict):
        """Renderiza o conteúdo (título, seções e código) em área rolável."""
        scroll = ScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            scroll, text=dados.get("titulo", ""), anchor="w", justify="left",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(fill="x", padx=24, pady=(20, 2))

        if dados.get("subtitulo"):
            ctk.CTkLabel(
                scroll, text=dados["subtitulo"], anchor="w",
                font=ctk.CTkFont(size=14), text_color="gray",
            ).pack(fill="x", padx=24, pady=(0, 12))

        for titulo, texto in dados.get("secoes", []):
            ctk.CTkLabel(
                scroll, text=titulo, anchor="w",
                font=ctk.CTkFont(size=16, weight="bold"),
            ).pack(fill="x", padx=24, pady=(14, 4))
            ctk.CTkLabel(
                scroll, text=texto, anchor="w", justify="left", wraplength=660,
                font=ctk.CTkFont(size=13), text_color=("gray20", "gray85"),
            ).pack(fill="x", padx=24, pady=(0, 2))

        if dados.get("codigo"):
            ctk.CTkLabel(
                scroll, text="Exemplo de código", anchor="w",
                font=ctk.CTkFont(size=16, weight="bold"),
            ).pack(fill="x", padx=24, pady=(18, 4))

            codigo = dados["codigo"]
            n_linhas = codigo.count("\n") + 1
            caixa = ctk.CTkTextbox(
                scroll, wrap="none", fg_color="#1e1e1e", text_color="#e6e6e6",
                font=ctk.CTkFont(family="Courier New", size=12),
                height=min(max(n_linhas * 20 + 16, 90), 360),
            )
            caixa.pack(fill="x", padx=24, pady=(0, 24))
            caixa.insert("1.0", codigo)
            caixa.configure(state="disabled")
