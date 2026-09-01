"""
Módulo contendo a aba de informações sobre o projeto e a base de dados.
"""

import customtkinter as ctk
from core.biomarkers import (
    BIOMARCADORES_BASE,
    ESTATISTICAS,
    GLOSSARIO_INTRO,
    GLOSSARIO_SUBTITULO,
    GLOSSARIO_UNIDADES,
)
from core.calculos import (
    DUAS_PORCENTAGENS,
    DUAS_PORCENTAGENS_FECHO,
    DUAS_PORCENTAGENS_INTRO,
    DUAS_PORCENTAGENS_TITULO,
    ONDE_NO_CODIGO,
    SECOES,
)
from utils.ui import ScrollableFrame, quebra_automatica
from views.info_window import InfoWindow


class AboutView(ctk.CTkFrame):
    """
    Frame responsável por exibir informações acadêmicas e técnicas do sistema.

    Apresenta dados sobre o TCC, a base de dados Wisconsin (WDBC),
    os modelos de IA utilizados e as tecnologias de explicabilidade (SHAP e UMAP).
    """

    def __init__(self, master, **kwargs):
        """
        Inicializa o frame e configura o layout de grade.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Janela principal que contém este frame.
        **kwargs
            Argumentos adicionais passados para o construtor do CTkFrame.
        """
        super().__init__(master, corner_radius=0, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._info_window = None
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói os cards de informação dentro de um frame rolável.
        """
        scroll = ScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)
        scroll.grid_columnconfigure((0, 1), weight=1)

        # Cabeçalho
        ctk.CTkLabel(
            scroll, text="Sobre o Projeto",
            font=ctk.CTkFont(size=28, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ctk.CTkLabel(
            scroll,
            text="O que é este trabalho, de onde vêm os dados e como o programa chega às suas contas",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 24))

        # Card: TCC
        self._build_tcc_card(scroll, row=2)

        # Cards de duas colunas: Dataset e Modelos
        self._build_dataset_card(scroll, row=3, col=0)
        self._build_models_card(scroll, row=3, col=1)

        # Card: Tecnologias XAI
        self._build_xai_card(scroll, row=4)

        # Card: Memorial de cálculo (as duas porcentagens + todas as contas)
        self._build_calculos_card(scroll, row=5)

        # Card: Glossário de Biomarcadores
        self._build_glossary_card(scroll, row=6)

    def _build_tcc_card(self, parent, row: int):
        """
        Constrói o card com informações acadêmicas do TCC.

        Parameters
        ----------
        parent : ctk.CTkBaseClass
            Frame pai onde o card será inserido.
        row : int
            Linha da grade onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(
            card, text="Trabalho de Conclusão de Curso",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=(16, 10))

        infos = [
            ("Título",
             "Desenvolvimento de um Software com Inteligência Artificial Explicável (XAI)\n"
             "para Apoio ao Diagnóstico Preditivo de Câncer de Mama utilizando SHAP e UMAP"),
            ("Autor",       "Gabriel Ast dos Santos"),
            ("Instituição", "Universidade Tuiuti do Paraná (UTP)"),
            ("Curso",       "Bacharelado em Ciência da Computação"),
            ("Orientador",  "Prof. Rodrigo Ramos Alves"),
            ("Ano",         "2026"),
        ]
        for label, value in infos:
            row_frame = ctk.CTkFrame(card, fg_color="transparent")
            row_frame.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(
                row_frame, text=f"{label}:",
                font=ctk.CTkFont(size=13, weight="bold"),
                width=110, anchor="w"
            ).pack(side="left")
            ctk.CTkLabel(
                row_frame, text=value,
                font=ctk.CTkFont(size=13),
                anchor="w", justify="left"
            ).pack(side="left", padx=(8, 0))

        ctk.CTkFrame(card, height=16, fg_color="transparent").pack()

    def _build_dataset_card(self, parent, row: int, col: int):
        """
        Constrói o card com informações sobre a base de dados Wisconsin.

        Parameters
        ----------
        parent : ctk.CTkBaseClass
            Frame pai onde o card será inserido.
        row : int
            Linha da grade onde o card será posicionado.
        col : int
            Coluna da grade onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=col, sticky="nsew", pady=(0, 16), padx=(0, 8))

        ctk.CTkLabel(
            card, text="Base de Dados",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=(16, 10))

        infos = [
            ("Nome",          "Wisconsin Diagnostic Breast Cancer (WDBC)"),
            ("Origem",        "UCI Machine Learning Repository"),
            ("Amostras",      "569 pacientes"),
            ("Classes",       "Maligno (212)  ·  Benigno (357)"),
            ("Atributos",     "30 biomarcadores morfológicos"),
            ("Extração",      "Punção Aspirativa por Agulha Fina (PAAF)"),
            ("Padronização",  "Z-Score  (μ = 0, σ = 1)"),
        ]
        for label, value in infos:
            row_frame = ctk.CTkFrame(card, fg_color="transparent")
            row_frame.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(
                row_frame, text=f"{label}:",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=100, anchor="w"
            ).pack(side="left")
            ctk.CTkLabel(
                row_frame, text=value,
                font=ctk.CTkFont(size=12),
                anchor="w", justify="left", wraplength=260
            ).pack(side="left", padx=(8, 0))

        ctk.CTkFrame(card, height=16, fg_color="transparent").pack()

    def _build_models_card(self, parent, row: int, col: int):
        """
        Constrói o card com os modelos de IA disponíveis no sistema.

        Parameters
        ----------
        parent : ctk.CTkBaseClass
            Frame pai onde o card será inserido.
        row : int
            Linha da grade onde o card será posicionado.
        col : int
            Coluna da grade onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=col, sticky="nsew", pady=(0, 16), padx=(8, 0))

        ctk.CTkLabel(
            card, text="Modelos de IA",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=(16, 10))

        modelos = [
            ("rf",  "Random Forest (RF)",            "Ensemble de árvores de decisão com bagging"),
            ("svm", "Support Vector Machine (SVM)",  "Hiperplano de separação com kernel RBF"),
            ("lr",  "Regressão Logística (LR)",       "Modelo linear probabilístico"),
            ("knn", "K-Nearest Neighbors (KNN)",      "Classificação por proximidade entre amostras"),
            ("dt",  "Árvore de Decisão (DT)",         "Regras de decisão binárias interpretáveis"),
        ]
        for chave, nome, desc in modelos:
            item = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
            item.pack(fill="x", padx=20, pady=4)
            ctk.CTkLabel(
                item, text=nome,
                font=ctk.CTkFont(size=12, weight="bold"), anchor="w"
            ).pack(anchor="w", padx=12, pady=(8, 2))
            ctk.CTkLabel(
                item, text=desc,
                font=ctk.CTkFont(size=11), text_color="gray", anchor="w"
            ).pack(anchor="w", padx=12, pady=(0, 2))
            ctk.CTkLabel(
                item, text="Ver detalhes  ›",
                font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498db", anchor="w"
            ).pack(anchor="w", padx=12, pady=(0, 8))
            self._tornar_clicavel(item, chave)

        ctk.CTkFrame(card, height=16, fg_color="transparent").pack()

    def _build_xai_card(self, parent, row: int):
        """
        Constrói o card com as tecnologias de explicabilidade SHAP e UMAP.

        Parameters
        ----------
        parent : ctk.CTkBaseClass
            Frame pai onde o card será inserido.
        row : int
            Linha da grade onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(
            card, text="Tecnologias de Explicabilidade (XAI)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=(16, 10))

        tecnologias = [
            (
                "shap",
                "SHAP",
                "SHapley Additive exPlanations",
                "Mostra o quanto cada biomarcador puxou a decisão para um lado ou para o outro, "
                "paciente por paciente. A conta reparte a previsão entre os atributos usando os "
                "valores de Shapley, uma ideia emprestada da teoria dos jogos. Serve para "
                "responder: por que este paciente recebeu este diagnóstico?",
            ),
            (
                "umap",
                "UMAP",
                "Uniform Manifold Approximation and Projection",
                "Espreme os 30 atributos em apenas 2, para eles caberem num gráfico. Serve para "
                "ver de relance o quanto os tumores benignos e os malignos formam grupos "
                "separados, e onde cada paciente novo cai em relação a esses grupos.",
            ),
        ]
        for chave, sigla, nome_completo, descricao in tecnologias:
            item = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
            item.pack(fill="x", padx=20, pady=4)
            header = ctk.CTkFrame(item, fg_color="transparent")
            header.pack(fill="x", padx=12, pady=(10, 4))
            ctk.CTkLabel(
                header, text=sigla,
                font=ctk.CTkFont(size=13, weight="bold")
            ).pack(side="left")
            ctk.CTkLabel(
                header, text=f"  ·  {nome_completo}",
                font=ctk.CTkFont(size=12), text_color="gray"
            ).pack(side="left")
            quebra_automatica(ctk.CTkLabel(
                item, text=descricao,
                font=ctk.CTkFont(size=12), text_color="gray",
                anchor="w", justify="left", wraplength=820
            )).pack(anchor="w", fill="x", padx=12, pady=(0, 4))
            ctk.CTkLabel(
                item, text="Ver detalhes  ›",
                font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498db", anchor="w"
            ).pack(anchor="w", padx=12, pady=(0, 10))
            self._tornar_clicavel(item, chave)

        ctk.CTkFrame(card, height=16, fg_color="transparent").pack()

    def _build_calculos_card(self, parent, row: int):
        """
        Constrói o card que documenta todos os cálculos do programa.

        Abre pela distinção entre as duas porcentagens da interface (a certeza
        do paciente e os cortes da régua), que é a dúvida que a tela provoca com
        mais frequência, e segue com as seções na ordem dos Passos 1 a 5. O
        conteúdo inteiro vem de ``core.calculos`` — aqui só se desenha.

        Parameters
        ----------
        parent : ctk.CTkBaseClass
            Frame pai onde o card será inserido.
        row : int
            Linha da grade onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(
            card, text="Como os números são calculados",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=(16, 0))
        ctk.CTkLabel(
            card, text="Memorial de cálculo: de onde vem cada valor exibido pelo sistema",
            font=ctk.CTkFont(size=12), text_color="gray"
        ).pack(anchor="w", padx=20, pady=(0, 10))

        # self._build_duas_porcentagens(card)

        for secao in SECOES:
            self._build_secao_calculo(card, secao)

        # Rodapé: o endereço de cada conta no código, para quem for auditar.
        rodape = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
        rodape.pack(fill="x", padx=20, pady=(14, 18))
        ctk.CTkLabel(
            rodape, text="ⓘ  Onde cada cálculo vive no código",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498db", anchor="w",
        ).pack(anchor="w", padx=12, pady=(10, 2))
        quebra_automatica(ctk.CTkLabel(
            rodape, text=ONDE_NO_CODIGO, font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray85"), anchor="w", justify="left", wraplength=880,
        )).pack(anchor="w", fill="x", padx=12, pady=(0, 10))

    def _build_duas_porcentagens(self, card):
        """
        Desenha o bloco de abertura: as duas porcentagens, lado a lado.

        As duas colunas existem para a comparação ser visual — uma porcentagem
        que descreve o paciente ao lado de outra que descreve a regra. Ler as
        duas como se fossem a mesma coisa é o que faz alguém comparar uma
        certeza de 18% com 50% e concluir o oposto do que o sistema decidiu.

        Parameters
        ----------
        card : ctk.CTkFrame
            Card onde o bloco será inserido.
        """
        ctk.CTkLabel(
            card, text=DUAS_PORCENTAGENS_TITULO,
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=20, pady=(2, 4))
        quebra_automatica(ctk.CTkLabel(
            card, text=DUAS_PORCENTAGENS_INTRO, font=ctk.CTkFont(size=12),
            anchor="w", justify="left", wraplength=900,
        )).pack(anchor="w", fill="x", padx=20, pady=(0, 10))

        colunas = ctk.CTkFrame(card, fg_color="transparent")
        colunas.pack(fill="x", padx=20, pady=(0, 10))
        colunas.grid_columnconfigure((0, 1), weight=1, uniform="porcentagem")

        for coluna, (titulo, natureza, exemplo, texto) in enumerate(DUAS_PORCENTAGENS):
            bloco = ctk.CTkFrame(colunas, fg_color=("gray85", "gray20"), corner_radius=6)
            bloco.grid(row=0, column=coluna, sticky="nsew",
                       padx=(0, 6) if coluna == 0 else (6, 0))
            ctk.CTkLabel(
                bloco, text=titulo, font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
            ).pack(anchor="w", padx=12, pady=(10, 0))
            ctk.CTkLabel(
                bloco, text=natureza, font=ctk.CTkFont(size=11),
                text_color="gray", anchor="w",
            ).pack(anchor="w", padx=12, pady=(0, 6))
            ctk.CTkLabel(
                bloco, text=exemplo, font=ctk.CTkFont(size=17, weight="bold"),
                text_color="#3498db", anchor="w",
            ).pack(anchor="w", padx=12, pady=(0, 6))
            quebra_automatica(ctk.CTkLabel(
                bloco, text=texto, font=ctk.CTkFont(size=12),
                text_color=("gray20", "gray85"), anchor="w", justify="left", wraplength=420,
            ), margem=24, minimo=200).pack(anchor="w", fill="x", padx=12, pady=(0, 12))

        fecho = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
        fecho.pack(fill="x", padx=20, pady=(0, 4))
        quebra_automatica(ctk.CTkLabel(
            fecho, text=DUAS_PORCENTAGENS_FECHO, font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray85"), anchor="w", justify="left", wraplength=880,
        )).pack(anchor="w", fill="x", padx=12, pady=10)

    def _build_secao_calculo(self, card, secao: dict):
        """
        Desenha uma seção do memorial: título, resumo e a tabela de itens.

        Parameters
        ----------
        card : ctk.CTkFrame
            Card onde a seção será inserida.
        secao : dict
            Item de ``core.calculos.SECOES`` — {'titulo', 'resumo', 'itens'}.
        """
        ctk.CTkLabel(
            card, text=secao['titulo'], font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=20, pady=(16, 4))
        quebra_automatica(ctk.CTkLabel(
            card, text=secao['resumo'], font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"), anchor="w", justify="left", wraplength=900,
        )).pack(anchor="w", fill="x", padx=20, pady=(0, 8))

        # Três colunas: o que é, a fórmula literal e o porquê. A do meio usa
        # fonte monoespaçada — uma fórmula em fonte proporcional perde o
        # alinhamento dos índices e fica ilegível.
        tabela = ctk.CTkFrame(card, fg_color="transparent")
        tabela.pack(fill="x", padx=20, pady=(0, 4))
        tabela.grid_columnconfigure(0, weight=0, minsize=170)
        tabela.grid_columnconfigure(1, weight=0, minsize=250)
        tabela.grid_columnconfigure(2, weight=1)

        for i, (nome, formula, explicacao) in enumerate(secao['itens']):
            bg = ("gray92", "gray17") if i % 2 else ("gray87", "gray21")
            ctk.CTkLabel(
                tabela, text=nome, font=ctk.CTkFont(size=12, weight="bold"),
                anchor="nw", justify="left", wraplength=160, fg_color=bg,
            ).grid(row=i, column=0, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)
            ctk.CTkLabel(
                tabela, text=formula or "—",
                font=ctk.CTkFont(family="monospace", size=12),
                text_color=("#0b6b3a", "#7ee2a8") if formula else ("gray50", "gray45"),
                anchor="nw", justify="left", wraplength=240, fg_color=bg,
            ).grid(row=i, column=1, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)
            quebra_automatica(ctk.CTkLabel(
                tabela, text=explicacao, font=ctk.CTkFont(size=12),
                text_color=("gray20", "gray80"),
                anchor="nw", justify="left", wraplength=440, fg_color=bg,
            ), margem=20, minimo=220).grid(
                row=i, column=2, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)

    def _build_glossary_card(self, parent, row: int):
        """
        Constrói o card com o glossário dos 30 biomarcadores (inline, sem popup).

        Apresenta a introdução, uma nota destacada sobre unidades de medida, o
        significado das 3 estatísticas (média/erro padrão/pior) e uma tabela
        alinhada com as 10 medições-base. O conteúdo vem de ``core.biomarkers``,
        a mesma fonte usada pelos tooltips da tabela de resultados.

        Parameters
        ----------
        parent : ctk.CTkBaseClass
            Frame pai onde o card será inserido.
        row : int
            Linha da grade onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(
            card, text="Glossário de Biomarcadores",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=20, pady=(16, 0))
        ctk.CTkLabel(
            card, text=GLOSSARIO_SUBTITULO,
            font=ctk.CTkFont(size=12), text_color="gray"
        ).pack(anchor="w", padx=20, pady=(0, 8))

        # Introdução
        quebra_automatica(ctk.CTkLabel(
            card, text=GLOSSARIO_INTRO, font=ctk.CTkFont(size=12),
            anchor="w", justify="left", wraplength=900,
        )).pack(anchor="w", fill="x", padx=20, pady=(0, 10))

        # Nota destacada sobre unidades de medida
        nota = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
        nota.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(
            nota, text="ⓘ  Unidades de medida",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498db", anchor="w",
        ).pack(anchor="w", padx=12, pady=(10, 2))
        quebra_automatica(ctk.CTkLabel(
            nota, text=GLOSSARIO_UNIDADES, font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray85"), anchor="w", justify="left", wraplength=880,
        )).pack(anchor="w", fill="x", padx=12, pady=(0, 10))

        # As 3 estatísticas (sufixos _mean/_se/_worst)
        ctk.CTkLabel(
            card, text="Como cada biomarcador é resumido (3 estatísticas por imagem)",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=20, pady=(2, 4))
        for sufixo, (rotulo, expl) in ESTATISTICAS.items():
            linha = ctk.CTkFrame(card, fg_color="transparent")
            linha.pack(fill="x", padx=20, pady=1)
            ctk.CTkLabel(
                linha, text=f"{rotulo}  (_{sufixo})",
                font=ctk.CTkFont(size=12, weight="bold"), width=150, anchor="w",
            ).pack(side="left")
            quebra_automatica(ctk.CTkLabel(
                linha, text=expl, font=ctk.CTkFont(size=12),
                text_color=("gray25", "gray80"), anchor="w", justify="left", wraplength=730,
            ), margem=10, minimo=220).pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Tabela alinhada dos 10 biomarcadores-base
        ctk.CTkLabel(
            card, text="Os 10 biomarcadores-base",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=20, pady=(12, 4))

        # A coluna do meio é a que cede espaço quando a janela estreita; as
        # laterais têm largura mínima só para o rótulo e a unidade não
        # quebrarem em todas as linhas.
        tabela = ctk.CTkFrame(card, fg_color="transparent")
        tabela.pack(fill="x", padx=20, pady=(0, 18))
        tabela.grid_columnconfigure(0, weight=0, minsize=130)
        tabela.grid_columnconfigure(1, weight=1)
        tabela.grid_columnconfigure(2, weight=0, minsize=150)

        cabecalhos = ("Biomarcador", "O que mede", "Unidade")
        for c, titulo in enumerate(cabecalhos):
            ctk.CTkLabel(
                tabela, text=titulo, font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w", fg_color=("gray75", "gray28"),
            ).grid(row=0, column=c, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)

        for i, (base, (rotulo, mede, unidade)) in enumerate(BIOMARCADORES_BASE.items(), start=1):
            bg = ("gray92", "gray17") if i % 2 else ("gray87", "gray21")
            ctk.CTkLabel(
                tabela, text=rotulo, font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w", fg_color=bg,
            ).grid(row=i, column=0, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)
            quebra_automatica(ctk.CTkLabel(
                tabela, text=mede[0].upper() + mede[1:], font=ctk.CTkFont(size=12),
                anchor="w", justify="left", wraplength=430, fg_color=bg,
            ), margem=20, minimo=180).grid(
                row=i, column=1, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)
            quebra_automatica(ctk.CTkLabel(
                tabela, text=unidade, font=ctk.CTkFont(size=12), text_color=("gray30", "gray70"),
                anchor="w", justify="left", wraplength=170, fg_color=bg,
            ), margem=20, minimo=120).grid(
                row=i, column=2, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)

    def _tornar_clicavel(self, item, chave: str):
        """
        Torna um item (e todos os seus filhos) clicável, abrindo os detalhes.

        Aplica também um leve realce ao passar o mouse, sinalizando que o item
        é interativo.

        Parameters
        ----------
        item : ctk.CTkFrame
            Frame do item a tornar clicável.
        chave : str
            Identificador do conteúdo a exibir (índice em CONTEUDO).
        """
        cor_normal = ("gray85", "gray20")
        cor_hover = ("gray80", "gray28")

        def clicar(_evento, k=chave):
            self._abrir_info(k)

        def entrar(_evento):
            item.configure(fg_color=cor_hover)

        def sair(_evento):
            item.configure(fg_color=cor_normal)

        def aplicar(widget):
            widget.bind("<Button-1>", clicar)
            widget.bind("<Enter>", entrar)
            widget.bind("<Leave>", sair)
            for filho in widget.winfo_children():
                aplicar(filho)

        aplicar(item)

    def _abrir_info(self, chave: str):
        """
        Abre (ou recria) a janela de detalhes do modelo ou tecnologia.

        Parameters
        ----------
        chave : str
            Identificador do conteúdo em CONTEUDO.
        """
        if self._info_window is not None and self._info_window.winfo_exists():
            self._info_window.destroy()
        self._info_window = InfoWindow(self, chave)
