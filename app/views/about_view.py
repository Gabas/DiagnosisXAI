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
from utils.ui import ScrollableFrame
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
            text="Informações sobre o trabalho acadêmico, a base de dados e as tecnologias empregadas",
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

        # Card: Glossário de Biomarcadores
        self._build_glossary_card(scroll, row=5)

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
                "Atribui a cada característica morfológica uma contribuição marginal para a predição, "
                "fundamentada na teoria dos jogos cooperativos (Shapley values). Permite identificar "
                "quais biomarcadores mais influenciaram o diagnóstico de cada paciente.",
            ),
            (
                "umap",
                "UMAP",
                "Uniform Manifold Approximation and Projection",
                "Reduz a dimensionalidade dos 30 atributos para 2D, permitindo visualizar a "
                "separabilidade geométrica entre tumores benignos e malignos no espaço de características.",
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
                header, text=f"  —  {nome_completo}",
                font=ctk.CTkFont(size=12), text_color="gray"
            ).pack(side="left")
            ctk.CTkLabel(
                item, text=descricao,
                font=ctk.CTkFont(size=12), text_color="gray",
                anchor="w", justify="left", wraplength=820
            ).pack(anchor="w", padx=12, pady=(0, 4))
            ctk.CTkLabel(
                item, text="Ver detalhes  ›",
                font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498db", anchor="w"
            ).pack(anchor="w", padx=12, pady=(0, 10))
            self._tornar_clicavel(item, chave)

        ctk.CTkFrame(card, height=16, fg_color="transparent").pack()

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
        ctk.CTkLabel(
            card, text=GLOSSARIO_INTRO, font=ctk.CTkFont(size=12),
            anchor="w", justify="left", wraplength=900,
        ).pack(anchor="w", fill="x", padx=20, pady=(0, 10))

        # Nota destacada sobre unidades de medida
        nota = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
        nota.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(
            nota, text="ⓘ  Unidades de medida",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498db", anchor="w",
        ).pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(
            nota, text=GLOSSARIO_UNIDADES, font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray85"), anchor="w", justify="left", wraplength=880,
        ).pack(anchor="w", fill="x", padx=12, pady=(0, 10))

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
            ctk.CTkLabel(
                linha, text=expl, font=ctk.CTkFont(size=12),
                text_color=("gray25", "gray80"), anchor="w", justify="left", wraplength=730,
            ).pack(side="left", padx=(8, 0))

        # Tabela alinhada dos 10 biomarcadores-base
        ctk.CTkLabel(
            card, text="Os 10 biomarcadores-base",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=20, pady=(12, 4))

        tabela = ctk.CTkFrame(card, fg_color="transparent")
        tabela.pack(fill="x", padx=20, pady=(0, 18))
        tabela.grid_columnconfigure(0, weight=0, minsize=140)
        tabela.grid_columnconfigure(1, weight=1)
        tabela.grid_columnconfigure(2, weight=0, minsize=180)

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
            ctk.CTkLabel(
                tabela, text=mede[0].upper() + mede[1:], font=ctk.CTkFont(size=12),
                anchor="w", justify="left", wraplength=430, fg_color=bg,
            ).grid(row=i, column=1, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)
            ctk.CTkLabel(
                tabela, text=unidade, font=ctk.CTkFont(size=12), text_color=("gray30", "gray70"),
                anchor="w", justify="left", wraplength=170, fg_color=bg,
            ).grid(row=i, column=2, sticky="nsew", padx=1, pady=1, ipadx=8, ipady=6)

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
