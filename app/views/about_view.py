"""
Módulo contendo a aba de informações sobre o projeto e a base de dados.
"""

import customtkinter as ctk
from utils.ui import ScrollableFrame


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
            ("Random Forest (RF)",            "Ensemble de árvores de decisão com bagging"),
            ("Support Vector Machine (SVM)",  "Hiperplano de separação com kernel RBF"),
            ("Regressão Logística (LR)",       "Modelo linear probabilístico"),
            ("K-Nearest Neighbors (KNN)",      "Classificação por proximidade entre amostras"),
            ("Árvore de Decisão (DT)",         "Regras de decisão binárias interpretáveis"),
        ]
        for nome, desc in modelos:
            item = ctk.CTkFrame(card, fg_color=("gray85", "gray20"), corner_radius=6)
            item.pack(fill="x", padx=20, pady=4)
            ctk.CTkLabel(
                item, text=nome,
                font=ctk.CTkFont(size=12, weight="bold"), anchor="w"
            ).pack(anchor="w", padx=12, pady=(8, 2))
            ctk.CTkLabel(
                item, text=desc,
                font=ctk.CTkFont(size=11), text_color="gray", anchor="w"
            ).pack(anchor="w", padx=12, pady=(0, 8))

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
                "SHAP",
                "SHapley Additive exPlanations",
                "Atribui a cada característica morfológica uma contribuição marginal para a predição, "
                "fundamentada na teoria dos jogos cooperativos (Shapley values). Permite identificar "
                "quais biomarcadores mais influenciaram o diagnóstico de cada paciente.",
            ),
            (
                "UMAP",
                "Uniform Manifold Approximation and Projection",
                "Reduz a dimensionalidade dos 30 atributos para 2D, permitindo visualizar a "
                "separabilidade geométrica entre tumores benignos e malignos no espaço de características.",
            ),
        ]
        for sigla, nome_completo, descricao in tecnologias:
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
            ).pack(anchor="w", padx=12, pady=(0, 10))

        ctk.CTkFrame(card, height=16, fg_color="transparent").pack()
