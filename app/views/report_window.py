"""
Módulo contendo a janela de relatório de explicabilidade (XAI).

Exibe, para a Árvore de Decisão, as características mais decisivas do modelo
e o detalhamento do raciocínio aplicado a cada paciente do lote.
"""

import customtkinter as ctk
from tkinter import ttk

from utils.ui import bind_treeview_mousewheel, responsive_geometry
from views.report_common import PatientPDFExportMixin


class ReportWindow(ctk.CTkToplevel, PatientPDFExportMixin):
    """
    Janela secundária que apresenta o relatório de explicabilidade da árvore.

    Estrutura-se em três blocos: cabeçalho com o resumo do lote, ranking
    global das características mais decisivas e uma área mestre-detalhe que
    expõe, por paciente, o caminho de regras e os fatores que pesaram na decisão.

    Attributes
    ----------
    _explicacoes : list[dict]
        Lista de explicações por paciente gerada pelo DecisionTreeExplainer.
    _por_indice : dict
        Mapa do índice do paciente (str) para a respectiva explicação.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"

    def __init__(self, master, importancias: list, explicacoes: list, **kwargs):
        """
        Inicializa a janela de relatório.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget pai que originou o relatório.
        importancias : list[tuple[str, float]]
            Ranking global de importância das características.
        explicacoes : list[dict]
            Explicações por paciente produzidas pelo DecisionTreeExplainer.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade — Árvore de Decisão")
        responsive_geometry(self, 980, 740)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}

        self._build_header()
        self._build_global(importancias)
        self._build_per_patient(explicacoes)

        # Garante que a janela surja à frente da principal.
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote diagnosticado."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 6))

        ctk.CTkLabel(
            header, text="Relatório de Explicabilidade",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")

        n = len(self._explicacoes)
        malignos = sum(1 for e in self._explicacoes if e['classe'] == 'Maligno')
        benignos = n - malignos
        ctk.CTkLabel(
            header,
            text=(f"Árvore de Decisão   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {benignos}"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

    def _build_global(self, importancias: list):
        """
        Constrói o painel com o ranking global de características decisivas.

        Parameters
        ----------
        importancias : list[tuple[str, float]]
            Pares (característica, importância) ordenados do maior para o menor.
        """
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Ganho de informação por biomarcador (redução de entropia)",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            frame,
            text="Critério de treino: entropia. O valor é a redução total de entropia "
                 "(ganho de informação) que cada atributo trouxe à árvore; a barra é a "
                 "participação relativa.",
            font=ctk.CTkFont(size=11), text_color="gray", justify="left", wraplength=880,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        if not importancias:
            ctk.CTkLabel(
                frame, text="Sem informação de importância disponível.",
                text_color="gray",
            ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 12))
            return

        # Aceita tanto (nome, participação) quanto (nome, participação, ganho_entropia).
        def _desempacota(item):
            nome, share = item[0], item[1]
            ganho = item[2] if len(item) > 2 else None
            return nome, share, ganho

        maior = _desempacota(importancias[0])[1] or 1.0
        for i, item in enumerate(importancias, start=2):
            nome, share, ganho = _desempacota(item)
            ctk.CTkLabel(
                frame, text=nome, anchor="w", font=ctk.CTkFont(size=12),
            ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=3)

            barra = ctk.CTkProgressBar(frame, height=14)
            barra.set(share / maior)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)

            texto = f"Δentropia {ganho:.3f}" if ganho is not None else f"{share * 100:.1f}%"
            ctk.CTkLabel(
                frame, text=texto, width=130, anchor="e",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkFrame(frame, height=8, fg_color="transparent").grid(
            row=len(importancias) + 2, column=0
        )

    def _build_per_patient(self, explicacoes: list):
        """
        Constrói a área mestre-detalhe com a decisão de cada paciente.

        Parameters
        ----------
        explicacoes : list[dict]
            Explicações por paciente a serem listadas e detalhadas.
        """
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 16))
        container.grid_columnconfigure(0, weight=3)
        container.grid_columnconfigure(1, weight=4)
        container.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            container, text="Decisão por paciente",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(8, 8))

        ctk.CTkButton(
            container, text="Exportar Paciente (PDF)", width=190,
            fg_color="#7f8c8d", hover_color="#95a5a6",
            command=self._exportar_pdf_paciente,
        ).grid(row=0, column=1, sticky="e", pady=(8, 8))

        self._style_tree()

        tree_frame = ctk.CTkFrame(container)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        colunas = ("paciente", "diagnostico", "fatores")
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings")
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.heading("fatores", text="Principais fatores")
        self._tree.column("paciente", width=70, anchor="center", stretch=False)
        self._tree.column("diagnostico", width=90, anchor="center", stretch=False)
        self._tree.column("fatores", width=300, anchor="w")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(tree_frame, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        bind_treeview_mousewheel(self._tree)

        self._tree.tag_configure("Maligno", foreground=self.COR_MALIGNO)
        self._tree.tag_configure("Benigno", foreground=self.COR_BENIGNO)

        for e in explicacoes:
            principais = ", ".join(c['feature'] for c in e['contribuicoes'][:2]) or "—"
            self._tree.insert(
                "", "end", iid=str(e['indice']),
                values=(e['indice'], e['classe'], principais),
                tags=(e['classe'],),
            )
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self._detalhe = ctk.CTkTextbox(
            container, wrap="word",
            font=ctk.CTkFont(family="Courier New", size=13),
        )
        self._detalhe.grid(row=1, column=1, sticky="nsew")
        self._detalhe.insert(
            "1.0",
            "Selecione um paciente na lista ao lado para ver o raciocínio "
            "completo da Árvore de Decisão.",
        )
        self._detalhe.configure(state="disabled")

        if explicacoes:
            primeiro = str(explicacoes[0]['indice'])
            self._tree.selection_set(primeiro)
            self._tree.focus(primeiro)

    def _on_select(self, _event=None):
        """Atualiza o painel de detalhe conforme o paciente selecionado."""
        selecao = self._tree.selection()
        if not selecao:
            return
        explicacao = self._por_indice.get(selecao[0])
        if not explicacao:
            return
        self._detalhe.configure(state="normal")
        self._detalhe.delete("1.0", "end")
        self._detalhe.insert("1.0", self._formatar_detalhe(explicacao))
        self._detalhe.configure(state="disabled")

    def _formatar_detalhe(self, e: dict) -> str:
        """
        Monta o texto explicativo completo da decisão de um paciente.

        Parameters
        ----------
        e : dict
            Explicação individual produzida pelo DecisionTreeExplainer.

        Returns
        -------
        str
            Texto formatado com diagnóstico, fatores decisivos e caminho de regras.
        """
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Diagnóstico da IA: {e['classe']}",
            "",
            "Características que mais pesaram nesta decisão:",
        ]
        for c in e['contribuicoes'][:6]:
            seta = "↑ Maligno" if c['direcao'] == 'Maligno' else "↓ Benigno"
            linhas.append(
                f"   • {c['feature']} = {c['valor']:.3f}"
                f"   ({seta}, impacto {abs(c['contribuicao']) * 100:.1f}%)"
            )

        linhas.append("")
        linhas.append("Caminho de decisão (regras aplicadas, da raiz à folha):")
        if e['caminho']:
            for i, regra in enumerate(e['caminho'], start=1):
                linhas.append(f"   {i}. {regra}")
        else:
            linhas.append("   (a árvore classificou diretamente na raiz)")

        return "\n".join(linhas)

    def _style_tree(self):
        """Aplica o tema escuro ao componente Treeview da lista de pacientes."""
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview", background="#2b2b2b", foreground="white",
            fieldbackground="#2b2b2b", borderwidth=0, rowheight=26,
        )
        style.map("Treeview", background=[("selected", "#1f538d")])
        style.configure(
            "Treeview.Heading", background="#1f538d",
            foreground="white", relief="flat",
        )
        style.map("Treeview.Heading", background=[("active", "#14375e")])