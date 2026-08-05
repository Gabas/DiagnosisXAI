"""
Módulo contendo a janela de relatório de explicabilidade do Random Forest.

Apresenta a importância global (Gini) e, por paciente, o consenso da floresta:
quantas das N árvores votaram em cada classe e a distribuição das probabilidades
que cada árvore atribuiu — mostrando se a decisão é um consenso forte ou uma
maioria apertada. Complementa o relatório SHAP do RF (que detalha atributos).
"""

import customtkinter as ctk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import (ScrollableFrame, ajustar_ao_conteudo, bind_treeview_mousewheel,
                      figura_responsiva, itens_visiveis, responsive_geometry)
from views.report_common import PatientPDFExportMixin


class RandomForestReportWindow(ctk.CTkToplevel, PatientPDFExportMixin):
    """
    Janela secundária com o relatório de explicabilidade do Random Forest.

    Reúne o ranking de biomarcadores (importância Gini), o consenso das árvores
    por paciente (voto duro + histograma das probabilidades) e a área
    mestre-detalhe. Selecionar um paciente atualiza o histograma de consenso.

    Attributes
    ----------
    _explicacoes : list[dict]
        Explicações por paciente geradas pelo RandomForestExplainer.
    _n_arvores : int
        Número de árvores da floresta.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_FUNDO = "#2b2b2b"

    def __init__(self, master, importancias: list, explicacoes: list,
                 contexto: dict, **kwargs):
        """
        Inicializa a janela de relatório do Random Forest.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o relatório.
        importancias : list[tuple[str, float]]
            Ranking de importância global (Gini).
        explicacoes : list[dict]
            Explicações por paciente produzidas pelo RandomForestExplainer.
        contexto : dict
            {'n_arvores', 'n_bins'} — metadados da floresta.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade — Random Forest")
        responsive_geometry(self, 1060, 840)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Todo o conteúdo vive num corpo rolável: o layout (cabeçalho + painéis
        # + mestre-detalhe) pede mais altura do que cabe num notebook, e sem
        # rolagem a parte de baixo ficava inacessível, não apenas apertada.
        self._corpo = ScrollableFrame(self, fg_color="transparent")
        self._corpo.grid(row=0, column=0, sticky="nsew")
        self._corpo.grid_columnconfigure(0, weight=1)
        self._corpo.grid_columnconfigure(1, weight=1)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}
        self._n_arvores = int(contexto.get('n_arvores', 0))
        self._n_bins = int(contexto.get('n_bins', 10))
        self._linhas_lista = itens_visiveis(self, 10, minimo=6)

        self._build_header()
        self._build_global(importancias)
        self._build_plot()
        self._build_per_patient(explicacoes)

        ajustar_ao_conteudo(self, self._corpo)
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote diagnosticado."""
        header = ctk.CTkFrame(self._corpo, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 6))

        ctk.CTkLabel(
            header, text="Relatório de Explicabilidade",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")

        n = len(self._explicacoes)
        malignos = sum(1 for e in self._explicacoes if e['classe'] == 'Maligno')
        limitrofes = sum(1 for e in self._explicacoes if e['limitrofe'])
        ctk.CTkLabel(
            header,
            text=(f"Random Forest ({self._n_arvores} árvores)   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {n - malignos}   ·   "
                  f"Casos limítrofes: {limitrofes}"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

    def _build_global(self, importancias: list):
        """
        Constrói o painel de importância global (Gini).

        Parameters
        ----------
        importancias : list[tuple[str, float]]
            Pares (característica, importância) ordenados do maior para o menor.
        """
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Biomarcadores mais relevantes (Gini)",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            frame, text="Redução total de impureza atribuída a cada atributo na floresta",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        importancias = importancias[:itens_visiveis(self, 10)]
        if not importancias:
            ctk.CTkLabel(frame, text="Sem informação disponível.", text_color="gray").grid(
                row=2, column=0, sticky="w", padx=16, pady=(0, 12))
            return

        maior = max(v for _, v in importancias) or 1.0
        for i, (nome, imp) in enumerate(importancias, start=2):
            ctk.CTkLabel(
                frame, text=nome, anchor="w", font=ctk.CTkFont(size=12),
            ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=3)

            barra = ctk.CTkProgressBar(frame, height=14, progress_color="#16a085")
            barra.set(imp / maior)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)

            ctk.CTkLabel(
                frame, text=f"{imp * 100:.2f}%", width=60, anchor="e",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkFrame(frame, height=8, fg_color="transparent").grid(
            row=len(importancias) + 2, column=0)

    def _build_plot(self):
        """Constrói o painel do histograma de consenso das árvores (por paciente)."""
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Consenso das árvores (paciente selecionado)",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        fig = Figure(figsize=figura_responsiva(self, 5.0, 3.8), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(self.COR_FUNDO)
        fig.tight_layout()

        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            frame,
            text="Cada barra = nº de árvores que atribuíram aquela faixa de P(Maligno).",
            font=ctk.CTkFont(size=10), text_color="gray",
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 10))

    def _desenhar_consenso(self, e: dict):
        """
        Desenha o histograma das probabilidades das árvores para um paciente.

        Parameters
        ----------
        e : dict
            Explicação individual, contendo 'hist' (contagem por faixa),
            'probabilidade' (média da floresta) e a classe predita.
        """
        hist = e['hist']
        n_bins = len(hist) or 1
        centros = [(i + 0.5) / n_bins for i in range(n_bins)]
        largura = (1.0 / n_bins) * 0.9
        cores = [self.COR_MALIGNO if c >= 0.5 else self.COR_BENIGNO for c in centros]

        self._ax.clear()
        self._ax.bar(centros, hist, width=largura, color=cores, edgecolor="none", zorder=2)
        self._ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=3)
        p = e['probabilidade'] / 100.0
        self._ax.axvline(p, color="#f1c40f", linewidth=1.6, zorder=4,
                         label=f"média da floresta = {e['probabilidade']:.0f}%")

        self._ax.set_xlim(0, 1)
        self._ax.set_xlabel("P(Maligno) atribuída por cada árvore", color="white", fontsize=9)
        self._ax.set_ylabel("nº de árvores", color="gray", fontsize=9)
        self._ax.tick_params(colors="gray", labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color("gray")
        self._ax.legend(facecolor=self.COR_FUNDO, edgecolor="gray",
                        labelcolor="white", fontsize=8, loc="upper center")
        self._canvas.draw_idle()

    def _build_per_patient(self, explicacoes: list):
        """
        Constrói a área mestre-detalhe com a decisão de cada paciente.

        Parameters
        ----------
        explicacoes : list[dict]
            Explicações por paciente a serem listadas e detalhadas.
        """
        container = ctk.CTkFrame(self._corpo, fg_color="transparent")
        container.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 16))
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

        colunas = ("paciente", "diagnostico", "confianca", "voto")
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings",
                                  height=self._linhas_lista)
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.heading("confianca", text="Confiança")
        self._tree.heading("voto", text="Voto (M/B)")
        self._tree.column("paciente", width=70, anchor="center", stretch=False)
        self._tree.column("diagnostico", width=90, anchor="center", stretch=False)
        self._tree.column("confianca", width=80, anchor="center", stretch=False)
        self._tree.column("voto", width=110, anchor="center")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(tree_frame, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        bind_treeview_mousewheel(self._tree)

        self._tree.tag_configure("Maligno", foreground=self.COR_MALIGNO)
        self._tree.tag_configure("Benigno", foreground=self.COR_BENIGNO)

        for e in explicacoes:
            voto = f"{e['votos_maligno']} / {e['votos_benigno']}"
            if e['limitrofe']:
                voto = "⚠ " + voto
            self._tree.insert(
                "", "end", iid=str(e['indice']),
                values=(e['indice'], e['classe'], f"{e['confianca']:.0f}%", voto),
                tags=(e['classe'],),
            )
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self._detalhe = ctk.CTkTextbox(
            container, wrap="word", height=self._linhas_lista * 26,
            font=ctk.CTkFont(family="Courier New", size=13),
        )
        self._detalhe.grid(row=1, column=1, sticky="nsew")
        self._detalhe.insert(
            "1.0",
            "Selecione um paciente na lista para ver o consenso das árvores. "
            "O histograma ao lado mostra como as árvores se distribuíram.",
        )
        self._detalhe.configure(state="disabled")

        if explicacoes:
            primeiro = str(explicacoes[0]['indice'])
            self._tree.selection_set(primeiro)
            self._tree.focus(primeiro)

    def _on_select(self, _event=None):
        """Atualiza o detalhe textual e o histograma de consenso do paciente."""
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

        self._desenhar_consenso(explicacao)

    def _formatar_detalhe(self, e: dict) -> str:
        """
        Monta o texto explicativo da decisão de um paciente.

        Parameters
        ----------
        e : dict
            Explicação individual produzida pelo RandomForestExplainer.

        Returns
        -------
        str
            Texto com diagnóstico, probabilidade média e voto das árvores.
        """
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Diagnóstico da IA: {e['classe']}  (confiança {e['confianca']:.0f}%)",
            f"P(Maligno) média da floresta: {e['probabilidade']:.1f}%",
            "",
            f"Voto das {self._n_arvores} árvores:",
            f"   Maligno: {e['votos_maligno']}     Benigno: {e['votos_benigno']}",
        ]
        if e['limitrofe']:
            linhas.append("")
            linhas.append("⚠ Caso limítrofe: a floresta está pouco decidida.")
            linhas.append("  As árvores se dividem perto do limiar — recomenda-se")
            linhas.append("  revisão clínica deste paciente.")
        else:
            linhas.append("")
            linhas.append("Consenso: a maioria das árvores concorda com o diagnóstico.")

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
