"""
Módulo contendo a janela de relatório de explicabilidade da Regressão Logística.

Apresenta as características de maior peso, um gráfico da fronteira de decisão
(onde cada paciente é posicionado em relação ao limiar de malignidade) e o
detalhamento, por paciente, das características que mais influenciaram a decisão.
"""

import customtkinter as ctk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import bind_treeview_mousewheel


class LogisticReportWindow(ctk.CTkToplevel):
    """
    Janela secundária com o relatório de explicabilidade da Regressão Logística.

    Reúne o ranking de características (com a direção de cada uma), o gráfico da
    fronteira de decisão e a área mestre-detalhe por paciente. Selecionar um
    paciente na lista destaca o seu ponto no gráfico.

    Attributes
    ----------
    _explicacoes : list[dict]
        Explicações por paciente geradas pelo LogisticRegressionExplainer.
    _coords : dict
        Mapa do índice do paciente (str) para a coordenada (x, y) no gráfico.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_FUNDO = "#2b2b2b"

    def __init__(self, master, importancias: list, explicacoes: list, **kwargs):
        """
        Inicializa a janela de relatório da Regressão Logística.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o relatório.
        importancias : list[dict]
            Ranking de características com coeficiente e direção.
        explicacoes : list[dict]
            Explicações por paciente produzidas pelo explicador.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade — Regressão Logística")
        self.geometry("1060x840")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}
        self._coords = {str(e['indice']): (e['x_plot'], e['y_plot']) for e in explicacoes}
        self._highlight = None

        self._build_header()
        self._build_global(importancias)
        self._build_plot(explicacoes)
        self._build_per_patient(explicacoes)

        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote diagnosticado."""
        header = ctk.CTkFrame(self, fg_color="transparent")
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
            text=(f"Regressão Logística   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {n - malignos}   ·   "
                  f"Casos limítrofes: {limitrofes}"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

    def _build_global(self, importancias: list):
        """
        Constrói o painel com as características de maior peso e sua direção.

        Parameters
        ----------
        importancias : list[dict]
            Itens com 'feature', 'coeficiente' e 'direcao'.
        """
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Biomarcadores de maior peso na decisão",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            frame, text="Vermelho indica malignidade · verde indica benignidade",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        if not importancias:
            ctk.CTkLabel(frame, text="Sem informação disponível.", text_color="gray").grid(
                row=2, column=0, sticky="w", padx=16, pady=(0, 12))
            return

        maior = max(abs(d['coeficiente']) for d in importancias) or 1.0
        for i, d in enumerate(importancias, start=2):
            cor = self.COR_MALIGNO if d['direcao'] == 'Maligno' else self.COR_BENIGNO
            ctk.CTkLabel(
                frame, text=d['feature'], anchor="w", font=ctk.CTkFont(size=12),
            ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=3)

            barra = ctk.CTkProgressBar(frame, height=14, progress_color=cor)
            barra.set(abs(d['coeficiente']) / maior)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)

            ctk.CTkLabel(
                frame, text=f"{d['coeficiente']:+.2f}", width=56, anchor="e",
                font=ctk.CTkFont(size=12), text_color=cor,
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkFrame(frame, height=8, fg_color="transparent").grid(
            row=len(importancias) + 2, column=0)

    def _build_plot(self, explicacoes: list):
        """
        Constrói o gráfico da fronteira de decisão da Regressão Logística.

        O eixo X é a distância (com sinal) até a fronteira real; a linha
        tracejada em x = 0 é o limiar de malignidade. O eixo Y apenas espalha
        os pacientes e não afeta o diagnóstico.

        Parameters
        ----------
        explicacoes : list[dict]
            Explicações por paciente, com coordenadas 'x_plot'/'y_plot'.
        """
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Fronteira de decisão",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        xs = [e['x_plot'] for e in explicacoes]
        ys = [e['y_plot'] for e in explicacoes]
        cores = [self.COR_MALIGNO if e['classe'] == 'Maligno' else self.COR_BENIGNO
                 for e in explicacoes]

        fig = Figure(figsize=(5.0, 4.2), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COR_FUNDO)

        if xs:
            xmin, xmax = min(xs + [0.0]), max(xs + [0.0])
            pad = (xmax - xmin) * 0.1 or 1.0
            xmin, xmax = xmin - pad, xmax + pad
        else:
            xmin, xmax = -1.0, 1.0
        ax.set_xlim(xmin, xmax)

        # Regiões de decisão (apenas contexto visual; o modelo decide em x = 0).
        ax.axvspan(xmin, 0, color=self.COR_BENIGNO, alpha=0.08)
        ax.axvspan(0, xmax, color=self.COR_MALIGNO, alpha=0.08)
        ax.axvline(0, color="white", linewidth=1.5, linestyle="--")

        ax.scatter(xs, ys, c=cores, s=22, alpha=0.85, edgecolors="none", zorder=3)

        # Marcador (vazio) usado para destacar o paciente selecionado.
        self._highlight = ax.scatter(
            [], [], s=160, facecolors="none", edgecolors="#f1c40f",
            linewidths=2.2, zorder=5)

        ax.set_xlabel("←  Benigno      distância da fronteira      Maligno  →",
                      color="white", fontsize=9)
        ax.set_ylabel("variação entre pacientes", color="gray", fontsize=9)
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")

        from matplotlib.lines import Line2D
        legenda = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=self.COR_MALIGNO,
                   markersize=7, label='Maligno'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=self.COR_BENIGNO,
                   markersize=7, label='Benigno'),
        ]
        ax.legend(handles=legenda, facecolor=self.COR_FUNDO, edgecolor="gray",
                  labelcolor="white", fontsize=8, loc="best")
        fig.tight_layout()

        self._ax = ax
        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _build_per_patient(self, explicacoes: list):
        """
        Constrói a área mestre-detalhe com a decisão de cada paciente.

        Parameters
        ----------
        explicacoes : list[dict]
            Explicações por paciente a serem listadas e detalhadas.
        """
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 16))
        container.grid_columnconfigure(0, weight=3)
        container.grid_columnconfigure(1, weight=4)
        container.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            container, text="Decisão por paciente",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(8, 8))

        self._style_tree()

        tree_frame = ctk.CTkFrame(container)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        colunas = ("paciente", "diagnostico", "prob", "obs")
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings")
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.heading("prob", text="P(Maligno)")
        self._tree.heading("obs", text="Observação")
        self._tree.column("paciente", width=70, anchor="center", stretch=False)
        self._tree.column("diagnostico", width=90, anchor="center", stretch=False)
        self._tree.column("prob", width=80, anchor="center", stretch=False)
        self._tree.column("obs", width=110, anchor="center")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(tree_frame, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        bind_treeview_mousewheel(self._tree)

        self._tree.tag_configure("Maligno", foreground=self.COR_MALIGNO)
        self._tree.tag_configure("Benigno", foreground=self.COR_BENIGNO)

        for e in explicacoes:
            obs = "⚠ limítrofe" if e['limitrofe'] else ""
            self._tree.insert(
                "", "end", iid=str(e['indice']),
                values=(e['indice'], e['classe'], f"{e['probabilidade']:.1f}%", obs),
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
            "Selecione um paciente na lista para ver as características que mais "
            "pesaram na decisão. O ponto correspondente é destacado no gráfico.",
        )
        self._detalhe.configure(state="disabled")

        if explicacoes:
            primeiro = str(explicacoes[0]['indice'])
            self._tree.selection_set(primeiro)
            self._tree.focus(primeiro)

    def _on_select(self, _event=None):
        """Atualiza o detalhe e destaca o paciente selecionado no gráfico."""
        selecao = self._tree.selection()
        if not selecao:
            return
        chave = selecao[0]
        explicacao = self._por_indice.get(chave)
        if not explicacao:
            return

        self._detalhe.configure(state="normal")
        self._detalhe.delete("1.0", "end")
        self._detalhe.insert("1.0", self._formatar_detalhe(explicacao))
        self._detalhe.configure(state="disabled")

        if self._highlight is not None and chave in self._coords:
            self._highlight.set_offsets([self._coords[chave]])
            self._canvas.draw_idle()

    def _formatar_detalhe(self, e: dict) -> str:
        """
        Monta o texto explicativo completo da decisão de um paciente.

        Parameters
        ----------
        e : dict
            Explicação individual produzida pelo LogisticRegressionExplainer.

        Returns
        -------
        str
            Texto com diagnóstico, probabilidade, distância e fatores decisivos.
        """
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Diagnóstico da IA: {e['classe']}",
            f"Probabilidade de malignidade: {e['probabilidade']:.1f}%",
            f"Distância da fronteira: {e['distancia']:+.2f}  "
            f"(quanto mais longe de 0, mais confiante)",
        ]
        if e['limitrofe']:
            linhas.append("")
            linhas.append("⚠ Caso limítrofe: o modelo está pouco decidido.")
            linhas.append("  Recomenda-se revisão clínica deste paciente.")

        linhas.append("")
        linhas.append("Características que mais pesaram nesta decisão:")
        for c in e['contribuicoes'][:6]:
            seta = "↑ Maligno" if c['direcao'] == 'Maligno' else "↓ Benigno"
            posicao = "acima" if c['acima_media'] else "abaixo"
            linhas.append(
                f"   • {c['feature']} = {c['valor']:.3f} ({posicao} da média)"
                f"   {seta}, {c['peso_pct']:.0f}% do peso"
            )

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
