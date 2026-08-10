"""
Módulo contendo a janela de relatório de explicabilidade da Regressão Logística.

Apresenta as características de maior peso, um gráfico da fronteira de decisão
(onde cada paciente é posicionado em relação ao limiar de malignidade) e o
detalhamento, por paciente, das características que mais influenciaram a decisão.
"""

import customtkinter as ctk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import (ScrollableFrame, adicionar_barra_zoom, ajustar_ao_conteudo,
                      bind_treeview_mousewheel, figura_responsiva, itens_visiveis,
                      responsive_geometry)
from views.report_common import cor_da_classe, PatientPDFExportMixin


class LogisticReportWindow(ctk.CTkToplevel, PatientPDFExportMixin):
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
    COR_REVISAR = "#e67e22"   # laranja: caso devolvido para revisão humana
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
        self._linhas_lista = itens_visiveis(self, 10, minimo=6)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}
        # Coordenadas na curva logística: (escore z = w·x+b, P(Maligno)).
        self._coords = {str(e['indice']): (e['distancia'], e['probabilidade'] / 100.0)
                        for e in explicacoes}
        self._highlight = None

        self._build_header()
        self._build_global(importancias)
        self._build_plot(explicacoes)
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
        benignos = sum(1 for e in self._explicacoes if e['classe'] == 'Benigno')
        adiados = n - malignos - benignos
        revisar = f"    Revisar: {adiados}" if adiados else ""
        limitrofes = sum(1 for e in self._explicacoes if e['limitrofe'])
        ctk.CTkLabel(
            header,
            text=(f"Regressão Logística   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {benignos}{revisar}   ·   "
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
        frame = ctk.CTkFrame(self._corpo)
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

        importancias = importancias[:itens_visiveis(self, 10)]
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
        Constrói o gráfico da função de decisão (curva logística).

        Eixo X: o escore linear z = w·x + b (a "distância" com sinal até a
        fronteira). Eixo Y: a probabilidade P(Maligno) = σ(z). A curva em S é a
        própria função logística que converte z em probabilidade; cada paciente
        fica exatamente sobre ela, em (seu z, sua P). A linha vertical em z = 0
        e a horizontal em P = 0,5 se cruzam no ponto de decisão.

        Parameters
        ----------
        explicacoes : list[dict]
            Explicações por paciente, com 'distancia' (z) e 'probabilidade'.
        """
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Função de decisão — curva logística",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        zs = [e['distancia'] for e in explicacoes]
        ps = [e['probabilidade'] / 100.0 for e in explicacoes]
        cores = [cor_da_classe(e['classe'])
                 for e in explicacoes]

        fig = Figure(figsize=figura_responsiva(self, 5.0, 3.8), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COR_FUNDO)

        limite = max(4.0, max((abs(z) for z in zs), default=0.0) * 1.15)
        grade = np.linspace(-limite, limite, 300)
        sigmoide = 1.0 / (1.0 + np.exp(-grade))

        # Regiões de decisão + limiares (o modelo decide em z = 0  ⇔  P = 0,5).
        ax.axvspan(-limite, 0, color=self.COR_BENIGNO, alpha=0.07)
        ax.axvspan(0, limite, color=self.COR_MALIGNO, alpha=0.07)
        ax.axhline(0.5, color="gray", linewidth=1.0, linestyle=":", zorder=1)
        ax.axvline(0, color="white", linewidth=1.3, linestyle="--", zorder=1)

        # A curva logística σ(z) e cada paciente sobre ela.
        ax.plot(grade, sigmoide, color="#5dade2", linewidth=2.0, zorder=2,
                label="σ(z) = 1 / (1 + e^−z)")
        ax.scatter(zs, ps, c=cores, s=26, alpha=0.9, edgecolors="white",
                   linewidths=0.4, zorder=4)

        # Marcador (vazio) para destacar o paciente selecionado.
        self._highlight = ax.scatter(
            [], [], s=170, facecolors="none", edgecolors="#f1c40f",
            linewidths=2.2, zorder=6)

        ax.set_xlim(-limite, limite)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("escore linear  z = w·x + b   (←  Benigno   ·   Maligno  →)",
                      color="white", fontsize=9)
        ax.set_ylabel("P(Maligno) = σ(z)", color="white", fontsize=9)
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.legend(facecolor=self.COR_FUNDO, edgecolor="gray", labelcolor="white",
                  fontsize=8, loc="upper left")
        fig.tight_layout()

        self._ax = ax
        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))

        barra = adicionar_barra_zoom(self._canvas, frame)
        barra.grid(row=2, column=0, sticky="w", padx=12)
        ctk.CTkLabel(
            frame,
            text="X = escore linear z (distância à fronteira). Y = probabilidade P(Maligno) "
                 "= σ(z). A curva converte z em probabilidade; z = 0 ⇔ P = 50% é a fronteira. "
                 "Use a lupa para dar zoom.",
            font=ctk.CTkFont(size=10), text_color="gray", wraplength=420, justify="left",
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(0, 10))

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

        colunas = ("paciente", "diagnostico", "prob", "obs")
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings",
                                  height=self._linhas_lista)
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
        self._tree.tag_configure("Revisar", foreground=self.COR_REVISAR)

        for e in explicacoes:
            obs = "⚠ limítrofe" if e['limitrofe'] else ""
            self._tree.insert(
                "", "end", iid=str(e['indice']),
                values=(e['indice'], e['classe'], f"{e['probabilidade']:.1f}%", obs),
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
        z = e['distancia']
        decisao = "z ≥ 0  →  Maligno" if z >= 0 else "z < 0  →  Benigno"
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Diagnóstico da IA: {e['classe']}",
            "",
            "FUNÇÃO DE DECISÃO (Regressão Logística)",
            "   z = w·x + b   (soma ponderada dos 30 biomarcadores",
            "                  padronizados, mais o viés b)",
            f"   z = {z:+.3f}",
            f"   P(Maligno) = σ(z) = 1 / (1 + e^(−z)) = {e['probabilidade']:.1f}%",
            f"   Decisão:  {decisao}",
            "",
            f"Distância da fronteira: {z:+.2f}  "
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
