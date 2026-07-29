"""
Módulo contendo a janela de relatório de explicabilidade do KNN.

Apresenta a importância global (por permutação), um mapa de vizinhança 2D
(onde cada paciente aparece entre os pacientes de treino, com seus vizinhos
destacados) e, por paciente, o voto dos K vizinhos que fundamenta a decisão.
"""

import customtkinter as ctk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import adicionar_barra_zoom, bind_treeview_mousewheel, responsive_geometry
from views.report_common import PatientPDFExportMixin


class KNNReportWindow(ctk.CTkToplevel, PatientPDFExportMixin):
    """
    Janela secundária com o relatório de explicabilidade do KNN.

    Reúne o ranking de biomarcadores (importância por permutação), o mapa de
    vizinhança 2D e a área mestre-detalhe por paciente. Selecionar um paciente
    destaca, no mapa, sua posição e os K vizinhos que decidiram o diagnóstico.

    Attributes
    ----------
    _explicacoes : list[dict]
        Explicações por paciente geradas pelo KNNExplainer.
    _train_2d : numpy.ndarray
        Projeção 2D dos pacientes de treino (fundo do mapa).
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_FUNDO = "#2b2b2b"

    def __init__(self, master, importancias: list, explicacoes: list,
                 contexto: dict, **kwargs):
        """
        Inicializa a janela de relatório do KNN.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o relatório.
        importancias : list[tuple[str, float]]
            Ranking de importância por permutação.
        explicacoes : list[dict]
            Explicações por paciente produzidas pelo KNNExplainer.
        contexto : dict
            {'k', 'train_2d', 'train_y'} — parâmetros do modelo e o mapa 2D
            dos pacientes de treino.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade — KNN")
        responsive_geometry(self, 1060, 840)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}
        self._k = int(contexto.get('k', 0))
        self._train_2d = np.asarray(contexto.get('train_2d', []), dtype=float)
        self._train_y = np.asarray(contexto.get('train_y', []))
        self._artistas = []  # artistas dinâmicos do destaque no gráfico

        self._build_header()
        self._build_global(importancias)
        self._build_plot()
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
            text=(f"KNN (k = {self._k})   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {n - malignos}   ·   "
                  f"Casos limítrofes: {limitrofes}"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

    def _build_global(self, importancias: list):
        """
        Constrói o painel de importância global (por permutação).

        Parameters
        ----------
        importancias : list[tuple[str, float]]
            Pares (característica, importância) ordenados do maior para o menor.
        """
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Biomarcadores mais relevantes (por permutação)",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            frame, text="Queda de acurácia do modelo ao embaralhar cada atributo",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        if not importancias:
            ctk.CTkLabel(frame, text="Sem informação disponível.", text_color="gray").grid(
                row=2, column=0, sticky="w", padx=16, pady=(0, 12))
            return

        maior = max(v for _, v in importancias) or 1.0
        for i, (nome, imp) in enumerate(importancias, start=2):
            ctk.CTkLabel(
                frame, text=nome, anchor="w", font=ctk.CTkFont(size=12),
            ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=3)

            barra = ctk.CTkProgressBar(frame, height=14, progress_color="#2980b9")
            barra.set(imp / maior)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)

            ctk.CTkLabel(
                frame, text=f"{imp * 100:.2f}%", width=60, anchor="e",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkFrame(frame, height=8, fg_color="transparent").grid(
            row=len(importancias) + 2, column=0)

    def _build_plot(self):
        """Constrói o mapa de vizinhança 2D (projeção PCA dos pacientes de treino)."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Mapa de vizinhança (projeção PCA 2D)",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        fig = Figure(figsize=(5.0, 4.2), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COR_FUNDO)

        if self._train_2d.size:
            ben = self._train_y == 0
            mal = self._train_y == 1
            ax.scatter(self._train_2d[ben, 0], self._train_2d[ben, 1],
                       c=self.COR_BENIGNO, s=10, alpha=0.25, edgecolors="none", zorder=1)
            ax.scatter(self._train_2d[mal, 0], self._train_2d[mal, 1],
                       c=self.COR_MALIGNO, s=10, alpha=0.25, edgecolors="none", zorder=1)

        ax.set_xlabel("componente principal 1", color="gray", fontsize=9)
        ax.set_ylabel("componente principal 2", color="gray", fontsize=9)
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")

        from matplotlib.lines import Line2D
        legenda = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=self.COR_MALIGNO,
                   markersize=7, label='Treino Maligno'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=self.COR_BENIGNO,
                   markersize=7, label='Treino Benigno'),
            Line2D([0], [0], marker='*', color='none', markerfacecolor='#f1c40f',
                   markeredgecolor='black', markersize=12, label='Paciente'),
        ]
        ax.legend(handles=legenda, facecolor=self.COR_FUNDO, edgecolor="gray",
                  labelcolor="white", fontsize=8, loc="best")
        fig.tight_layout()

        self._ax = ax
        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))

        # Barra de zoom/pan interativa (lupa para explorar a vizinhança).
        barra = adicionar_barra_zoom(self._canvas, frame)
        barra.grid(row=2, column=0, sticky="w", padx=12)

        ctk.CTkLabel(
            frame,
            text="Use a lupa para dar zoom. Posições aproximadas — os vizinhos são "
                 "calculados nos 30 biomarcadores.",
            font=ctk.CTkFont(size=10), text_color="gray", wraplength=420,
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(0, 10))

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
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings")
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
            container, wrap="word",
            font=ctk.CTkFont(family="Courier New", size=13),
        )
        self._detalhe.grid(row=1, column=1, sticky="nsew")
        self._detalhe.insert(
            "1.0",
            "Selecione um paciente na lista para ver o voto dos vizinhos. "
            "O paciente e seus vizinhos são destacados no mapa.",
        )
        self._detalhe.configure(state="disabled")

        if explicacoes:
            primeiro = str(explicacoes[0]['indice'])
            self._tree.selection_set(primeiro)
            self._tree.focus(primeiro)

    def _on_select(self, _event=None):
        """Atualiza o detalhe e destaca o paciente e seus vizinhos no mapa."""
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

        self._destacar(explicacao)

    def _destacar(self, e: dict):
        """Redesenha, no mapa, o paciente selecionado e seus K vizinhos."""
        for art in self._artistas:
            art.remove()
        self._artistas.clear()

        qx, qy = e['x_plot'], e['y_plot']
        # linhas e pontos dos vizinhos
        for viz in e['vizinhos']:
            idx = viz['indice']
            if idx >= len(self._train_2d):
                continue
            vx, vy = self._train_2d[idx]
            cor = self.COR_MALIGNO if viz['classe'] == 'Maligno' else self.COR_BENIGNO
            linha, = self._ax.plot([qx, vx], [qy, vy], color="gray",
                                   linewidth=0.5, alpha=0.5, zorder=2)
            ponto = self._ax.scatter([vx], [vy], c=cor, s=45,
                                     edgecolors="white", linewidths=0.7, zorder=4)
            self._artistas.extend([linha, ponto])

        estrela = self._ax.scatter([qx], [qy], marker="*", s=280, c="#f1c40f",
                                   edgecolors="black", linewidths=1.0, zorder=6)
        self._artistas.append(estrela)
        self._canvas.draw_idle()

    def _formatar_detalhe(self, e: dict) -> str:
        """
        Monta o texto explicativo completo da decisão de um paciente.

        Parameters
        ----------
        e : dict
            Explicação individual produzida pelo KNNExplainer.

        Returns
        -------
        str
            Texto com diagnóstico, voto dos vizinhos e a lista de vizinhos.
        """
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Diagnóstico da IA: {e['classe']}  (confiança {e['confianca']:.0f}%)",
            "",
            f"Voto dos {self._k} vizinhos mais próximos (contagem simples):",
            f"   Maligno: {e['votos_maligno']}     Benigno: {e['votos_benigno']}",
        ]
        if e['pondera_distancia']:
            linhas.append("")
            linhas.append("Peso ponderado por distância (o que decide de fato):")
            linhas.append(f"   Maligno: {e['peso_maligno']:.0f}%     Benigno: {e['peso_benigno']:.0f}%")
            if e['votos_maligno'] > e['votos_benigno']:
                maioria_bruta = 'Maligno'
            elif e['votos_benigno'] > e['votos_maligno']:
                maioria_bruta = 'Benigno'
            else:
                maioria_bruta = None
            if maioria_bruta != e['classe']:
                linhas.append("   (a contagem simples sozinha não seria conclusiva — "
                               "o vizinho mais próximo pesou mais)")
        if e['limitrofe']:
            linhas.append("")
            linhas.append("⚠ Caso limítrofe: voto apertado entre as classes.")
            linhas.append("  Recomenda-se revisão clínica deste paciente.")

        linhas.append("")
        linhas.append("Vizinhos mais próximos (paciente de treino · diagnóstico · distância):")
        for i, viz in enumerate(e['vizinhos'], start=1):
            linhas.append(
                f"   {i:2d}. treino #{viz['indice']:<4d} {viz['classe']:8s} "
                f"dist {viz['distancia']:.2f}"
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
