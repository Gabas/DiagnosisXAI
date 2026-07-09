"""
Módulo contendo a janela de relatório de explicabilidade do SVM (kernel RBF).

Diferente de um SVM linear, o kernel RBF não define um hiperplano no espaço
original das características — por isso este relatório não desenha uma
"fronteira" geométrica. Em vez disso, mostra exatamente o que decide cada
paciente: os vetores de suporte (pacientes de treino) mais similares a ele,
ponderados pela força de cada um na soma que forma a decision_function. O mapa
de fundo é o mesmo embedding UMAP usado no Mapa Populacional do app, com os
vetores de suporte em destaque — os únicos pacientes de treino que de fato
participam da decisão.
"""

import customtkinter as ctk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import bind_treeview_mousewheel
from views.report_common import PatientPDFExportMixin


class SVMReportWindow(ctk.CTkToplevel, PatientPDFExportMixin):
    """
    Janela secundária com o relatório de explicabilidade do SVM (kernel RBF).

    Reúne o ranking de biomarcadores (importância por permutação), o mapa
    populacional (UMAP) com os vetores de suporte em destaque, e a área
    mestre-detalhe por paciente. Selecionar um paciente traça, no mapa, as
    ligações com os vetores de suporte que mais pesaram em sua decisão.

    Attributes
    ----------
    _explicacoes : list[dict]
        Explicações por paciente geradas pelo SVMExplainer.
    _train_2d : numpy.ndarray
        Embedding UMAP dos pacientes de treino (fundo do mapa).
    _batch_2d : numpy.ndarray ou None
        Posição aproximada de cada paciente do lote no embedding — None
        quando o Mapa Populacional não está disponível (.pkl sem UMAP).
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_FUNDO = "#2b2b2b"

    def __init__(self, master, importancias: list, explicacoes: list,
                 contexto: dict, batch_2d=None, **kwargs):
        """
        Inicializa a janela de relatório do SVM.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o relatório.
        importancias : list[tuple[str, float]]
            Ranking de importância por permutação.
        explicacoes : list[dict]
            Explicações por paciente produzidas pelo SVMExplainer.
        contexto : dict
            {'train_2d', 'train_y', 'sv_indices', 'n_support',
             'n_support_maligno', 'n_support_benigno'} — o embedding de treino
             e quais desses pacientes são vetores de suporte.
        batch_2d : array-like ou None
            Posição de cada paciente do lote no embedding (n_lote × 2), na
            mesma ordem de ``explicacoes``. None se o embedding UMAP não
            estiver disponível.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade — SVM")
        self.geometry("1060x860")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}
        self._pos_por_indice = {str(e['indice']): i for i, e in enumerate(explicacoes)}

        self._train_2d = np.asarray(contexto.get('train_2d', []), dtype=float)
        self._train_y = np.asarray(contexto.get('train_y', []))
        self._sv_indices = np.asarray(contexto.get('sv_indices', []), dtype=int)
        self._n_support = int(contexto.get('n_support', 0))
        self._n_sv_mal = contexto.get('n_support_maligno')
        self._n_sv_ben = contexto.get('n_support_benigno')

        self._batch_2d = np.asarray(batch_2d, dtype=float) if batch_2d is not None else None
        self._artistas = []  # artistas dinâmicos do destaque no gráfico

        self._build_header()
        self._build_global(importancias)
        self._build_plot()
        self._build_per_patient(explicacoes)

        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote e dos vetores de suporte."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 6))

        ctk.CTkLabel(
            header, text="Relatório de Explicabilidade",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")

        n = len(self._explicacoes)
        malignos = sum(1 for e in self._explicacoes if e['classe'] == 'Maligno')
        limitrofes = sum(1 for e in self._explicacoes if e['limitrofe'])
        sv_txt = f"{self._n_support} vetores de suporte"
        if self._n_sv_mal is not None and self._n_sv_ben is not None:
            sv_txt += f" ({self._n_sv_mal} Maligno / {self._n_sv_ben} Benigno)"
        ctk.CTkLabel(
            header,
            text=(f"SVM (kernel RBF)   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {n - malignos}   ·   "
                  f"Casos limítrofes: {limitrofes}\n{sv_txt}"),
            font=ctk.CTkFont(size=13), text_color="gray", justify="left",
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

            barra = ctk.CTkProgressBar(frame, height=14, progress_color="#d35400")
            barra.set(imp / maior)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)

            ctk.CTkLabel(
                frame, text=f"{imp * 100:.2f}%", width=60, anchor="e",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkFrame(frame, height=8, fg_color="transparent").grid(
            row=len(importancias) + 2, column=0)

    def _build_plot(self):
        """Constrói o mapa populacional (UMAP) com os vetores de suporte em destaque."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Mapa populacional (UMAP) e vetores de suporte",
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
                       c=self.COR_BENIGNO, s=8, alpha=0.15, edgecolors="none", zorder=1)
            ax.scatter(self._train_2d[mal, 0], self._train_2d[mal, 1],
                       c=self.COR_MALIGNO, s=8, alpha=0.15, edgecolors="none", zorder=1)

            # Vetores de suporte: os únicos pacientes de treino que de fato
            # participam da decisão — destacados sobre o fundo apagado.
            if self._sv_indices.size:
                sv_pos = self._train_2d[self._sv_indices]
                sv_classe = self._train_y[self._sv_indices]
                sv_ben, sv_mal = sv_classe == 0, sv_classe == 1
                ax.scatter(sv_pos[sv_ben, 0], sv_pos[sv_ben, 1], c=self.COR_BENIGNO,
                           s=32, alpha=0.75, edgecolors="white", linewidths=0.4, zorder=2)
                ax.scatter(sv_pos[sv_mal, 0], sv_pos[sv_mal, 1], c=self.COR_MALIGNO,
                           s=32, alpha=0.75, edgecolors="white", linewidths=0.4, zorder=2)

        if self._batch_2d is not None:
            for e, (x, y) in zip(self._explicacoes, self._batch_2d):
                # Peso visual (tamanho/opacidade) proporcional à confiança do
                # modelo — quanto mais perto da margem, mais discreto o marcador.
                conf = max(0.0, min(1.0, (e['confianca'] - 50.0) / 50.0))
                cor = self.COR_MALIGNO if e['classe'] == 'Maligno' else self.COR_BENIGNO
                ax.scatter([x], [y], c=cor, s=32 + 40 * conf, marker="D",
                          alpha=0.35 + 0.65 * conf, edgecolors="white",
                          linewidths=0.6, zorder=3)
        else:
            ax.text(0.5, 0.5, "Mapa indisponível\n(regenere o wisconsin.pkl)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)

        ax.set_xlabel("UMAP-1", color="gray", fontsize=9)
        ax.set_ylabel("UMAP-2", color="gray", fontsize=9)
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")

        from matplotlib.lines import Line2D
        legenda = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor="gray",
                   markersize=6, label='Treino (população)'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor="white",
                   markeredgecolor='gray', markersize=8, label='Vetor de suporte'),
            Line2D([0], [0], marker='D', color='none', markerfacecolor="white",
                   markeredgecolor='gray', markersize=8, label='Paciente (lote)'),
        ]
        ax.legend(handles=legenda, facecolor=self.COR_FUNDO, edgecolor="gray",
                  labelcolor="white", fontsize=8, loc="best")
        fig.tight_layout()

        self._ax = ax
        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            frame,
            text="Tamanho/opacidade do paciente = confiança do modelo. "
                 "Selecione um paciente para ver seus vetores de suporte.",
            font=ctk.CTkFont(size=10), text_color="gray", wraplength=420,
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 10))

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

        colunas = ("paciente", "diagnostico", "confianca", "margem")
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings")
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.heading("confianca", text="Confiança")
        self._tree.heading("margem", text="Distância à margem")
        self._tree.column("paciente", width=70, anchor="center", stretch=False)
        self._tree.column("diagnostico", width=90, anchor="center", stretch=False)
        self._tree.column("confianca", width=80, anchor="center", stretch=False)
        self._tree.column("margem", width=130, anchor="center")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(tree_frame, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        bind_treeview_mousewheel(self._tree)

        self._tree.tag_configure("Maligno", foreground=self.COR_MALIGNO)
        self._tree.tag_configure("Benigno", foreground=self.COR_BENIGNO)

        for e in explicacoes:
            margem = f"{e['distancia']:+.3f}"
            if e['limitrofe']:
                margem = "⚠ " + margem
            self._tree.insert(
                "", "end", iid=str(e['indice']),
                values=(e['indice'], e['classe'], f"{e['confianca']:.0f}%", margem),
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
            "Selecione um paciente na lista para ver os vetores de suporte "
            "que mais pesaram na sua decisão.",
        )
        self._detalhe.configure(state="disabled")

        if explicacoes:
            primeiro = str(explicacoes[0]['indice'])
            self._tree.selection_set(primeiro)
            self._tree.focus(primeiro)

    def _on_select(self, _event=None):
        """Atualiza o detalhe e destaca, no mapa, os vetores de suporte do paciente."""
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

        self._destacar(explicacao, selecao[0])

    def _destacar(self, e: dict, chave: str):
        """Redesenha, no mapa, o paciente selecionado e seus vetores de suporte."""
        for art in self._artistas:
            art.remove()
        self._artistas.clear()

        if self._batch_2d is None:
            return
        pos = self._pos_por_indice.get(chave)
        if pos is None:
            return
        qx, qy = self._batch_2d[pos]

        vetores = e['top_maligno'] + e['top_benigno']
        maior = max((abs(v['contribuicao']) for v in vetores), default=1.0) or 1.0
        for v in vetores:
            idx = v['indice_treino']
            if idx >= len(self._train_2d):
                continue
            vx, vy = self._train_2d[idx]
            cor = self.COR_MALIGNO if v['classe'] == 'Maligno' else self.COR_BENIGNO
            largura = 0.4 + 2.2 * (abs(v['contribuicao']) / maior)
            linha, = self._ax.plot([qx, vx], [qy, vy], color=cor,
                                   linewidth=largura, alpha=0.6, zorder=4)
            ponto = self._ax.scatter([vx], [vy], c=cor, s=60,
                                     edgecolors="white", linewidths=0.8, zorder=5)
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
            Explicação individual produzida pelo SVMExplainer.

        Returns
        -------
        str
            Texto com diagnóstico, distância à margem e os vetores de suporte
            que mais pesaram na decisão.
        """
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Diagnóstico da IA: {e['classe']}  (confiança {e['confianca']:.0f}%)",
            f"P(Maligno) = {e['probabilidade']:.1f}%   ·   "
            f"distância à margem = {e['distancia']:+.3f}",
        ]
        if e['limitrofe']:
            linhas.append("")
            linhas.append("⚠ Caso limítrofe: o modelo está pouco decidido.")
            linhas.append("  Recomenda-se revisão clínica deste paciente.")

        linhas.append("")
        linhas.append("Balanço de forças (soma ponderada da similaridade com os")
        linhas.append("vetores de suporte — decide a soma, não um único fator):")
        linhas.append(f"   A favor de Maligno   ({self._n_sv_mal} vetores):  +{e['forca_maligno']:.3f}")
        linhas.append(f"   A favor de Benigno   ({self._n_sv_ben} vetores):  -{e['forca_benigno']:.3f}")
        linhas.append(f"   Viés do modelo:                    {e['vies']:+.3f}")
        linhas.append(f"   {'-' * 44}")
        linhas.append(f"   Distância à margem (resultado):    {e['distancia']:+.3f}")

        for rotulo, chave in (("Maligno", "top_maligno"), ("Benigno", "top_benigno")):
            linhas.append("")
            linhas.append(f"Vetores de suporte mais influentes a favor de {rotulo}")
            linhas.append("(paciente de treino · contribuição · similaridade):")
            if not e[chave]:
                linhas.append("   (nenhum vetor de suporte deste lado)")
            for i, v in enumerate(e[chave], start=1):
                linhas.append(
                    f"   {i:2d}. treino #{v['indice_treino']:<4d} "
                    f"{v['contribuicao']:+.3f}   similaridade {v['similaridade']:.2f}"
                )

        linhas.append("")
        linhas.append(
            "Nota: o SVM (kernel RBF) não tem um \"hiperplano\" no espaço original\n"
            "dos 30 biomarcadores — a decisão soma a similaridade deste paciente com\n"
            "TODOS os vetores de suporte (não só os listados acima); os valores de\n"
            "\"a favor de\" já incluem essa soma completa de cada lado."
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
