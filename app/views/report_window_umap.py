"""
Módulo contendo a janela do Mapa Populacional (UMAP).

Mostra o embedding UMAP 2D dos pacientes de treino (calculado no notebook) e
posiciona sobre ele os pacientes do lote atual, coloridos pelo diagnóstico da
IA. É independente do modelo — serve para situar os novos pacientes em relação
à população conhecida. A posição de cada paciente é aproximada por média
ponderada dos vizinhos no embedding (feita no app, sem depender do umap).
"""

import webbrowser

import customtkinter as ctk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import (ScrollableFrame, ajustar_ao_conteudo, bind_treeview_mousewheel,
                      figura_responsiva, itens_visiveis, responsive_geometry)


class UmapMapWindow(ctk.CTkToplevel):
    """
    Janela do mapa populacional UMAP.

    Attributes
    ----------
    _train_2d : numpy.ndarray
        Embedding UMAP dos pacientes de treino (fundo do mapa).
    _batch_2d : numpy.ndarray
        Posição aproximada de cada paciente do lote no embedding.
    _pacientes : list[dict]
        Itens {'indice', 'classe'} de cada paciente do lote.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_FUNDO = "#2b2b2b"

    def __init__(self, master, train_2d, train_y, batch_2d, pacientes: list, **kwargs):
        """
        Inicializa a janela do mapa UMAP.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o mapa.
        train_2d : array-like
            Embedding UMAP dos pacientes de treino (n_treino × 2).
        train_y : array-like
            Rótulos reais dos pacientes de treino (0/1).
        batch_2d : array-like
            Posição de cada paciente do lote no embedding (n_lote × 2).
        pacientes : list[dict]
            Itens {'indice', 'classe'} de cada paciente do lote.
        **kwargs
            Argumentos adicionais para o CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Mapa Populacional — UMAP")
        responsive_geometry(self, 980, 760)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Todo o conteúdo vive num corpo rolável: o mapa mais a lista pedem mais
        # altura do que cabe num notebook, e sem rolagem a parte de baixo ficava
        # inacessível, não apenas apertada.
        self._corpo = ScrollableFrame(self, fg_color="transparent")
        self._corpo.grid(row=0, column=0, sticky="nsew")
        self._corpo.grid_columnconfigure(0, weight=3)
        self._corpo.grid_columnconfigure(1, weight=1)
        self._linhas_lista = itens_visiveis(self, 16, minimo=8)

        self._train_2d = np.asarray(train_2d, dtype=float)
        self._train_y = np.asarray(train_y)
        self._batch_2d = np.asarray(batch_2d, dtype=float)
        self._pacientes = pacientes
        self._destaque = None

        self._build_header()
        self._build_plot()
        self._build_lista()

        ajustar_ao_conteudo(self, self._corpo)
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote."""
        header = ctk.CTkFrame(self._corpo, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 6))
        ctk.CTkLabel(
            header, text="Mapa Populacional (UMAP)",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")
        n = len(self._pacientes)
        malignos = sum(1 for p in self._pacientes if p['classe'] == 'Maligno')
        ctk.CTkLabel(
            header,
            text=(f"{n} paciente(s) do lote sobre a população de treino   ·   "
                  f"Maligno: {malignos}    Benigno: {n - malignos}"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

        # Versão interativa (Bokeh): abre no navegador com hover, zoom e legenda.
        ctk.CTkButton(
            header, text="🔍  Ver mapa interativo (navegador)",
            command=self._abrir_interativo, width=270,
            fg_color="#8e44ad", hover_color="#9b59b6",
        ).pack(anchor="w", pady=(10, 0))
        self._lbl_interativo = ctk.CTkLabel(
            header, text="", font=ctk.CTkFont(size=11), text_color="gray",
        )
        self._lbl_interativo.pack(anchor="w", pady=(2, 0))

    def _build_plot(self):
        """Constrói o mapa UMAP (treino ao fundo + pacientes do lote)."""
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 16))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        fig = Figure(figsize=figura_responsiva(self, 6.2, 5.4), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(self.COR_FUNDO)

        tr = self._train_2d
        ben, mal = self._train_y == 0, self._train_y == 1
        self._ax.scatter(tr[ben, 0], tr[ben, 1], c=self.COR_BENIGNO, s=12,
                         alpha=0.22, edgecolors="none", zorder=1)
        self._ax.scatter(tr[mal, 0], tr[mal, 1], c=self.COR_MALIGNO, s=12,
                         alpha=0.22, edgecolors="none", zorder=1)

        bt = self._batch_2d
        cores = [self.COR_MALIGNO if p['classe'] == 'Maligno' else self.COR_BENIGNO
                 for p in self._pacientes]
        self._ax.scatter(bt[:, 0], bt[:, 1], c=cores, s=42, marker="D",
                         edgecolors="white", linewidths=0.6, zorder=3)

        self._ax.set_xlabel("UMAP-1", color="gray", fontsize=9)
        self._ax.set_ylabel("UMAP-2", color="gray", fontsize=9)
        self._ax.tick_params(colors="gray", labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color("gray")

        from matplotlib.lines import Line2D
        legenda = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor="gray",
                   markersize=7, label='Treino (população)'),
            Line2D([0], [0], marker='D', color='none', markerfacecolor=self.COR_MALIGNO,
                   markeredgecolor='white', markersize=8, label='Lote — Maligno'),
            Line2D([0], [0], marker='D', color='none', markerfacecolor=self.COR_BENIGNO,
                   markeredgecolor='white', markersize=8, label='Lote — Benigno'),
        ]
        self._ax.legend(handles=legenda, facecolor=self.COR_FUNDO, edgecolor="gray",
                        labelcolor="white", fontsize=8, loc="best")
        fig.tight_layout()

        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))

        ctk.CTkLabel(
            frame,
            text="Losangos = pacientes do lote. Posições aproximadas por vizinhança no embedding.",
            font=ctk.CTkFont(size=10), text_color="gray",
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 10))

    def _build_lista(self):
        """Constrói a lista de pacientes (para destacar um no mapa)."""
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 16))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Pacientes do lote",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        self._style_tree()
        tree_frame = ctk.CTkFrame(frame, fg_color="transparent")
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(tree_frame, columns=("paciente", "diagnostico"),
                                  show="headings", height=self._linhas_lista)
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.column("paciente", width=80, anchor="center")
        self._tree.column("diagnostico", width=100, anchor="center")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(tree_frame, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        bind_treeview_mousewheel(self._tree)

        self._tree.tag_configure("Maligno", foreground=self.COR_MALIGNO)
        self._tree.tag_configure("Benigno", foreground=self.COR_BENIGNO)
        for i, p in enumerate(self._pacientes):
            self._tree.insert("", "end", iid=str(i),
                              values=(p['indice'], p['classe']), tags=(p['classe'],))
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

    def _abrir_interativo(self):
        """
        Gera o mapa populacional interativo (Bokeh) e o abre no navegador.

        Bokeh renderiza para HTML/JS — o mapa é salvo em arquivo autocontido e
        aberto no navegador padrão, complementando o mapa estático desta janela.
        """
        try:
            from utils.bokeh_map import gerar_mapa_html
            caminho = gerar_mapa_html(
                self._train_2d, self._train_y, self._batch_2d, self._pacientes)
            webbrowser.open(f"file://{caminho}")
            self._lbl_interativo.configure(
                text="Mapa interativo aberto no navegador.", text_color="#2ecc71")
        except ImportError:
            self._lbl_interativo.configure(
                text="Instale o Bokeh para o mapa interativo:  pip install bokeh",
                text_color="#e74c3c")
        except Exception as e:
            self._lbl_interativo.configure(
                text=f"Não foi possível gerar o mapa interativo: {e}", text_color="#e74c3c")

    def _on_select(self, _event=None):
        """Destaca no mapa o paciente selecionado."""
        selecao = self._tree.selection()
        if not selecao:
            return
        i = int(selecao[0])
        if self._destaque is not None:
            self._destaque.remove()
        x, y = self._batch_2d[i]
        self._destaque = self._ax.scatter([x], [y], marker="*", s=320, c="#f1c40f",
                                          edgecolors="black", linewidths=1.0, zorder=6)
        self._canvas.draw_idle()

    def _style_tree(self):
        """Aplica o tema escuro ao componente Treeview."""
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white",
                        fieldbackground="#2b2b2b", borderwidth=0, rowheight=26)
        style.map("Treeview", background=[("selected", "#1f538d")])
        style.configure("Treeview.Heading", background="#1f538d",
                        foreground="white", relief="flat")
        style.map("Treeview.Heading", background=[("active", "#14375e")])
