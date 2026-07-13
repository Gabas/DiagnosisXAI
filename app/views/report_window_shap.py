"""
Módulo contendo a janela de relatório de explicabilidade SHAP.

É uma janela única, parametrizada pelo modelo: mostra a importância global
(pré-calculada no notebook) e, ao selecionar um paciente, calcula sob demanda
(lazy) as contribuições SHAP daquele paciente e as apresenta como um gráfico de
cascata (waterfall). O cálculo por paciente leva frações de segundo, mesmo para
os modelos que usam o KernelExplainer (SVM e KNN).
"""

import customtkinter as ctk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import bind_treeview_mousewheel, responsive_geometry


class ShapReportWindow(ctk.CTkToplevel):
    """
    Janela de relatório SHAP para um modelo específico.

    Attributes
    ----------
    _explainer : ShapExplainer
        Wrapper que calcula os valores SHAP por paciente.
    _X : numpy.ndarray
        Dados padronizados do lote (entrada do SHAP).
    _reais : numpy.ndarray
        Valores brutos das características (apenas para exibição).
    _pacientes : list[dict]
        Lista de {'indice', 'classe'} para a tabela de pacientes.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_FUNDO = "#2b2b2b"

    def __init__(self, master, explainer, X_scaled, valores_reais,
                 pacientes: list, importancias: list, titulo: str, **kwargs):
        """
        Inicializa a janela de relatório SHAP.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o relatório.
        explainer : ShapExplainer
            Explicador SHAP do modelo.
        X_scaled : array-like
            Dados padronizados do lote (n × 30).
        valores_reais : array-like
            Valores brutos das características (n × 30), para exibição.
        pacientes : list[dict]
            Itens {'indice', 'classe'} de cada paciente do lote.
        importancias : list[tuple[str, float]]
            Importância global SHAP (média de |shap|) por característica.
        titulo : str
            Rótulo do modelo (ex.: "Random Forest").
        **kwargs
            Argumentos adicionais para o CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title(f"Relatório SHAP — {titulo}")
        responsive_geometry(self, 1040, 820)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._explainer = explainer
        self._X = np.asarray(X_scaled, dtype=float)
        self._reais = np.asarray(valores_reais, dtype=float)
        self._pacientes = pacientes
        self._titulo = titulo

        self._build_header()
        self._build_global(importancias)
        self._build_lista()
        self._build_detalhe()

        self.after(150, self.lift)
        self.after(200, self.focus)

        if pacientes:
            self._tree.selection_set("0")
            self._tree.focus("0")

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 6))

        ctk.CTkLabel(
            header, text=f"Relatório SHAP — {self._titulo}",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")

        n = len(self._pacientes)
        malignos = sum(1 for p in self._pacientes if p['classe'] == 'Maligno')
        modo = "aproximado (KernelExplainer)" if self._explainer.is_lazy else "exato (TreeExplainer)"
        ctk.CTkLabel(
            header,
            text=(f"{n} paciente(s)   ·   Maligno: {malignos}    Benigno: {n - malignos}"
                  f"   ·   SHAP {modo}   ·   base + Σ(shap) = P(Maligno)"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

    def _build_global(self, importancias: list):
        """
        Constrói o painel de importância global (média de |shap| por atributo).

        Parameters
        ----------
        importancias : list[tuple[str, float]]
            Pares (característica, importância) ordenados do maior para o menor.
        """
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Importância global (SHAP)",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            frame, text="Média do impacto absoluto de cada atributo sobre a predição",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        importancias = importancias[:10]
        if not importancias:
            ctk.CTkLabel(frame, text="Sem informação disponível.", text_color="gray").grid(
                row=2, column=0, sticky="w", padx=16, pady=(0, 12))
            return

        maior = max(v for _, v in importancias) or 1.0
        for i, (nome, imp) in enumerate(importancias, start=2):
            ctk.CTkLabel(
                frame, text=nome, anchor="w", font=ctk.CTkFont(size=12),
            ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=3)
            barra = ctk.CTkProgressBar(frame, height=14, progress_color="#8e44ad")
            barra.set(imp / maior)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)
            ctk.CTkLabel(
                frame, text=f"{imp:.3f}", width=56, anchor="e",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkFrame(frame, height=8, fg_color="transparent").grid(
            row=len(importancias) + 2, column=0)

    def _build_lista(self):
        """Constrói a lista de pacientes (seleciona qual explicar)."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Selecione um paciente",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        self._style_tree()
        tree_frame = ctk.CTkFrame(frame, fg_color="transparent")
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(tree_frame, columns=("paciente", "diagnostico"),
                                  show="headings")
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.column("paciente", width=90, anchor="center")
        self._tree.column("diagnostico", width=110, anchor="center")
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

    def _build_detalhe(self):
        """Constrói a área do gráfico de cascata (waterfall) por paciente."""
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 16))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        self._lbl_detalhe = ctk.CTkLabel(
            frame, text="Selecione um paciente para calcular as contribuições SHAP.",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        )
        self._lbl_detalhe.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        fig = Figure(figsize=(9, 3.4), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(self.COR_FUNDO)
        fig.subplots_adjust(left=0.28, right=0.97, top=0.90, bottom=0.12)
        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _on_select(self, _event=None):
        """Calcula (lazy) e desenha o waterfall SHAP do paciente selecionado."""
        selecao = self._tree.selection()
        if not selecao:
            return
        i = int(selecao[0])

        self._lbl_detalhe.configure(text="Calculando contribuições SHAP…")
        self.update_idletasks()

        r = self._explainer.explain_one(self._X[i], self._reais[i])
        p = self._pacientes[i]

        top = r['contribuicoes'][:10][::-1]  # invertido: maior impacto no topo do barh
        nomes = [f"{c['feature']} = {c['valor']:.2f}" for c in top]
        valores = [c['shap'] for c in top]
        cores = [self.COR_MALIGNO if v > 0 else self.COR_BENIGNO for v in valores]

        self._ax.clear()
        self._ax.barh(range(len(top)), valores, color=cores)
        self._ax.axvline(0, color="gray", linewidth=0.8)
        self._ax.set_yticks(range(len(top)))
        self._ax.set_yticklabels(nomes, fontsize=8, color="white")
        self._ax.set_xlabel("← Benigno      contribuição SHAP para P(Maligno)      Maligno →",
                            color="white", fontsize=9)
        self._ax.tick_params(colors="gray", labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color("gray")
        self._canvas.draw_idle()

        self._lbl_detalhe.configure(
            text=(f"Paciente {p['indice']}  ·  {p['classe']}  ·  "
                  f"P(Maligno) = {r['prob'] * 100:.1f}%   (base {r['base'] * 100:.1f}%)"))

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
