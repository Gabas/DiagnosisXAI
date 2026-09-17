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

Sobre desenhar a margem
-----------------------
A margem NÃO é traçada como contorno sobre o mapa UMAP, e isso é deliberado: o
UMAP é uma projeção não invertível, então não há como avaliar a
``decision_function`` num ponto arbitrário do plano — só restaria interpolar o
escore dos pacientes vizinhos. Medida sobre o treino (validação leave-one-out),
essa interpolação acerta o lado da fronteira em ~96% dos casos no geral, mas
erra justamente onde a margem existe para avisar: de cada dois pacientes com
|z| < 1 (dentro da margem, baixa confiança), um seria desenhado FORA dela. É o
mesmo critério já registrado em ``core/calculos.py`` para o detector de perfis
atípicos: o mapa 2D serve para inspeção visual, não para decisão.

A margem aparece então nos dois lugares onde é exata:
- no mapa, a cor de cada ponto é o escore z real daquele paciente (nenhum valor
  interpolado), e os pacientes com |z| < 1 recebem um anel;
- no painel da margem, o eixo horizontal É o escore z, com as faixas em -1, 0 e
  +1 matematicamente corretas. Como z é um escalar, esse eixo não é uma
  simplificação do problema: é o objeto real.
"""

import webbrowser

import customtkinter as ctk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.ui import (ScrollableFrame, adicionar_barra_zoom, ajustar_ao_conteudo,
                      bind_treeview_mousewheel, figura_responsiva, itens_visiveis,
                      responsive_geometry)
from views.report_common import cor_da_classe, PatientPDFExportMixin


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
    _train_z : numpy.ndarray ou None
        Escore de decisão exato de cada paciente de treino. None em sessões
        antigas do histórico, salvas antes deste campo existir — nesse caso o
        mapa volta a colorir por classe e o painel da margem omite o fundo.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_REVISAR = "#e67e22"   # laranja: caso devolvido para revisão humana
    COR_FUNDO = "#2b2b2b"

    # Escala de cor do escore z. Diverge nas mesmas cores que o resto do app usa
    # para as classes (verde=Benigno, vermelho=Maligno) — introduzir uma terceira
    # cor aqui faria "Benigno" ter dois significados visuais na mesma tela. O
    # valor numérico continua legível na coluna "Distância à margem" da lista,
    # então a informação não depende só da cor.
    CMAP_Z = LinearSegmentedColormap.from_list(
        "svm_z", ["#2ecc71", "#7fd9a5", "#d8d8d8", "#ef9a90", "#e74c3c"])
    Z_MAX = 3.0   # saturação da escala: |z| > 3 é decisão folgada dos dois lados

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
             e quais desses pacientes são vetores de suporte. Aceita também
             'train_z' (escore de decisão de cada paciente de treino), montado
             por ``predict_view._contexto_svm``.
        batch_2d : array-like ou None
            Posição de cada paciente do lote no embedding (n_lote × 2), na
            mesma ordem de ``explicacoes``. None se o embedding UMAP não
            estiver disponível.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade: SVM")
        responsive_geometry(self, 1060, 860)
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
        self._pos_por_indice = {str(e['indice']): i for i, e in enumerate(explicacoes)}

        self._train_2d = np.asarray(contexto.get('train_2d', []), dtype=float)
        self._train_y = np.asarray(contexto.get('train_y', []))
        self._sv_indices = np.asarray(contexto.get('sv_indices', []), dtype=int)
        self._n_support = int(contexto.get('n_support', 0))
        self._n_sv_mal = contexto.get('n_support_maligno')
        self._n_sv_ben = contexto.get('n_support_benigno')

        train_z = contexto.get('train_z')
        self._train_z = np.asarray(train_z, dtype=float) if train_z is not None else None
        if self._train_z is not None and len(self._train_z) != len(self._train_2d):
            self._train_z = None   # contexto inconsistente: não arrisca colorir errado

        self._batch_2d = np.asarray(batch_2d, dtype=float) if batch_2d is not None else None
        self._artistas = []          # artistas dinâmicos do destaque no mapa
        self._artistas_margem = []   # idem, no painel da margem
        self._margem_xy = {}         # posição de cada paciente no painel da margem

        self._build_header()
        self._build_global(importancias)
        self._build_plot()
        self._build_margem()
        self._build_per_patient(explicacoes)

        ajustar_ao_conteudo(self, self._corpo)
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote e dos vetores de suporte."""
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
        sv_txt = f"{self._n_support} vetores de suporte"
        if self._n_sv_mal is not None and self._n_sv_ben is not None:
            sv_txt += f" ({self._n_sv_mal} Maligno / {self._n_sv_ben} Benigno)"
        ctk.CTkLabel(
            header,
            text=(f"SVM (kernel RBF)   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {benignos}{revisar}   ·   "
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
        frame = ctk.CTkFrame(self._corpo)
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

        importancias = importancias[:itens_visiveis(self, 10)]
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
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Mapa populacional (UMAP) e vetores de suporte",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        fig = Figure(figsize=figura_responsiva(self, 5.0, 3.8), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COR_FUNDO)

        colorido = self._train_z is not None and self._train_2d.size
        if colorido:
            # Cor = escore de decisão REAL de cada paciente de treino. Nenhum
            # valor é interpolado: o que está pintado é o que o SVM calculou
            # para aquele paciente, não uma estimativa para a região do plano.
            norma = Normalize(vmin=-self.Z_MAX, vmax=self.Z_MAX)
            pontos = ax.scatter(self._train_2d[:, 0], self._train_2d[:, 1],
                                c=self._train_z, cmap=self.CMAP_Z, norm=norma,
                                s=18, alpha=0.85, edgecolors="none", zorder=1)
            barra_cor = fig.colorbar(pontos, ax=ax, fraction=0.045, pad=0.02,
                                     extend="both")
            barra_cor.set_label("z = escore de decisão", color="gray", fontsize=8)
            barra_cor.ax.tick_params(colors="gray", labelsize=7)
            barra_cor.outline.set_edgecolor("gray")

        elif self._train_2d.size:
            ben = self._train_y == 0
            mal = self._train_y == 1
            ax.scatter(self._train_2d[ben, 0], self._train_2d[ben, 1],
                       c=self.COR_BENIGNO, s=8, alpha=0.15, edgecolors="none", zorder=1)
            ax.scatter(self._train_2d[mal, 0], self._train_2d[mal, 1],
                       c=self.COR_MALIGNO, s=8, alpha=0.15, edgecolors="none", zorder=1)

        # Vetores de suporte: os únicos pacientes de treino que de fato
        # participam da decisão. Com o mapa colorido por z a cor já está
        # ocupada, então eles são marcados pelo contorno, não pelo preenchimento.
        if self._train_2d.size and self._sv_indices.size:
            sv_pos = self._train_2d[self._sv_indices]
            if colorido:
                # Todo paciente com |z| < 1 é vetor de suporte (o contrário não
                # vale: há vetores de suporte fora da margem, os que o modelo
                # errou ou acertou raspando). Marcar as duas coisas com anéis
                # separados poria dois anéis no mesmo ponto — então é um anel
                # só, mais forte para quem está dentro da margem.
                dentro = np.abs(self._train_z[self._sv_indices]) < 1.0
                ax.scatter(sv_pos[~dentro, 0], sv_pos[~dentro, 1], facecolors="none",
                           edgecolors="#7f8c8d", s=36, linewidths=0.6,
                           alpha=0.65, zorder=2)
                ax.scatter(sv_pos[dentro, 0], sv_pos[dentro, 1], facecolors="none",
                           edgecolors="white", s=52, linewidths=0.9,
                           alpha=0.75, zorder=2)
            else:
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
                cor = cor_da_classe(e['classe'])
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
        if colorido:
            legenda = [
                Line2D([0], [0], marker='o', color='none', markerfacecolor="gray",
                       markersize=6, label='Treino (cor = z real)'),
                Line2D([0], [0], marker='D', color='none', markerfacecolor="white",
                       markeredgecolor='gray', markersize=8, label='Paciente (lote)'),
                Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                       markeredgecolor='white', markeredgewidth=1.2, markersize=8,
                       label='Vetor de suporte · |z| < 1'),
                Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                       markeredgecolor='#7f8c8d', markersize=7,
                       label='Vetor de suporte · |z| ≥ 1'),
            ]
            # Com quatro entradas a legenda cobriria o miolo do mapa. Vai para
            # baixo dos eixos, em duas colunas — e presa à figura (fig.legend),
            # não aos eixos: assim não escorrega para fora da borda quando
            # figura_responsiva encolhe o gráfico em telas menores.
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.34)
            fig.legend(handles=legenda, facecolor=self.COR_FUNDO, edgecolor="gray",
                       labelcolor="white", fontsize=7.5, ncol=2,
                       loc="lower center", bbox_to_anchor=(0.5, 0.005))
        else:
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

        # Zoom/pan no mapa.
        barra = adicionar_barra_zoom(self._canvas, frame)
        barra.grid(row=2, column=0, sticky="w", padx=12)

        if colorido:
            dica = ("Cor = escore z real de cada paciente de treino (nada interpolado). "
                    "Os pacientes dentro da margem aparecem nos dois aglomerados — por "
                    "isso não há uma fronteira a traçar aqui; ela está no painel abaixo. "
                    "Selecione um paciente para ver seus vetores de suporte.")
        else:
            dica = ("Tamanho/opacidade do paciente = confiança do modelo. "
                    "Selecione um paciente para ver seus vetores de suporte.")
        ctk.CTkLabel(
            frame, text=dica, font=ctk.CTkFont(size=10), text_color="gray",
            wraplength=420, justify="left",
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(4, 10))

    def _build_margem(self):
        """
        Constrói o painel da margem: o eixo horizontal É o escore de decisão.

        É aqui que a margem aparece desenhada, e de forma exata: como z é um
        escalar, a faixa entre z = -1 e z = +1 (a "rua" que o SVM maximiza) e a
        fronteira em z = 0 são linhas verticais sem nenhuma aproximação. O
        histograma ao fundo é a distribuição do treino, que situa o lote na
        população; os losangos são os pacientes deste lote.
        """
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 10))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Margem do SVM: onde cada paciente cai em relação à fronteira",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        fig = Figure(figsize=figura_responsiva(self, 10.0, 2.7), dpi=100)
        fig.patch.set_facecolor(self.COR_FUNDO)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COR_FUNDO)

        zs = np.array([float(e['distancia']) for e in self._explicacoes], dtype=float)
        todos = zs if self._train_z is None else np.concatenate([zs, self._train_z])
        limite = max(self.Z_MAX, float(np.abs(todos).max()) + 0.3) if todos.size else self.Z_MAX
        ax.set_xlim(-limite, limite)
        ax.set_ylim(0, 1)

        # Faixas: a margem no centro, as zonas de decisão folgada nas pontas.
        ax.axvspan(-1, 1, color="#7f8c8d", alpha=0.20, zorder=0)
        ax.axvspan(-limite, -1, color=self.COR_BENIGNO, alpha=0.07, zorder=0)
        ax.axvspan(1, limite, color=self.COR_MALIGNO, alpha=0.07, zorder=0)
        ax.axvline(0, color="white", linestyle="--", linewidth=1.8, zorder=1)
        for borda in (-1, 1):   # bordas da margem: passam pelos vetores de suporte
            ax.axvline(borda, color="gray", linestyle=":", linewidth=1.0, zorder=1)

        # Distribuição do treino, rente à base — o pano de fundo populacional.
        if self._train_z is not None and self._train_z.size:
            contagens, bordas = np.histogram(self._train_z, bins=40)
            if contagens.max():
                alturas = contagens / contagens.max() * 0.20
                ax.bar(bordas[:-1], alturas, width=np.diff(bordas), align="edge",
                       color="#95a5a6", alpha=0.40, linewidth=0, zorder=1)

        # Pacientes do lote, espalhados na vertical só para não se sobreporem:
        # a posição horizontal é o z exato, a vertical não carrega informação.
        ys = self._espalhar(zs, 2 * limite)
        for e, x, y in zip(self._explicacoes, zs, ys):
            self._margem_xy[str(e['indice'])] = (float(x), float(y))
            ax.scatter([x], [y], c=cor_da_classe(e['classe']), s=58, marker="D",
                       edgecolors="white", linewidths=0.7, zorder=3)

        # Os nomes das zonas vão num eixo secundário acima do gráfico, e não
        # dentro dele: num lote com muitos casos limítrofes a colmeia cresce em
        # altura e alcançaria qualquer texto colocado no topo da área de dados.
        topo = ax.secondary_xaxis("top")
        topo.set_xticks([-(limite + 1) / 2, 0.0, (limite + 1) / 2])
        topo.set_xticklabels(["Benigno\n(decisão folgada)", "margem\n(baixa confiança)",
                              "Maligno\n(decisão folgada)"])
        topo.tick_params(colors="gray", labelsize=8, length=0)
        topo.spines["top"].set_color("none")

        ax.set_xlabel("z = escore de decisão (decision_function, valor exato)",
                      color="gray", fontsize=9)
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis="x", colors="gray", labelsize=8)
        for lado, spine in ax.spines.items():
            spine.set_color("gray" if lado == "bottom" else "none")
        fig.tight_layout()

        self._ax_margem = ax
        self._canvas_margem = FigureCanvasTkAgg(fig, master=frame)
        self._canvas_margem.draw()
        self._canvas_margem.get_tk_widget().grid(
            row=1, column=0, sticky="nsew", padx=12, pady=(0, 4))

        rodape = ctk.CTkFrame(frame, fg_color="transparent")
        rodape.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        rodape.grid_columnconfigure(1, weight=1)

        # Mesma margem, com hover paciente a paciente, no navegador.
        ctk.CTkButton(
            rodape, text="🔍  Abrir versão interativa (navegador)",
            command=self._abrir_margem_interativa, width=280,
            fg_color="#8e44ad", hover_color="#9b59b6",
        ).grid(row=0, column=0, sticky="w")
        self._lbl_margem = ctk.CTkLabel(
            rodape, text="", font=ctk.CTkFont(size=11), text_color="gray")
        self._lbl_margem.grid(row=0, column=1, sticky="w", padx=(12, 0))

        fundo = ("  A área cinza ao fundo é a distribuição do treino."
                 if self._train_z is not None else "")
        ctk.CTkLabel(
            frame,
            text=("A faixa cinza central é a margem que o SVM maximiza; suas bordas "
                  "(z = ±1) passam pelos vetores de suporte, e a linha tracejada é a "
                  "fronteira (z = 0). Quem cai dentro da faixa é caso de baixa "
                  "confiança." + fundo +
                  "  A altura dos losangos não tem significado: serve só para separá-los."),
            font=ctk.CTkFont(size=10), text_color="gray", justify="left", wraplength=900,
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(0, 10))

    @staticmethod
    def _espalhar(xs, largura_eixo: float, faixa=(0.28, 0.92), minimo=0.035):
        """
        Distribui verticalmente pontos de mesma abscissa, para que não se cubram.

        Percorre os pontos da esquerda para a direita e coloca cada um na faixa
        mais baixa que ainda esteja livre naquela altura — uma colmeia simples.
        Determinístico (ao contrário de um deslocamento aleatório), então o mesmo
        lote desenha sempre igual, inclusive no PDF exportado.

        Parameters
        ----------
        xs : array-like
            Abscissas dos pontos (aqui, o escore z de cada paciente).
        largura_eixo : float
            Extensão total do eixo x, usada para converter ``minimo`` em unidades
            de dado.
        faixa : tuple, optional
            Intervalo vertical (em coordenadas de dado) a ocupar.
        minimo : float, optional
            Distância horizontal mínima entre pontos de uma mesma altura, como
            fração da largura do eixo.

        Returns
        -------
        numpy.ndarray
            Altura de cada ponto, na ordem original de ``xs``.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.full(xs.shape, (faixa[0] + faixa[1]) / 2.0)
        if xs.size == 0:
            return ys

        folga = minimo * max(largura_eixo, 1e-9)
        niveis = []                      # último x ocupado em cada altura
        for i in np.argsort(xs):
            alvo = next((n for n, ultimo in enumerate(niveis) if xs[i] - ultimo > folga),
                        len(niveis))
            if alvo == len(niveis):
                niveis.append(xs[i])
            else:
                niveis[alvo] = xs[i]
            ys[i] = alvo

        # Níveis 0,1,2,... -> alturas alternando em torno do centro da faixa
        # (centro, acima, abaixo, mais acima, ...), para a nuvem ficar
        # equilibrada em vez de crescer só para um lado.
        usados = int(ys.max()) + 1
        if usados > 1:
            centro = (faixa[0] + faixa[1]) / 2.0
            # O nível mais alto fica a (usados // 2) passos do centro; é esse
            # múltiplo que precisa caber em meia faixa, não (usados - 1).
            passo = (faixa[1] - faixa[0]) / 2.0 / (usados // 2)
            alturas = np.array([
                centro + ((n + 1) // 2) * passo * (1 if n % 2 else -1)
                for n in range(usados)
            ])
            # Com um número par de níveis sobra sempre um acima do centro; sem
            # reequilibrar, a nuvem fica encostada no topo da faixa.
            alturas -= (alturas.max() + alturas.min()) / 2.0 - centro
            ys = alturas[ys.astype(int)]
        return ys

    def _build_per_patient(self, explicacoes: list):
        """
        Constrói a área mestre-detalhe com a decisão de cada paciente.

        Parameters
        ----------
        explicacoes : list[dict]
            Explicações por paciente a serem listadas e detalhadas.
        """
        container = ctk.CTkFrame(self._corpo, fg_color="transparent")
        container.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 16))
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
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings",
                                  height=self._linhas_lista)
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
        self._tree.tag_configure("Revisar", foreground=self.COR_REVISAR)

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
            container, wrap="word", height=self._linhas_lista * 26,
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

    def _figuras_pdf(self):
        """
        Figuras a embutir no PDF do paciente: o mapa e o painel da margem.

        O mapa sozinho não diz onde o paciente caiu em relação à fronteira —
        essa é justamente a informação que o painel da margem carrega, com a
        estrela já posicionada no z dele.

        Returns
        -------
        list
            As figuras matplotlib da janela, na ordem em que aparecem na tela.
        """
        return [self._canvas.figure, self._canvas_margem.figure]

    def _abrir_margem_interativa(self):
        """
        Abre no navegador a mesma margem do painel, agora com zoom e hover (Bokeh).

        O painel embutido já mostra a margem exata; esta versão existe para
        lotes grandes, em que identificar um paciente específico pelo losango
        fica difícil — aqui o hover mostra índice, diagnóstico, confiança e z.
        """
        try:
            from utils.bokeh_map import gerar_margem_svm_html
            caminho = gerar_margem_svm_html(self._explicacoes)
            webbrowser.open(f"file://{caminho}")
            self._lbl_margem.configure(
                text="Gráfico da margem aberto no navegador.", text_color="#2ecc71")
        except ImportError:
            self._lbl_margem.configure(
                text="Instale o Bokeh para o gráfico interativo:  pip install bokeh",
                text_color="#e74c3c")
        except Exception as e:
            self._lbl_margem.configure(
                text=f"Não foi possível gerar o gráfico: {e}", text_color="#e74c3c")

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
        """Redesenha, no mapa e na margem, o paciente selecionado e seus vetores."""
        self._destacar_na_margem(chave)

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

    def _destacar_na_margem(self, chave: str):
        """Marca, no painel da margem, a posição exata do paciente selecionado."""
        for art in self._artistas_margem:
            art.remove()
        self._artistas_margem.clear()

        posicao = self._margem_xy.get(chave)
        if posicao is None:
            return
        x, y = posicao

        # Linha vertical no z do paciente: deixa explícito de que lado da
        # fronteira ele caiu e a que distância das bordas da margem.
        linha = self._ax_margem.axvline(x, color="#f1c40f", linewidth=1.0,
                                        alpha=0.6, zorder=4)
        estrela = self._ax_margem.scatter([x], [y], marker="*", s=300, c="#f1c40f",
                                          edgecolors="black", linewidths=1.0, zorder=5)

        # O rótulo vai rente à base, e não junto da estrela: no alto ele
        # esbarraria nos nomes das faixas, e a linha vertical já liga os dois.
        inicio, fim = self._ax_margem.get_xlim()
        fracao = (x - inicio) / (fim - inicio)
        alinhamento = "left" if fracao < 0.06 else ("right" if fracao > 0.94 else "center")
        rotulo = self._ax_margem.text(
            x, 0.035, f" z = {x:+.2f} ", ha=alinhamento, va="bottom",
            color="#f1c40f", fontsize=9, zorder=5,
            bbox=dict(facecolor=self.COR_FUNDO, edgecolor="none", pad=1.5, alpha=0.85))
        self._artistas_margem.extend([linha, estrela, rotulo])
        self._canvas_margem.draw_idle()

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
        linhas.append("vetores de suporte, porque decide a soma e não um único fator):")
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
            "dos 30 biomarcadores. A decisão soma a similaridade deste paciente com\n"
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
