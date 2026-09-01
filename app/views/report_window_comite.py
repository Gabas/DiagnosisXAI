"""
Módulo contendo a janela de relatório de explicabilidade do comitê.

O comitê não tem estrutura interna para abrir — a decisão dele é a média das
probabilidades calibradas dos membros. Esta janela mostra, então, o acordo entre
eles: quanto cada membro se afasta do consenso ao longo do lote e, por paciente,
onde cada um se posicionou em relação ao próprio limiar. Com a recusa ligada, é
aqui que se vê **por que** um caso foi devolvido — os membros discordaram ou
todos ficaram inseguros (ver ``core.committee``).
"""

import customtkinter as ctk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.committee import (MOTIVO_CAUTELA, MOTIVO_DISCORDANCIA, MOTIVO_FRONTEIRA,
                            MOTIVO_MAIORIA)
from utils.ui import (ScrollableFrame, ajustar_ao_conteudo, bind_treeview_mousewheel,
                      figura_responsiva, itens_visiveis, responsive_geometry)
from views.report_common import PatientPDFExportMixin


class ComiteReportWindow(ctk.CTkToplevel, PatientPDFExportMixin):
    """
    Janela de explicabilidade do comitê de voto suave.

    Reúne o painel de concordância dos membros (lote inteiro), o gráfico de
    posições do paciente selecionado e a área mestre-detalhe com o motivo de
    cada decisão.

    Attributes
    ----------
    _explicacoes : list[dict]
        Explicações por paciente produzidas por ``core.committee.explicar``.
    _membros : list[str]
        Nomes dos modelos que compõem o comitê.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"
    COR_REVISAR = "#e67e22"
    COR_FUNDO = "#2b2b2b"

    # Texto exibido na coluna "Motivo" da lista de pacientes.
    _MOTIVOS = {
        MOTIVO_DISCORDANCIA: "⚠ discordância",
        MOTIVO_FRONTEIRA: "fronteira",
        MOTIVO_CAUTELA: "cautela da política",
        MOTIVO_MAIORIA: "maioria",
        'consenso': "consenso",
    }

    def __init__(self, master, membros: list, limiares: dict, limiar_comite: float,
                 faixa, explicacoes: list, resumo: dict, **kwargs):
        """
        Inicializa a janela de relatório do comitê.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou o relatório.
        membros : list[str]
            Nomes dos modelos que compõem o comitê.
        limiares : dict[str, float]
            Limiar de operação de cada membro, em porcentagem.
        limiar_comite : float
            Limiar de operação do comitê, em porcentagem.
        faixa : list[float] ou None
            ``[inferior, superior]`` da recusa, em porcentagem, quando ligada.
        explicacoes : list[dict]
            Explicações por paciente (ver ``core.committee.explicar``).
        resumo : dict
            Estatísticas do lote: concordância por membro e contagem de motivos.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Relatório de Explicabilidade: Comitê (voto suave)")
        responsive_geometry(self, 1060, 840)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._corpo = ScrollableFrame(self, fg_color="transparent")
        self._corpo.grid(row=0, column=0, sticky="nsew")
        self._corpo.grid_columnconfigure(0, weight=1)
        self._corpo.grid_columnconfigure(1, weight=1)

        self._explicacoes = explicacoes
        self._por_indice = {str(e['indice']): e for e in explicacoes}
        self._membros = membros
        self._limiares = limiares
        self._limiar_comite = limiar_comite
        self._faixa = faixa
        self._resumo = resumo
        self._linhas_lista = itens_visiveis(self, 10, minimo=6)

        self._build_header()
        self._build_global()
        self._build_plot()
        self._build_per_patient(explicacoes)

        ajustar_ao_conteudo(self, self._corpo)
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build_header(self):
        """Constrói o cabeçalho com o resumo do lote e dos motivos."""
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
        ctk.CTkLabel(
            header,
            text=(f"Comitê de {len(self._membros)} modelos   ·   {n} paciente(s)   ·   "
                  f"Maligno: {malignos}    Benigno: {benignos}{revisar}"),
            font=ctk.CTkFont(size=13), text_color="gray",
        ).pack(anchor="w")

        motivos = self._resumo.get('motivos', {})
        if adiados:
            ctk.CTkLabel(
                header,
                text=(f"Dos {adiados} caso(s) devolvido(s): "
                      f"{motivos.get(MOTIVO_DISCORDANCIA, 0)} por discordância entre os membros, "
                      f"{motivos.get(MOTIVO_FRONTEIRA, 0)} por estarem na fronteira da decisão e "
                      f"{motivos.get(MOTIVO_CAUTELA, 0)} por cautela da política. Nestes últimos "
                      f"os membros concordavam, e quem adiou foi a largura da faixa de recusa."),
                font=ctk.CTkFont(size=12), text_color=self.COR_REVISAR,
                justify="left", wraplength=980,
            ).pack(anchor="w", pady=(4, 0))

    def _build_global(self):
        """Constrói o painel de concordância dos membros ao longo do lote."""
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Concordância de cada membro com o comitê",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            frame,
            text="Em que fração do lote o membro, sozinho e pelo próprio limiar, chegaria ao "
                 "mesmo lado que a média do comitê. Quem concorda pouco é quem mais desloca "
                 "o resultado.",
            font=ctk.CTkFont(size=11), text_color="gray", justify="left", wraplength=420,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        por_membro = self._resumo.get('por_membro', {})
        for i, nome in enumerate(self._membros, start=2):
            dados = por_membro.get(nome, {})
            taxa = dados.get('taxa')

            ctk.CTkLabel(
                frame, text=f"{nome}  (limiar {self._limiares.get(nome, 50):.0f}%)",
                anchor="w", font=ctk.CTkFont(size=12),
            ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=3)

            barra = ctk.CTkProgressBar(frame, height=14, progress_color="#8e44ad")
            barra.set((taxa or 0) / 100.0)
            barra.grid(row=i, column=1, sticky="ew", padx=8, pady=3)

            ctk.CTkLabel(
                frame, text="—" if taxa is None else f"{taxa:.0f}%", width=60, anchor="e",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).grid(row=i, column=2, sticky="e", padx=(8, 16), pady=3)

        ctk.CTkLabel(
            frame,
            text="Nenhum membro decide sozinho: a média é sempre dos quatro. A concordância "
                 "mede alinhamento, não peso: todos pesam igual.",
            font=ctk.CTkFont(size=10), text_color="gray", justify="left", wraplength=420,
        ).grid(row=len(self._membros) + 2, column=0, columnspan=3,
               sticky="w", padx=16, pady=(6, 12))

    def _build_plot(self):
        """Constrói o painel do gráfico de posições dos membros (por paciente)."""
        frame = ctk.CTkFrame(self._corpo)
        frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="Posição de cada membro (paciente selecionado)",
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

        legenda = ("Cada barra é a P(Maligno) de um membro; o losango é o limiar dele. "
                   "A linha amarela é a média (a decisão do comitê).")
        if self._faixa:
            legenda += " A faixa laranja é a zona de recusa."
        ctk.CTkLabel(
            frame, text=legenda, font=ctk.CTkFont(size=10), text_color="gray",
            justify="left", wraplength=420,
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 10))

    def _desenhar_posicoes(self, e: dict):
        """
        Desenha onde cada membro se posicionou para um paciente.

        Cada membro aparece com sua probabilidade e com o próprio limiar
        marcado: é a distância entre os dois — e não a probabilidade crua — que
        diz se aquele membro está convicto, já que cada modelo opera num corte
        diferente.

        Parameters
        ----------
        e : dict
            Explicação individual produzida por ``core.committee.explicar``.
        """
        detalhes = e['membros']
        nomes = [d['nome'] for d in detalhes]
        valores = [d['probabilidade'] for d in detalhes]
        posicoes = list(range(len(detalhes)))

        self._ax.clear()

        if self._faixa:
            self._ax.axhspan(self._faixa[0], self._faixa[1], color=self.COR_REVISAR,
                             alpha=0.16, zorder=1)

        cores = [self.COR_MALIGNO if d['classe'] == 'Maligno' else self.COR_BENIGNO
                 for d in detalhes]
        self._ax.bar(posicoes, valores, width=0.55, color=cores, edgecolor="none", zorder=2)
        self._ax.scatter(posicoes, [d['limiar'] for d in detalhes], marker="D", s=46,
                         color="white", edgecolors="black", linewidths=0.6, zorder=4)

        media = e['probabilidade']
        self._ax.axhline(media, color="#f1c40f", linewidth=1.8, zorder=5,
                         label=f"comitê = {media:.0f}%")
        self._ax.axhline(self._limiar_comite, color="gray", linewidth=0.9,
                         linestyle="--", zorder=3,
                         label=f"limiar do comitê = {self._limiar_comite:.0f}%")

        self._ax.set_ylim(0, 100)
        self._ax.set_xticks(posicoes)
        self._ax.set_xticklabels([n.split()[0] for n in nomes], color="gray", fontsize=8)
        self._ax.set_ylabel("P(Maligno) calibrada (%)", color="gray", fontsize=9)
        self._ax.tick_params(colors="gray", labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color("gray")
        self._ax.legend(facecolor=self.COR_FUNDO, edgecolor="gray",
                        labelcolor="white", fontsize=8, loc="upper right")
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

        colunas = ("paciente", "diagnostico", "voto", "motivo")
        self._tree = ttk.Treeview(tree_frame, columns=colunas, show="headings",
                                  height=self._linhas_lista)
        self._tree.heading("paciente", text="Paciente")
        self._tree.heading("diagnostico", text="Diagnóstico")
        self._tree.heading("voto", text="Voto (M/B)")
        self._tree.heading("motivo", text="Motivo")
        self._tree.column("paciente", width=70, anchor="center", stretch=False)
        self._tree.column("diagnostico", width=90, anchor="center", stretch=False)
        self._tree.column("voto", width=80, anchor="center", stretch=False)
        self._tree.column("motivo", width=110, anchor="w")
        self._tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(tree_frame, command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        bind_treeview_mousewheel(self._tree)

        self._tree.tag_configure("Maligno", foreground=self.COR_MALIGNO)
        self._tree.tag_configure("Benigno", foreground=self.COR_BENIGNO)
        self._tree.tag_configure("Revisar", foreground=self.COR_REVISAR)

        for e in explicacoes:
            self._tree.insert(
                "", "end", iid=str(e['indice']),
                values=(e['indice'], e['classe'],
                        f"{e['votos_maligno']} / {e['votos_benigno']}",
                        self._MOTIVOS.get(e['motivo'], e['motivo'])),
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
            "Selecione um paciente na lista para ver como os membros do comitê se "
            "posicionaram e o que motivou a decisão.",
        )
        self._detalhe.configure(state="disabled")

        if explicacoes:
            primeiro = str(explicacoes[0]['indice'])
            self._tree.selection_set(primeiro)
            self._tree.focus(primeiro)

    def _on_select(self, _event=None):
        """Atualiza o gráfico e o detalhe conforme o paciente selecionado."""
        selecao = self._tree.selection()
        if not selecao:
            return
        explicacao = self._por_indice.get(selecao[0])
        if not explicacao:
            return
        self._desenhar_posicoes(explicacao)
        self._detalhe.configure(state="normal")
        self._detalhe.delete("1.0", "end")
        self._detalhe.insert("1.0", self._formatar_detalhe(explicacao))
        self._detalhe.configure(state="disabled")

    def _formatar_detalhe(self, e: dict) -> str:
        """
        Monta o texto explicativo da decisão do comitê para um paciente.

        Parameters
        ----------
        e : dict
            Explicação individual produzida por ``core.committee.explicar``.

        Returns
        -------
        str
            Texto com a decisão, a posição de cada membro e o motivo.
        """
        linhas = [
            f"PACIENTE {e['indice']}",
            f"Decisão do comitê: {e['classe']}  (média das probabilidades = "
            f"{e['probabilidade']:.1f}%)",
            "",
            "Como cada membro se posicionou:",
        ]
        for d in sorted(e['membros'], key=lambda x: x['probabilidade'], reverse=True):
            seta = "↑" if d['margem'] >= 0 else "↓"
            convicto = "convicto" if d['convicto'] else "em cima do corte"
            linhas.append(
                f"   • {d['nome']:22s} {d['probabilidade']:5.1f}%   "
                f"{seta} {abs(d['margem']):4.1f} pp do próprio limiar "
                f"({d['limiar']:.0f}%)  ·  {d['classe']}, {convicto}"
            )

        linhas.append("")
        linhas.append(f"Dispersão entre os membros: {e['amplitude']:.1f} pontos percentuais "
                      f"entre o mais alto e o mais baixo.")
        linhas.append("")
        linhas.extend(self._explicar_motivo(e))
        return "\n".join(linhas)

    def _explicar_motivo(self, e: dict) -> list:
        """
        Traduz o motivo da decisão em uma recomendação de leitura.

        É a parte que a média sozinha não dá: dois casos com a mesma
        probabilidade final podem ter chegado lá por caminhos opostos.
        """
        discordantes = ", ".join(e['discordantes']) or "nenhum"

        if e['motivo'] == MOTIVO_DISCORDANCIA:
            return [
                "POR QUE FOI DEVOLVIDO: os membros discordaram entre si.",
                f"   Divisão de {e['votos_maligno']} contra {e['votos_benigno']}, com "
                f"{e['amplitude']:.1f} pp entre o mais alto e o mais baixo. A média caiu na "
                f"zona de recusa por cancelamento, não por ignorância.",
                f"   Divergiram da inclinação do comitê: {discordantes}.",
                "   Vale olhar o relatório individual de quem discordou: o desacordo costuma "
                "apontar um perfil que um dos modelos reconhece e os outros não.",
            ]
        if e['motivo'] == MOTIVO_FRONTEIRA:
            return [
                "POR QUE FOI DEVOLVIDO: a média ficou em cima do limiar de decisão.",
                f"   {e['probabilidade']:.1f}% contra um limiar de {self._limiar_comite:.1f}%. "
                f"um deslocamento pequeno inverteria a resposta. É o caso genuinamente "
                f"indeciso, e a decisão é clínica: nenhum modelo deste app vai resolvê-lo.",
            ]
        if e['motivo'] == MOTIVO_CAUTELA:
            return [
                "POR QUE FOI DEVOLVIDO: cautela da política de operação, não dúvida do modelo.",
                f"   Os membros concordaram entre si (dispersão de apenas {e['amplitude']:.1f} pp) "
                f"e a média ({e['probabilidade']:.1f}%) está longe do limiar de "
                f"{self._limiar_comite:.1f}%. O caso só foi devolvido porque a faixa de recusa "
                f"vai de {self._faixa[0]:.1f}% a {self._faixa[1]:.1f}%.",
                "   Essa faixa foi calibrada para não errar nenhum caso do treino, e por isso é "
                "definida pelos dois pacientes mais atípicos que ele continha. É larga o "
                "bastante para alcançar casos como este. Se a fila de revisão for grande demais, "
                "é aqui que ela pode ser encurtada, aceitando algum risco.",
            ]
        if e['motivo'] == MOTIVO_MAIORIA:
            return [
                f"DECISÃO POR MAIORIA: {e['votos_maligno']} membro(s) apontaram Maligno e "
                f"{e['votos_benigno']}, Benigno.",
                f"   Divergiram do resultado: {discordantes}.",
                "   A média pendeu para o lado da maioria, mas o desacordo é um sinal de "
                "cautela: confirme com o relatório de quem discordou.",
            ]
        return [
            "DECISÃO POR CONSENSO: todos os membros apontaram o mesmo lado.",
            "   É o cenário mais confiável do comitê: o acerto não depende de um modelo só.",
        ]

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