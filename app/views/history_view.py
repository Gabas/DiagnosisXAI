"""
Módulo contendo a aba de histórico de sessões de diagnóstico.
"""

import customtkinter as ctk
from tkinter import messagebox
from core.history_manager import HistoryManager
from utils.ui import ScrollableFrame
from views import report_launchers
from views.report_window import ReportWindow
from views.report_window_lr import LogisticReportWindow
from views.report_window_knn import KNNReportWindow
from views.report_window_rf import RandomForestReportWindow


class HistoryView(ctk.CTkFrame):
    """
    Frame responsável por exibir e gerenciar o histórico de sessões de diagnóstico.

    Apresenta um card por sessão registrada, com data, arquivo, modelo utilizado,
    distribuição de diagnósticos e acurácia (quando a auditoria foi executada).

    Attributes
    ----------
    _manager : HistoryManager
        Instância responsável pela leitura e escrita do histórico em disco.
    _lbl_count : ctk.CTkLabel
        Rótulo que exibe o número total de sessões registradas.
    _btn_clear : ctk.CTkButton
        Botão para apagar todo o histórico após confirmação.
    _scroll : ctk.CTkScrollableFrame
        Área rolável onde os cards de sessão são renderizados.
    """

    def __init__(self, master, **kwargs):
        """
        Inicializa o frame do histórico e constrói a interface base.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Janela principal que contém este frame.
        **kwargs
            Argumentos adicionais passados para o construtor do CTkFrame.
        """
        super().__init__(master, corner_radius=0, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._manager = HistoryManager()
        self._detail_window = None
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói os elementos estáticos da interface: cabeçalho e área rolável.
        Os cards de sessão são populados dinamicamente em refresh().
        """
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 0))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Histórico de Diagnósticos",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self._lbl_count = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(size=13), text_color="gray",
        )
        self._lbl_count.grid(row=1, column=0, sticky="w")

        self._btn_clear = ctk.CTkButton(
            header, text="Limpar Histórico", width=160,
            fg_color="#c0392b", hover_color="#e74c3c",
            command=self._confirm_clear,
        )
        self._btn_clear.grid(row=0, column=1, rowspan=2, sticky="e")

        self._scroll = ScrollableFrame(self, fg_color="transparent")
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=24, pady=16)
        self._scroll.grid_columnconfigure(0, weight=1)

    def refresh(self):
        """
        Recarrega o histórico do disco e reconstrói a lista de cards.
        Deve ser chamado sempre que a aba for exibida para garantir dados atualizados.
        """
        for widget in self._scroll.winfo_children():
            widget.destroy()

        entries = self._manager.load()

        if not entries:
            self._lbl_count.configure(text="Nenhuma sessão registrada")
            self._btn_clear.configure(state="disabled")
            ctk.CTkLabel(
                self._scroll,
                text="Nenhuma sessão registrada ainda.\nExecute um diagnóstico para começar.",
                font=ctk.CTkFont(size=14), text_color="gray", justify="center",
            ).grid(row=0, column=0, pady=60)
            return

        count = len(entries)
        plural = "s" if count > 1 else ""
        self._lbl_count.configure(text=f"{count} sessão{plural} registrada{plural}")
        self._btn_clear.configure(state="normal")

        for i, entry in enumerate(entries):
            self._build_card(self._scroll, entry, index=i)

    def _build_card(self, parent, entry: dict, index: int):
        """
        Constrói e posiciona o card de uma única sessão de diagnóstico.

        O card inteiro é clicável (abre os detalhes da sessão), exceto o botão
        de exclusão, que remove apenas aquela consulta do histórico.

        Parameters
        ----------
        parent : ctk.CTkScrollableFrame
            Frame rolável onde o card será inserido.
        entry : dict
            Dicionário com os dados da sessão (timestamp, arquivo, modelo, etc.).
        index : int
            Posição da sessão na lista carregada — usada tanto para o
            posicionamento na grade quanto para a exclusão individual.
        """
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.grid(row=index, column=0, sticky="ew", pady=(0, 10))

        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=12)

        # Linha 1: timestamp · nome do arquivo · [✕] · total de pacientes
        row1 = ctk.CTkFrame(content, fg_color="transparent")
        row1.pack(fill="x")

        ctk.CTkLabel(
            row1, text=entry.get('timestamp', '—'),
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(side="left")

        ctk.CTkLabel(
            row1, text=f"  ·  {entry.get('arquivo', '—')}",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        btn_delete = ctk.CTkButton(
            row1, text="✕", width=28, height=28,
            fg_color="transparent", hover_color="#c0392b",
            text_color=("gray30", "gray70"),
            command=lambda i=index: self._confirm_delete_one(i),
        )
        btn_delete.pack(side="right", padx=(8, 0))

        total = entry.get('total', 0)
        ctk.CTkLabel(
            row1, text=f"{total} paciente{'s' if total != 1 else ''}",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(side="right")

        # Linha 2: nome do modelo
        ctk.CTkLabel(
            content, text=f"Modelo: {entry.get('modelo', '—')}",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", pady=(6, 0))

        # Linha 3: distribuição Maligno / Benigno (apenas para modelo único)
        malignos = entry.get('malignos')
        benignos = entry.get('benignos')
        if malignos is not None and benignos is not None:
            row3 = ctk.CTkFrame(content, fg_color="transparent")
            row3.pack(fill="x", pady=(2, 0))

            ctk.CTkLabel(
                row3, text=f"Maligno: {malignos}",
                font=ctk.CTkFont(size=12), text_color="#e74c3c",
            ).pack(side="left")

            ctk.CTkLabel(
                row3, text="  |  ",
                font=ctk.CTkFont(size=12), text_color="gray",
            ).pack(side="left")

            ctk.CTkLabel(
                row3, text=f"Benigno: {benignos}",
                font=ctk.CTkFont(size=12), text_color="#2ecc71",
            ).pack(side="left")

        # Linha 4: acurácia (disponível somente após a etapa de auditoria)
        acuracia = entry.get('acuracia')
        if acuracia:
            acc_frame = ctk.CTkFrame(content, fg_color=("gray85", "gray20"), corner_radius=6)
            acc_frame.pack(fill="x", pady=(8, 0))

            acc_text = "  ·  ".join(f"{k}: {v:.2f}%" for k, v in acuracia.items())
            ctk.CTkLabel(
                acc_frame, text=f"Acurácia — {acc_text}",
                font=ctk.CTkFont(size=12), text_color="#2ecc71",
            ).pack(anchor="w", padx=10, pady=6)

        # Torna todo o card clicável para abrir os detalhes, exceto o ✕ (excluir).
        self._bind_click(card, lambda _e, en=entry: self._open_details(en), skip=btn_delete)

    def _bind_click(self, widget, handler, skip):
        """
        Vincula um clique (Button-1) a um widget e a todos os seus descendentes.

        Parameters
        ----------
        widget : tkinter widget
            Widget raiz a partir do qual o vínculo é propagado.
        handler : callable
            Função chamada no clique, recebendo o evento como argumento.
        skip : tkinter widget
            Widget (e sua subárvore) a ser ignorado — tipicamente o botão excluir.
        """
        if widget is skip:
            return
        widget.bind("<Button-1>", handler)
        for child in widget.winfo_children():
            self._bind_click(child, handler, skip)

    def _confirm_delete_one(self, index: int):
        """
        Pede confirmação e exclui uma única consulta do histórico.

        Parameters
        ----------
        index : int
            Posição da sessão na lista carregada.
        """
        if messagebox.askyesno(
            "Excluir Consulta",
            "Deseja excluir esta consulta do histórico?\nEsta ação não pode ser desfeita.",
        ):
            self._manager.delete(index)
            self.refresh()

    def _open_details(self, entry: dict):
        """
        Abre (ou recria) a janela com os detalhes completos de uma sessão.

        Parameters
        ----------
        entry : dict
            Dados da sessão selecionada.
        """
        if self._detail_window is not None and self._detail_window.winfo_exists():
            self._detail_window.destroy()
        self._detail_window = SessionDetailWindow(self, entry)

    def _confirm_clear(self):
        """
        Exibe uma caixa de diálogo de confirmação antes de apagar o histórico.
        O histórico só é removido se o utilizador confirmar a ação.
        """
        if messagebox.askyesno(
            "Limpar Histórico",
            "Tem certeza que deseja apagar todo o histórico de sessões?\nEsta ação não pode ser desfeita.",
        ):
            self._manager.clear()
            self.refresh()


class SessionDetailWindow(ctk.CTkToplevel):
    """
    Janela secundária com os detalhes completos de uma sessão de diagnóstico.

    Exibe os metadados da consulta (data, arquivo, modelo, total), a
    distribuição de diagnósticos e a acurácia por modelo, quando a etapa de
    auditoria foi executada.
    """

    COR_MALIGNO = "#e74c3c"
    COR_BENIGNO = "#2ecc71"

    # Tipos de relatório de explicabilidade -> (rótulo, classe da janela).
    _TIPOS = {
        'arvore': ("Árvore de Decisão", ReportWindow),
        'logistica': ("Regressão Logística", LogisticReportWindow),
        'knn': ("KNN", KNNReportWindow),
        'randomforest': ("Random Forest", RandomForestReportWindow),
    }

    def __init__(self, master, entry: dict, **kwargs):
        """
        Inicializa a janela de detalhes.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Widget que originou a abertura dos detalhes.
        entry : dict
            Dados da sessão a serem exibidos.
        **kwargs
            Argumentos adicionais para o construtor do CTkToplevel.
        """
        super().__init__(master, **kwargs)
        self.title("Detalhes da Sessão")
        self.geometry("480x620")
        self.grid_columnconfigure(0, weight=1)
        self._entry = entry
        self._report_windows = {}
        self._loader = None
        self._shap_cache = {}
        self._build(entry)
        self.after(150, self.lift)
        self.after(200, self.focus)

    def _build(self, entry: dict):
        """Constrói o conteúdo da janela a partir dos dados da sessão."""
        wrapper = ScrollableFrame(self, fg_color="transparent")
        wrapper.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            wrapper, text="Detalhes da Sessão",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", pady=(0, 12))

        # --- Metadados principais ---
        info = ctk.CTkFrame(wrapper)
        info.pack(fill="x", pady=(0, 12))
        campos = [
            ("Data e hora", entry.get('timestamp', '—')),
            ("Arquivo", entry.get('arquivo', '—')),
            ("Modelo", entry.get('modelo', '—')),
            ("Total de pacientes", str(entry.get('total', '—'))),
        ]
        for rotulo, valor in campos:
            linha = ctk.CTkFrame(info, fg_color="transparent")
            linha.pack(fill="x", padx=14, pady=4)
            ctk.CTkLabel(
                linha, text=f"{rotulo}:", width=150, anchor="w",
                font=ctk.CTkFont(size=13, weight="bold"),
            ).pack(side="left")
            ctk.CTkLabel(
                linha, text=valor, anchor="w", justify="left",
                font=ctk.CTkFont(size=13), wraplength=270,
            ).pack(side="left", padx=(8, 0))

        # --- Distribuição (apenas para modelo único) ---
        malignos = entry.get('malignos')
        benignos = entry.get('benignos')
        if malignos is not None and benignos is not None:
            dist = ctk.CTkFrame(wrapper)
            dist.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(
                dist, text="Distribuição dos diagnósticos",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(anchor="w", padx=14, pady=(10, 6))
            barra = ctk.CTkFrame(dist, fg_color="transparent")
            barra.pack(fill="x", padx=14, pady=(0, 10))
            ctk.CTkLabel(
                barra, text=f"Maligno: {malignos}", text_color=self.COR_MALIGNO,
                font=ctk.CTkFont(size=13, weight="bold"),
            ).pack(side="left")
            ctk.CTkLabel(
                barra, text=f"Benigno: {benignos}", text_color=self.COR_BENIGNO,
                font=ctk.CTkFont(size=13, weight="bold"),
            ).pack(side="left", padx=(20, 0))

        # --- Acurácia (apenas após auditoria) ---
        acc = ctk.CTkFrame(wrapper)
        acc.pack(fill="x")
        ctk.CTkLabel(
            acc, text="Acurácia (auditoria)",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(10, 6))

        acuracia = entry.get('acuracia')
        if acuracia:
            for modelo, valor in acuracia.items():
                linha = ctk.CTkFrame(acc, fg_color="transparent")
                linha.pack(fill="x", padx=14, pady=2)
                ctk.CTkLabel(
                    linha, text=modelo, anchor="w", font=ctk.CTkFont(size=13),
                ).pack(side="left")
                ctk.CTkLabel(
                    linha, text=f"{valor:.2f}%", anchor="e", text_color="#2ecc71",
                    font=ctk.CTkFont(size=13, weight="bold"),
                ).pack(side="right")
            ctk.CTkFrame(acc, height=6, fg_color="transparent").pack()
        else:
            ctk.CTkLabel(
                acc, text="Auditoria não executada nesta sessão.",
                text_color="gray", font=ctk.CTkFont(size=12),
            ).pack(anchor="w", padx=14, pady=(0, 10))

        # --- Relatórios de explicabilidade ---
        relatorios = self._normalizar_relatorio(entry.get('relatorio'))

        # Relatórios exatos (um botão por modelo interpretável).
        for tipo, (rotulo, _) in self._TIPOS.items():
            if tipo in relatorios:
                ctk.CTkButton(
                    wrapper,
                    text=f"Ver Relatório de Explicabilidade — {rotulo}",
                    fg_color="#2980b9", hover_color="#3498db",
                    command=lambda t=tipo: self._abrir_relatorio(t),
                ).pack(fill="x", pady=(14, 0))

        # SHAP: um botão por modelo cujo lote foi salvo (reabre interativo).
        shap = relatorios.get('shap')
        if shap:
            for key in shap.get('modelos', []):
                nome = report_launchers.MODELO_POR_KEY.get(key, key)
                ctk.CTkButton(
                    wrapper, text=f"Ver Relatório SHAP — {nome}",
                    fg_color="#8e44ad", hover_color="#9b59b6",
                    command=lambda k=key: self._abrir_shap(k),
                ).pack(fill="x", pady=(8, 0))

        # UMAP: mapa populacional do lote (reabre interativo).
        if 'umap' in relatorios:
            ctk.CTkButton(
                wrapper, text="Ver Mapa Populacional — UMAP",
                fg_color="#16a085", hover_color="#1abc9c",
                command=self._abrir_umap,
            ).pack(fill="x", pady=(8, 0))

    @staticmethod
    def _normalizar_relatorio(relatorio) -> dict:
        """
        Normaliza o campo 'relatorio' para o formato por modelo.

        Aceita o formato atual (chaves por modelo: 'arvore'/'logistica'/'knn')
        e o formato antigo (relatório plano da árvore, com
        'importancias'/'explicacoes' na raiz).

        Parameters
        ----------
        relatorio : dict ou None
            Conteúdo do campo 'relatorio' da sessão.

        Returns
        -------
        dict
            Mapa {tipo: dados_do_relatorio}, possivelmente vazio.
        """
        if not relatorio:
            return {}
        # Reconhece relatórios exatos (Árvore/LR/KNN/RF) e também SHAP/UMAP.
        chaves = (*SessionDetailWindow._TIPOS, 'shap', 'umap')
        if any(chave in relatorio for chave in chaves):
            return relatorio
        if 'explicacoes' in relatorio:  # formato antigo: somente a árvore
            return {'arvore': relatorio}
        return {}

    def _obter_loader(self):
        """
        Devolve um ModelLoader, criando-o sob demanda (lazy) na 1ª necessidade.

        Usado apenas ao reabrir SHAP/UMAP, que precisam do modelo, do background
        e do embedding de treino — evitando carregar o .pkl ao abrir os detalhes.

        Returns
        -------
        ModelLoader
            Carregador com os artefatos do wisconsin.pkl.
        """
        if self._loader is None:
            from core.inference import ModelLoader
            self._loader = ModelLoader()
        return self._loader

    def _abrir_relatorio(self, tipo: str):
        """
        Abre (ou recria) a janela do relatório exato do tipo indicado.

        Parameters
        ----------
        tipo : str
            'arvore', 'logistica', 'knn' ou 'randomforest'.
        """
        relatorios = self._normalizar_relatorio(self._entry.get('relatorio'))
        dados = relatorios.get(tipo)
        if not dados:
            return
        self._fechar_janela(tipo)
        _, Classe = self._TIPOS[tipo]
        self._report_windows[tipo] = Classe(self, **dados)

    def _abrir_shap(self, key: str):
        """
        Reabre, interativamente, o relatório SHAP de um modelo salvo na sessão.

        Reconstrói o explicador a partir do modelo carregado e reaproveita o
        lote (padronizado/bruto) persistido — o cálculo por paciente segue lazy.

        Parameters
        ----------
        key : str
            Chave do modelo ('dt', 'rf', 'lr', 'svm' ou 'knn').
        """
        shap = self._normalizar_relatorio(self._entry.get('relatorio')).get('shap')
        if not shap:
            return
        self._fechar_janela(f'shap_{key}')
        self._report_windows[f'shap_{key}'] = report_launchers.abrir_shap(
            self, self._obter_loader(), key,
            shap['X_scaled'], shap['X_raw'], shap['indices'], self._shap_cache)

    def _abrir_umap(self):
        """Reabre o Mapa Populacional (UMAP) do lote salvo na sessão."""
        umap = self._normalizar_relatorio(self._entry.get('relatorio')).get('umap')
        if not umap:
            return
        self._fechar_janela('umap')
        self._report_windows['umap'] = report_launchers.abrir_umap(
            self, self._obter_loader(), umap['batch_2d'], umap['pacientes'])

    def _fechar_janela(self, chave: str):
        """Fecha a janela de relatório previamente aberta com a chave indicada."""
        janela = self._report_windows.get(chave)
        if janela is not None and janela.winfo_exists():
            janela.destroy()
