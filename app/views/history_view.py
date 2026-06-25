"""
Módulo contendo a aba de histórico de sessões de diagnóstico.
"""

import customtkinter as ctk
from tkinter import messagebox
from core.history_manager import HistoryManager


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

        self._scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
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
            self._build_card(self._scroll, entry, row=i)

    def _build_card(self, parent, entry: dict, row: int):
        """
        Constrói e posiciona o card de uma única sessão de diagnóstico.

        Parameters
        ----------
        parent : ctk.CTkScrollableFrame
            Frame rolável onde o card será inserido.
        entry : dict
            Dicionário com os dados da sessão (timestamp, arquivo, modelo, etc.).
        row : int
            Linha da grade do frame pai onde o card será posicionado.
        """
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 10))

        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=12)

        # Linha 1: timestamp · nome do arquivo · total de pacientes
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
