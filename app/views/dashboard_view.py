"""
Módulo contendo a tela inicial simplificada do sistema.
"""

import customtkinter as ctk


class DashboardView(ctk.CTkFrame):
    """
    Frame responsável pela tela inicial do aplicativo.

    Apresenta o nome do sistema, uma breve descrição, o status dos
    modelos carregados e um botão de acesso rápido ao diagnóstico.
    """

    def __init__(self, master, **kwargs):
        """
        Inicializa o frame e armazena a referência à janela principal.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            Janela principal que contém este frame.
        **kwargs
            Argumentos adicionais passados para o construtor do CTkFrame.
        """
        super().__init__(master, corner_radius=0, fg_color="transparent", **kwargs)
        self._main_window = master
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói os elementos visuais centralizados da tela inicial.
        """
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            center,
            text="DiagnosisXAI",
            font=ctk.CTkFont(size=54, weight="bold"),
        ).pack(pady=(0, 10))

        ctk.CTkLabel(
            center,
            text="Diagnóstico preditivo de câncer de mama\ncom Inteligência Artificial Explicável",
            font=ctk.CTkFont(size=16),
            text_color="gray",
            justify="center",
        ).pack(pady=(0, 40))

        status_frame = ctk.CTkFrame(center, fg_color="transparent")
        status_frame.pack(pady=(0, 48))
        ctk.CTkLabel(
            status_frame, text="●", text_color="#27ae60", font=ctk.CTkFont(size=16)
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            status_frame,
            text="5 modelos prontos  ·  30 biomarcadores ativos  ·  Base Wisconsin (WDBC)",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(side="left")

        ctk.CTkButton(
            center,
            text="Iniciar Novo Diagnóstico",
            width=240,
            height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=lambda: self._main_window.select_frame_by_name("predict"),
        ).pack()
