"""
Módulo responsável pela interface principal e navegação do sistema.
"""

import customtkinter as ctk
from utils.ui import responsive_geometry
from views.dashboard_view import DashboardView
from views.history_view import HistoryView
from views.predict_view import PredictView
from views.about_view import AboutView


class MainWindow(ctk.CTk):
    """
    Janela principal do Sistema de Diagnóstico XAI.

    Gerencia o menu lateral (sidebar) e atua como o contêiner central
    que alterna entre as diferentes telas (views) da aplicação.

    Attributes
    ----------
    sidebar_frame : ctk.CTkFrame
        Frame lateral contendo os botões de navegação e configurações.
    dashboard_view : DashboardView
        Instância do frame referente à tela inicial.
    predict_view : PredictView
        Instância do frame referente à tela de Novo Diagnóstico.
    history_view : HistoryView
        Instância do frame referente à aba de histórico de sessões.
    about_view : AboutView
        Instância do frame referente à aba de informações do projeto.
    """

    def __init__(self):
        """
        Inicializa a janela principal, definindo dimensões, grid
        e instanciando os componentes da interface.
        """
        super().__init__()

        self.title("DiagnosisXAI")
        responsive_geometry(self, 1000, 700)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._create_sidebar()
        self._create_views()

        self.select_frame_by_name("dashboard")

    def _create_sidebar(self):
        """
        Cria e posiciona o menu lateral e seus botões de navegação.
        """
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="DiagnosisXAI",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(24, 24))

        nav_btn_cfg = dict(
            corner_radius=0, height=40, border_spacing=10,
            fg_color="transparent", text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"), anchor="w",
        )

        self.btn_dashboard = ctk.CTkButton(
            self.sidebar_frame, text="Início", **nav_btn_cfg,
            command=lambda: self.select_frame_by_name("dashboard"),
        )
        self.btn_dashboard.grid(row=1, column=0, sticky="ew")

        self.btn_predict = ctk.CTkButton(
            self.sidebar_frame, text="Novo Diagnóstico", **nav_btn_cfg,
            command=lambda: self.select_frame_by_name("predict"),
        )
        self.btn_predict.grid(row=2, column=0, sticky="ew")

        self.btn_history = ctk.CTkButton(
            self.sidebar_frame, text="Histórico", **nav_btn_cfg,
            command=lambda: self.select_frame_by_name("history"),
        )
        self.btn_history.grid(row=3, column=0, sticky="ew")

        self.btn_about = ctk.CTkButton(
            self.sidebar_frame, text="Sobre", **nav_btn_cfg,
            command=lambda: self.select_frame_by_name("about"),
        )
        self.btn_about.grid(row=4, column=0, sticky="ew")

        # Spacer empurra os controles de tema para o fundo
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar_frame, text="Tema:", anchor="w"
        )
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_optionmenu.grid(row=7, column=0, padx=20, pady=(10, 20))

    def _create_views(self):
        """
        Instancia as telas (frames) que serão exibidas na área central.
        """
        self.dashboard_view = DashboardView(self)
        self.predict_view = PredictView(self)
        self.history_view = HistoryView(self)
        self.about_view = AboutView(self)

    def select_frame_by_name(self, name: str):
        """
        Alterna a visualização da tela central baseada no nome informado.

        Parameters
        ----------
        name : str
            Identificador da tela a ser exibida ('dashboard', 'predict', 'history' ou 'about').
        """
        active_color = ("gray75", "gray25")

        self.btn_dashboard.configure(fg_color=active_color if name == "dashboard" else "transparent")
        self.btn_predict.configure(fg_color=active_color if name == "predict"     else "transparent")
        self.btn_history.configure(fg_color=active_color if name == "history"     else "transparent")
        self.btn_about.configure(fg_color=active_color if name == "about"         else "transparent")

        self.dashboard_view.grid_forget()
        self.predict_view.grid_forget()
        self.history_view.grid_forget()
        self.about_view.grid_forget()

        if name == "dashboard":
            self.dashboard_view.grid(row=0, column=1, sticky="nsew")
        elif name == "predict":
            self.predict_view.grid(row=0, column=1, sticky="nsew")
        elif name == "history":
            self.history_view.refresh()
            self.history_view.grid(row=0, column=1, sticky="nsew")
        elif name == "about":
            self.about_view.grid(row=0, column=1, sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        """
        Altera o tema visual de toda a aplicação.

        Parameters
        ----------
        new_appearance_mode : str
            O modo de aparência escolhido ('Dark', 'Light' ou 'System').
        """
        ctk.set_appearance_mode(new_appearance_mode)
