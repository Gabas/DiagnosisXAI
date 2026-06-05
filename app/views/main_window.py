"""
Módulo responsável pela interface principal e navegação do sistema.
"""

import customtkinter as ctk
from views.dashboard_view import DashboardView
from views.predict_view import PredictView
# from views.history_view import HistoryView

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
        Instância do frame referente à tela de Dashboard.
    predict_view : PredictView
        Instância do frame referente à tela de Novo Diagnóstico.
    """

    def __init__(self):
        """
        Inicializa a janela principal, definindo dimensões, grid 
        e instanciando os componentes da interface.
        """
        super().__init__()

        # Configurações da Janela
        self.title("Sistema de Diagnóstico XAI")
        self.geometry("1000x700")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._create_sidebar()
        self._create_views()

        # Seleciona a tela inicial
        self.select_frame_by_name("dashboard")

    def _create_sidebar(self):
        """
        Cria e posiciona o menu lateral e seus botões de navegação.
        """
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Diagnosis XAI\n🎗️", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.btn_dashboard = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=40, border_spacing=10, 
                                           text="Dashboard", fg_color="transparent", text_color=("gray10", "gray90"),
                                           hover_color=("gray70", "gray30"), anchor="w", 
                                           command=lambda: self.select_frame_by_name("dashboard"))
        self.btn_dashboard.grid(row=1, column=0, sticky="ew")

        self.btn_predict = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=40, border_spacing=10, 
                                         text="Novo Diagnóstico", fg_color="transparent", text_color=("gray10", "gray90"),
                                         hover_color=("gray70", "gray30"), anchor="w", 
                                         command=lambda: self.select_frame_by_name("predict"))
        self.btn_predict.grid(row=2, column=0, sticky="ew")
        # self.btn_history = ctk.CTkButton(...)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Tema:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                            command=self.change_appearance_mode_event)
        self.appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10, 20))

    def _create_views(self):
        """
        Instancia as telas (frames) que serão exibidas na área central.
        """
        self.dashboard_view = DashboardView(self)
        self.predict_view = PredictView(self)
        # self.history_view = HistoryView(self)

    def select_frame_by_name(self, name: str):
        """
        Alterna a visualização da tela central baseada no nome informado.

        Parameters
        ----------
        name : str
            Identificador da tela a ser exibida ('dashboard' ou 'predict').
        """
        self.btn_dashboard.configure(fg_color=("gray75", "gray25") if name == "dashboard" else "transparent")
        self.btn_predict.configure(fg_color=("gray75", "gray25") if name == "predict" else "transparent")
        
        self.dashboard_view.grid_forget()
        self.predict_view.grid_forget() 

        if name == "dashboard":
            self.dashboard_view.grid(row=0, column=1, sticky="nsew")
        elif name == "predict": 
            self.predict_view.grid(row=0, column=1, sticky="nsew")
            
    def change_appearance_mode_event(self, new_appearance_mode: str):
        """
        Altera o tema visual de toda a aplicação.

        Parameters
        ----------
        new_appearance_mode : str
            O modo de aparência escolhido ('Dark', 'Light' ou 'System').
        """
        ctk.set_appearance_mode(new_appearance_mode)