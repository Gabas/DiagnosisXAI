# app/views/main_window.py
import customtkinter as ctk
from views.dashboard_view import DashboardView
# Importaremos as outras views conforme formos criando:
# from views.predict_view import PredictView
# from views.history_view import HistoryView

class MainWindow(ctk.CTk):
    def __init__(self):
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

        # Botões comentados até criarmos as outras views:
        # self.btn_predict = ctk.CTkButton(...)
        # self.btn_history = ctk.CTkButton(...)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Tema:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                            command=self.change_appearance_mode_event)
        self.appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10, 20))

    def _create_views(self):
        # Inicializa o frame do Dashboard (agora isolado em outra classe)
        self.dashboard_view = DashboardView(self)
        
        # self.predict_view = PredictView(self)
        # self.history_view = HistoryView(self)

    def select_frame_by_name(self, name):
        self.btn_dashboard.configure(fg_color=("gray75", "gray25") if name == "dashboard" else "transparent")
        
        self.dashboard_view.grid_forget()

        if name == "dashboard":
            self.dashboard_view.grid(row=0, column=1, sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)