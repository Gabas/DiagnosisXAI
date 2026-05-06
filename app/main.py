import customtkinter as ctk
from tkinter import filedialog
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green") 

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configurações da Janela
        self.title("Sistema de Diagnóstico XAI")
        self.geometry("1000x700")
        
        # Responsividade do layout principal
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ==================== MENU LATERAL ====================
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Diagnosis XAI\n🎗️", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Botões de Navegação
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

        self.btn_history = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=40, border_spacing=10, 
                                         text="Histórico", fg_color="transparent", text_color=("gray10", "gray90"),
                                         hover_color=("gray70", "gray30"), anchor="w", 
                                         command=lambda: self.select_frame_by_name("history"))
        self.btn_history.grid(row=3, column=0, sticky="ew")

        # Seletor de Tema
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Tema:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                            command=self.change_appearance_mode_event)
        self.appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10, 20))


        # ==================== FRAMES DAS TELAS ====================
        
        # TELA 1: Dashboard
        self.dashboard_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.dashboard_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.dashboard_frame.grid_rowconfigure(2, weight=1)
        self._setup_dashboard()

        # TELA 2: Novo Diagnóstico (Importação)
        self.predict_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.predict_frame.grid_columnconfigure(0, weight=1)
        self.predict_frame.grid_rowconfigure(1, weight=1)
        self._setup_predict()

        # TELA 3: Histórico
        self.history_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.history_frame.grid_columnconfigure(0, weight=1)
        self.history_frame.grid_rowconfigure(1, weight=1)
        self._setup_history()

        # Seleciona a tela inicial
        self.select_frame_by_name("dashboard")

    # ==================== CONSTRUÇÃO DAS TELAS ====================

    def _setup_dashboard(self):
        title = ctk.CTkLabel(self.dashboard_frame, text="Visão Geral dos Dados", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 20), sticky="w")

        # Cards de Métricas (Placeholder para métricas do WDBC)
        card1 = ctk.CTkFrame(self.dashboard_frame, fg_color=("gray85", "gray20"))
        card1.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card1, text="Total de Amostras", font=ctk.CTkFont(size=14)).pack(pady=(15, 5))
        ctk.CTkLabel(card1, text="569", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(0, 15))

        card2 = ctk.CTkFrame(self.dashboard_frame, fg_color=("gray85", "gray20"))
        card2.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card2, text="Casos Benignos", font=ctk.CTkFont(size=14)).pack(pady=(15, 5))
        ctk.CTkLabel(card2, text="357", font=ctk.CTkFont(size=28, weight="bold"), text_color="#2ecc71").pack(pady=(0, 15))

        card3 = ctk.CTkFrame(self.dashboard_frame, fg_color=("gray85", "gray20"))
        card3.grid(row=1, column=2, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card3, text="Casos Malignos", font=ctk.CTkFont(size=14)).pack(pady=(15, 5))
        ctk.CTkLabel(card3, text="212", font=ctk.CTkFont(size=28, weight="bold"), text_color="#e74c3c").pack(pady=(0, 15))

        # Espaço para Gráficos (UMAP/PCA)
        chart_frame = ctk.CTkFrame(self.dashboard_frame, fg_color=("gray85", "gray15"))
        chart_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=20, sticky="nsew")
        ctk.CTkLabel(chart_frame, text="[ Área reservada para renderização do gráfico UMAP ]", text_color="gray").place(relx=0.5, rely=0.5, anchor="center")

    def _setup_predict(self):
        title = ctk.CTkLabel(self.predict_frame, text="Importação de Lote Clínico", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        upload_frame = ctk.CTkFrame(self.predict_frame)
        upload_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        upload_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(upload_frame, text="Selecione um arquivo .csv contendo as variáveis extraídas por FNA:", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=20, pady=(30, 5), sticky="w")

        self.file_path_var = ctk.StringVar(value="Nenhum arquivo selecionado")
        ctk.CTkLabel(upload_frame, textvariable=self.file_path_var, font=ctk.CTkFont(size=12, slant="italic"), text_color="gray").grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        btn_upload = ctk.CTkButton(upload_frame, text="Procurar Arquivo CSV", command=self.select_file)
        btn_upload.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        self.btn_run = ctk.CTkButton(self.predict_frame, text="Processar Lote e Gerar XAI", height=45, font=ctk.CTkFont(size=15, weight="bold"), state="disabled")
        self.btn_run.grid(row=2, column=0, padx=20, pady=30, sticky="ew")

    def _setup_history(self):
        title = ctk.CTkLabel(self.history_frame, text="Histórico de Análises", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Frame com rolagem para listar o histórico
        self.scrollable_history = ctk.CTkScrollableFrame(self.history_frame, label_text="Relatórios Gerados (PDF/CSV)")
        self.scrollable_history.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.scrollable_history.grid_columnconfigure(0, weight=1)

        # Gerando itens de exemplo para o histórico
        for i in range(5):
            item_frame = ctk.CTkFrame(self.scrollable_history, fg_color=("gray85", "gray20"))
            item_frame.grid(row=i, column=0, padx=10, pady=5, sticky="ew")
            item_frame.grid_columnconfigure(1, weight=1)

            lbl_date = ctk.CTkLabel(item_frame, text=f"Lote_{10-i}.csv\nData: 0{i+1}/04/2026", justify="left")
            lbl_date.grid(row=0, column=0, padx=10, pady=10, sticky="w")

            btn_view = ctk.CTkButton(item_frame, text="Abrir Relatório SHAP", width=120)
            btn_view.grid(row=0, column=2, padx=10, pady=10, sticky="e")

    # ==================== LÓGICA E EVENTOS ====================

    def select_frame_by_name(self, name):
        # Atualiza a cor dos botões para indicar a aba ativa
        self.btn_dashboard.configure(fg_color=("gray75", "gray25") if name == "dashboard" else "transparent")
        self.btn_predict.configure(fg_color=("gray75", "gray25") if name == "predict" else "transparent")
        self.btn_history.configure(fg_color=("gray75", "gray25") if name == "history" else "transparent")

        # Esconde todos os frames
        self.dashboard_frame.grid_forget()
        self.predict_frame.grid_forget()
        self.history_frame.grid_forget()

        # Mostra o frame selecionado
        if name == "dashboard":
            self.dashboard_frame.grid(row=0, column=1, sticky="nsew")
        elif name == "predict":
            self.predict_frame.grid(row=0, column=1, sticky="nsew")
        elif name == "history":
            self.history_frame.grid(row=0, column=1, sticky="nsew")

    def select_file(self):
        filetypes = (('Arquivos CSV', '*.csv'), ('Todos os arquivos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecione o arquivo CSV', initialdir='/', filetypes=filetypes)

        if filename:
            self.selected_file_path = filename
            nome_arquivo = os.path.basename(filename)
            self.file_path_var.set(f"Arquivo selecionado: {nome_arquivo}")
            self.btn_run.configure(state="normal")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = App()
    app.mainloop()