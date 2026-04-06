import customtkinter as ctk
from tkinter import filedialog
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green") 

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # janela
        self.title("Sistema de Diagnóstico XAI")
        self.geometry("900x650")
        
        # responsividade
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # menu lateral
        self.sidebar_frame = ctk.CTkFrame(self, width=400, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # logo / título
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Diagnosis XAI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 20))

        # botões
        self.btn_dashboard = ctk.CTkButton(self.sidebar_frame, text="Dashboard", fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=10)

        self.btn_predict = ctk.CTkButton(self.sidebar_frame, text="Novo Diagnóstico")
        self.btn_predict.grid(row=2, column=0, padx=20, pady=10)

        self.btn_history = ctk.CTkButton(self.sidebar_frame, text="Histórico", fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.btn_history.grid(row=3, column=0, padx=20, pady=10)

        # seletor de tema
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Tema de Aparência:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10, 20))


        # tela de importação
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.main_frame, text="Importação de Dados Clínicos", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # upload
        self.upload_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.upload_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.upload_frame.grid_columnconfigure(0, weight=1)

        self.instruction_label = ctk.CTkLabel(self.upload_frame, text="Selecione um arquivo .csv contendo os dados para predição:", font=ctk.CTkFont(size=14))
        self.instruction_label.grid(row=0, column=0, padx=10, pady=(15, 5), sticky="w")

        # variável para armazenar e mostrar o caminho do arquivo
        self.file_path_var = ctk.StringVar(value="Nenhum arquivo selecionado")
        self.file_label = ctk.CTkLabel(self.upload_frame, textvariable=self.file_path_var, font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.file_label.grid(row=1, column=0, padx=10, pady=(0, 15), sticky="w")

        # botão para abrir o explorador de arquivos
        self.btn_upload = ctk.CTkButton(self.upload_frame, text="Procurar Arquivo CSV", command=self.select_file)
        self.btn_upload.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # botão de diagnóstico
        self.btn_run = ctk.CTkButton(self.main_frame, text="Analisar Diagnóstico com XAI", height=45, font=ctk.CTkFont(size=15, weight="bold"), state="disabled")
        self.btn_run.grid(row=2, column=0, padx=20, pady=30, sticky="ew")

        self.selected_file_path = None

    # ==================== FUNÇÕES E EVENTOS ====================

    def select_file(self):
        filetypes = (
            ('Arquivos CSV', '*.csv'),
            ('Todos os arquivos', '*.*')
        )

        filename = filedialog.askopenfilename(
            title='Selecione o arquivo CSV',
            initialdir='/',
            filetypes=filetypes
        )

        # se o usuário escolheu um arquivo (não cancelou a janela)
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