"""
Módulo contendo a interface para importação e visualização de dados em lote.
"""

import customtkinter as ctk
from tkinter import filedialog, ttk
import os
import pandas as pd

class PredictView(ctk.CTkFrame):
    """
    Frame responsável pelo upload e pré-visualização de arquivos CSV clínicos.

    Permite que o usuário selecione um arquivo de lote, visualize seus 
    dados em uma tabela renderizada dinamicamente e dispare o modelo de IA.

    Attributes
    ----------
    df : pandas.DataFrame ou None
        Armazena o dataframe lido a partir do arquivo CSV.
    selected_file_path : str
        Caminho absoluto do arquivo selecionado no sistema.
    tree : ttk.Treeview
        Componente visual de tabela para exibir os dados do lote.
        
    Parameters
    ----------
    master : ctk.CTkBaseClass
        O widget pai ao qual este frame pertence.
    **kwargs
        Argumentos adicionais passados para o construtor do CTkFrame.
    """

    def __init__(self, master, **kwargs):
        """Inicializa o frame de predição e variáveis de estado."""
        super().__init__(master, corner_radius=10, fg_color="transparent", **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) 
        
        self.df = None  
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói a interface, incluindo botões de upload, labels 
        informativos e o componente Treeview para a tabela.
        """
        title = ctk.CTkLabel(self, text="Importação de Lote Clínico", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # --- Frame de Upload ---
        upload_frame = ctk.CTkFrame(self)
        upload_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        upload_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(upload_frame, text="Selecione um arquivo .csv contendo as variáveis extraídas por FNA:", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        self.file_path_var = ctk.StringVar(value="Nenhum arquivo selecionado")
        ctk.CTkLabel(upload_frame, textvariable=self.file_path_var, font=ctk.CTkFont(size=12, slant="italic"), text_color="gray").grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        btn_upload = ctk.CTkButton(upload_frame, text="Procurar Arquivo CSV", command=self.select_file)
        btn_upload.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        # --- Frame do Preview dos Dados (Tabela) ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(1, weight=1) 

        ctk.CTkLabel(self.preview_frame, text="Pré-visualização dos Dados:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        # Criando a tabela (Treeview)
        self.tree = ttk.Treeview(self.preview_frame, show="headings")
        self.tree.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="nsew")

        # Usando CTkScrollbar no lugar do scrollbar nativo
        scrollbar_y = ctk.CTkScrollbar(self.preview_frame, orientation="vertical", command=self.tree.yview)
        scrollbar_y.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ns")
        
        scrollbar_x = ctk.CTkScrollbar(self.preview_frame, orientation="horizontal", command=self.tree.xview)
        scrollbar_x.grid(row=2, column=0, padx=(10, 0), pady=(0, 10), sticky="ew")
        
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Customizando o estilo da tabela
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])
        style.configure("Treeview.Heading", background="#1f538d", foreground="white", relief="flat")
        style.map("Treeview.Heading", background=[('active', '#14375e')])

        # --- Botão de Processamento ---
        self.btn_run = ctk.CTkButton(self, text="Processar Lote e Gerar XAI", height=45, font=ctk.CTkFont(size=15, weight="bold"), state="disabled")
        self.btn_run.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

    def select_file(self):
        """
        Abre um explorador de arquivos do sistema focado em formatos .csv.

        Caso um arquivo válido seja selecionado, atualiza o caminho exibido
        na interface e invoca o carregamento dos dados na tabela.
        """
        filetypes = (('Arquivos CSV', '*.csv'), ('Todos os arquivos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecione o arquivo CSV', initialdir='/', filetypes=filetypes)

        if filename:
            self.selected_file_path = filename
            nome_arquivo = os.path.basename(filename)
            self.file_path_var.set(f"Arquivo selecionado: {nome_arquivo}")
            
            self._load_and_preview_csv(filename)

    def _load_and_preview_csv(self, filepath: str):
        """
        Lê um arquivo CSV utilizando pandas e popula o componente Treeview.

        Parameters
        ----------
        filepath : str
            Caminho absoluto ou relativo apontando para o arquivo .csv.

        Raises
        ------
        Exception
            Captura qualquer falha de leitura (ex: formatação inválida) e 
            atualiza a UI com uma mensagem de erro, desativando processamento.
        """
        try:
            # Lê o CSV inteiro
            self.df = pd.read_csv(filepath)
            
            # Limpa dados antigos da tabela
            self.tree.delete(*self.tree.get_children())
            
            # Configura os cabeçalhos das colunas dinamicamente
            colunas = list(self.df.columns)
            self.tree["columns"] = colunas
            
            for col in colunas:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, minwidth=120, stretch=False, anchor="center") 
            
            # Insere TODOS os dados na tabela
            for indice, linha in self.df.iterrows():
                valores = list(linha)
                self.tree.insert("", "end", values=valores)
                
            self.btn_run.configure(state="normal")
                
        except Exception as e:
            self.file_path_var.set(f"Erro ao carregar arquivo: {e}")
            self.btn_run.configure(state="disabled")