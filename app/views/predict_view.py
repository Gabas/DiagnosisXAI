"""
Módulo contendo a interface em etapas para importação, padronização, predição e auditoria.
"""

import customtkinter as ctk
from tkinter import filedialog, ttk
import os
import pandas as pd

from core.batch_processor import BatchProcessor
from core.inference import ModelLoader
from core.predictor import PredictorEngine

class PredictView(ctk.CTkFrame):
    """
    Frame responsável pelo pipeline completo de diagnóstico assistido e validação.

    Attributes
    ----------
    df_bruto : pandas.DataFrame ou None
        Dados recém-importados do CSV.
    df_padronizado : pandas.DataFrame ou None
        Dados processados através da transformação Z-score.
    df_resultado : pandas.DataFrame ou None
        Dados consolidados com a predição da inferência.
    """

    def __init__(self, master, **kwargs):
        """
        Inicializa o frame de predição, configurando o layout e instanciando os motores lógicos.

        Parameters
        ----------
        master : ctk.CTkBaseClass
            O widget pai ao qual este frame pertence.
        **kwargs
            Argumentos adicionais passados para o construtor do CTkFrame.
        """
        super().__init__(master, corner_radius=10, fg_color="transparent", **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1) 
        
        self.df_bruto = None
        self.df_padronizado = None
        self.df_resultado = None
        
        self.model_loader = ModelLoader()
        self.predictor = PredictorEngine(self.model_loader)
        
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói e posiciona os componentes visuais da interface de predição.
        """
        title = ctk.CTkLabel(self, text="Diagnóstico Assistido por IA", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # --- Passo 1: Upload ---
        upload_frame = ctk.CTkFrame(self)
        upload_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(upload_frame, text="Passo 1: Importar Dados Brutos (CSV)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        self.file_path_var = ctk.StringVar(value="Nenhum arquivo selecionado")
        ctk.CTkLabel(upload_frame, textvariable=self.file_path_var, font=ctk.CTkFont(size=12, slant="italic"), text_color="gray").grid(row=1, column=0, padx=20, pady=0, sticky="w")
        ctk.CTkButton(upload_frame, text="Procurar Arquivo", command=self.select_file).grid(row=2, column=0, padx=20, pady=10, sticky="w")

        # --- Passo 2: Padronização ---
        padroniza_frame = ctk.CTkFrame(self)
        padroniza_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(padroniza_frame, text="Passo 2: Higienizar e Escalar (Z-Score)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        self.btn_standardize = ctk.CTkButton(padroniza_frame, text="Aplicar Padronização", state="disabled", command=self.standardize_data, fg_color="#d35400", hover_color="#e67e22")
        self.btn_standardize.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        # --- Passo 3: Inferência de IA ---
        ia_frame = ctk.CTkFrame(self)
        ia_frame.grid(row=3, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(ia_frame, text="Passo 3: Inteligência Artificial", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        
        # Volta a adicionar a opção "Todos (Comparação)" no topo da lista
        modelos_disponiveis = ["Todos (Comparação)"] + list(self.model_loader.models.keys())
        self.model_selector = ctk.CTkOptionMenu(ia_frame, values=modelos_disponiveis, state="disabled")
        self.model_selector.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.btn_run = ctk.CTkButton(ia_frame, text="Processar Diagnóstico", state="disabled", command=self.process_batch, fg_color="#27ae60", hover_color="#2ecc71")
        self.btn_run.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # --- Passo 4: Auditoria (Opcional) ---
        audit_frame = ctk.CTkFrame(self)
        audit_frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(audit_frame, text="Passo 4: Auditoria Acadêmica (Opcional)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.btn_audit = ctk.CTkButton(audit_frame, text="Carregar Gabarito (CSV)", state="disabled", command=self.run_audit, fg_color="#8e44ad", hover_color="#9b59b6")
        self.btn_audit.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        
        self.lbl_audit_results = ctk.CTkLabel(audit_frame, text="", justify="left", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_audit_results.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # --- Tabela de Preview ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(1, weight=1) 

        self.tree = ttk.Treeview(self.preview_frame, show="headings")
        self.tree.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="nsew")

        scrollbar_y = ctk.CTkScrollbar(self.preview_frame, orientation="vertical", command=self.tree.yview)
        scrollbar_y.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ns")
        scrollbar_x = ctk.CTkScrollbar(self.preview_frame, orientation="horizontal", command=self.tree.xview)
        scrollbar_x.grid(row=2, column=0, padx=(10, 0), pady=(0, 10), sticky="ew")
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])
        style.configure("Treeview.Heading", background="#1f538d", foreground="white", relief="flat")
        style.map("Treeview.Heading", background=[('active', '#14375e')])

    def select_file(self):
        """
        Abre o explorador de arquivos do sistema focado na seleção de arquivos CSV.

        Caso um arquivo seja selecionado, atualiza o caminho na interface e invoca
        a leitura e pré-visualização dos dados.
        """
        filetypes = (('Arquivos CSV', '*.csv'), ('Todos os arquivos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecione o arquivo CSV', initialdir='/', filetypes=filetypes)

        if filename:
            self.file_path_var.set(f"Arquivo selecionado: {os.path.basename(filename)}")
            self._load_and_preview_csv(filename)

    def _load_and_preview_csv(self, filepath: str):
        """
        Lê o arquivo CSV selecionado, renderiza os dados brutos no Treeview e 
        reseta o estado das etapas subsequentes.

        Parameters
        ----------
        filepath : str
            Caminho absoluto do arquivo CSV a ser carregado.
        """
        try:
            self.df_bruto = pd.read_csv(filepath)
            self._update_treeview_with_data(self.df_bruto)
            
            # Reset e habilitação dos próximos passos
            self.btn_standardize.configure(state="normal")
            self.model_selector.configure(state="disabled")
            self.btn_run.configure(state="disabled")
            self.btn_audit.configure(state="disabled")
            self.lbl_audit_results.configure(text="")
        except Exception as e:
            self.file_path_var.set(f"Erro ao carregar arquivo: {e}")

    def standardize_data(self):
        """
        Instancia o BatchProcessor para higienizar e padronizar os dados brutos,
        atualiza a visualização e habilita a etapa de inferência.
        """
        if self.df_bruto is not None:
            try:
                processor = BatchProcessor()
                self.df_padronizado = processor.process(self.df_bruto)
                self._update_treeview_with_data(self.df_padronizado)
                
                self.model_selector.configure(state="normal")
                self.btn_run.configure(state="normal")
            except Exception as e:
                print(f"Erro na padronização: {e}")

    def process_batch(self):
        """
        Obtém o modelo selecionado, submete os dados padronizados ao PredictorEngine,
        atualiza o Treeview com as predições e habilita a etapa de auditoria.
        """
        if self.df_padronizado is not None:
            try:
                modelo_escolhido = self.model_selector.get()
                self.df_resultado = self.predictor.predict(self.df_padronizado, modelo_escolhido)
                self._update_treeview_with_data(self.df_resultado)
                
                # Libera o Passo 4 após a IA rodar
                self.btn_audit.configure(state="normal")
                self.lbl_audit_results.configure(text="")
            except Exception as e:
                print(f"Erro durante a inferência: {e}")

    def run_audit(self):
        
        """Abre o CSV com diagnósticos reais, adiciona à tabela e calcula a acurácia."""
        filetypes = (('Arquivos CSV', '*.csv'), ('Todos os arquivos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecione o arquivo Gabarito', initialdir='/', filetypes=filetypes)

        if filename:
            try:
                df_gabarito = pd.read_csv(filename)

                # --- VALIDAÇÕES DE INTEGRIDADE ---
                if 'diagnosis' not in df_gabarito.columns:
                    self.lbl_audit_results.configure(text="Erro: O arquivo não possui a coluna 'diagnosis'.", text_color="#e74c3c")
                    return

                if len(df_gabarito) != len(self.df_bruto):
                    self.lbl_audit_results.configure(text=f"Erro: O gabarito tem {len(df_gabarito)} pacientes, mas o lote inicial tem {len(self.df_bruto)}.", text_color="#e74c3c")
                    return

                primeira_col = self.df_bruto.columns[0]
                dados_brutos = self.df_bruto[primeira_col].round(4).tolist()
                dados_gabarito = df_gabarito[primeira_col].round(4).tolist()

                if dados_brutos != dados_gabarito:
                    self.lbl_audit_results.configure(text="Erro de Validação: Os dados morfólogicos não correspondem ao lote do Passo 1.", text_color="#e74c3c")
                    return
                # ----------------------------------

                # 1. Traduz a coluna diagnosis para texto e adiciona ao DataFrame de resultado
                gabarito_text = ['Maligno' if val == 1 else 'Benigno' for val in df_gabarito['diagnosis']]
                self.df_resultado['Diagnóstico_Real'] = gabarito_text
                
                # 2. Atualiza a tabela na tela para mostrar a nova coluna "Diagnóstico_Real"
                self._update_treeview_with_data(self.df_resultado)

                # 3. Calcula a acurácia
                resultados_texto = "Acurácia no Lote:\n"
                colunas_ia = [col for col in self.df_resultado.columns if col.startswith('IA_')]
                
                if colunas_ia:
                    # Se rodou "Todos os Modelos", mostra a nota de cada um
                    for col in colunas_ia:
                        acertos = (self.df_resultado[col] == self.df_resultado['Diagnóstico_Real']).sum()
                        acc = (acertos / len(gabarito_text)) * 100
                        resultados_texto += f"   • {col.replace('IA_', '')}: {acc:.2f}%\n"
                else:
                    # Se rodou apenas um modelo
                    acertos = (self.df_resultado['Diagnóstico_IA'] == self.df_resultado['Diagnóstico_Real']).sum()
                    acc = (acertos / len(gabarito_text)) * 100
                    resultados_texto += f"   • Modelo Selecionado: {acc:.2f}%"

                self.lbl_audit_results.configure(text=resultados_texto, text_color="#2ecc71")

            except Exception as e:
                self.lbl_audit_results.configure(text=f"Erro na auditoria: {e}", text_color="#e74c3c")

    def _update_treeview_with_data(self, df: pd.DataFrame):
        """
        Limpa as informações atuais do componente Treeview e o repopula com
        os dados estruturados do DataFrame fornecido.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame contendo os dados a serem renderizados visualmente na tabela.
        """
        self.tree.delete(*self.tree.get_children())
        colunas = list(df.columns)
        self.tree["columns"] = colunas
        
        for col in colunas:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, minwidth=130, stretch=False, anchor="center") 
            
        for indice, linha in df.iterrows():
            valores = [round(val, 4) if isinstance(val, float) else val for val in linha]
            self.tree.insert("", "end", values=valores)