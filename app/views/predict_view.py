"""
Módulo contendo a interface em etapas para importação, padronização, predição e auditoria.
"""

import customtkinter as ctk
from tkinter import filedialog, ttk
import os
import numpy as np
import pandas as pd

from core.batch_processor import BatchProcessor
from core.biomarkers import descricao_coluna
from core.history_manager import HistoryManager
from core.inference import ModelLoader
from core.predictor import PredictorEngine
from utils.ui import ScrollableFrame, bind_treeview_mousewheel, HeadingTooltip
from utils.pdf_report import export_batch_report, resolve_reports_dir
from views.report_window import ReportWindow
from views.report_window_lr import LogisticReportWindow
from views.report_window_knn import KNNReportWindow
from views.report_window_rf import RandomForestReportWindow
from views.report_window_svm import SVMReportWindow

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

    NOME_ARVORE = "Árvore de Decisão"
    NOME_LOGISTICA = "Regressão Logística"
    NOME_KNN = "KNN"
    NOME_RF = "Random Forest"
    NOME_SVM = "SVM"
    NOME_TODOS = "Todos (Comparação)"

    # Mapa: tipo de relatório exato -> classe da janela correspondente.
    _CLASSES_RELATORIO = {
        'arvore': ReportWindow,
        'logistica': LogisticReportWindow,
        'knn': KNNReportWindow,
        'randomforest': RandomForestReportWindow,
        'svm': SVMReportWindow,
    }

    # Nome do modelo (no seletor) <-> chave curta usada no SHAP.
    _MODELO_KEY = {
        "Árvore de Decisão": 'dt',
        "Random Forest": 'rf',
        "Regressão Logística": 'lr',
        "SVM": 'svm',
        "KNN": 'knn',
    }
    _KEY_MODELO = {v: k for k, v in _MODELO_KEY.items()}

    # Rótulos exibidos no menu de relatórios do Passo 5.
    _ROTULOS_RELATORIO = {
        'arvore': "Árvore de Decisão — regras",
        'logistica': "Regressão Logística — contribuições",
        'knn': "KNN — vizinhos",
        'randomforest': "Random Forest — consenso das árvores",
        'svm': "SVM — vetores de suporte",
        'shap_dt': "Árvore de Decisão — SHAP",
        'shap_rf': "Random Forest — SHAP",
        'shap_lr': "Regressão Logística — SHAP",
        'shap_svm': "SVM — SHAP",
        'shap_knn': "KNN — SHAP",
        'umap': "Mapa Populacional — UMAP",
    }

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
        self.grid_rowconfigure(0, weight=1)

        self.df_bruto = None
        self.df_padronizado = None
        self.df_limpo = None
        self.df_resultado = None

        self._ultima_explicacao = {}
        self._ultima_acuracia = None
        self._report_windows = {}
        self._shap_disponiveis = []
        self._opcoes_relatorio = {}
        self._shap_cache = {}

        self.model_loader = ModelLoader()
        self.predictor = PredictorEngine(self.model_loader)
        self._history_manager = HistoryManager()
        
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói e posiciona os componentes visuais da interface de predição.
        """
        container = ScrollableFrame(self, fg_color="transparent")
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(container, text="Diagnóstico Assistido por IA", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # --- Passo 1: Upload ---
        upload_frame = ctk.CTkFrame(container)
        upload_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(upload_frame, text="Passo 1: Importar Dados Brutos (CSV)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        self.file_path_var = ctk.StringVar(value="Nenhum arquivo selecionado")
        ctk.CTkLabel(upload_frame, textvariable=self.file_path_var, font=ctk.CTkFont(size=12, slant="italic"), text_color="gray").grid(row=1, column=0, padx=20, pady=0, sticky="w")
        ctk.CTkButton(upload_frame, text="Procurar Arquivo", command=self.select_file).grid(row=2, column=0, padx=20, pady=10, sticky="w")

        # --- Passo 2: Padronização ---
        padroniza_frame = ctk.CTkFrame(container)
        padroniza_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(padroniza_frame, text="Passo 2: Higienizar e Escalar (Z-Score)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        self.btn_standardize = ctk.CTkButton(padroniza_frame, text="Aplicar Padronização", state="disabled", command=self.standardize_data, fg_color="#d35400", hover_color="#e67e22")
        self.btn_standardize.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.lbl_standardize_error = ctk.CTkLabel(padroniza_frame, text="", font=ctk.CTkFont(size=12), text_color="#e74c3c", justify="left")
        self.lbl_standardize_error.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="w")

        # --- Passo 3: Inferência de IA ---
        ia_frame = ctk.CTkFrame(container)
        ia_frame.grid(row=3, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(ia_frame, text="Passo 3: Inteligência Artificial", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        
        # Volta a adicionar a opção "Todos (Comparação)" no topo da lista
        modelos_disponiveis = ["Todos (Comparação)"] + list(self.model_loader.models.keys())
        self.model_selector = ctk.CTkOptionMenu(ia_frame, values=modelos_disponiveis, state="disabled")
        self.model_selector.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.btn_run = ctk.CTkButton(ia_frame, text="Processar Diagnóstico", state="disabled", command=self.process_batch, fg_color="#27ae60", hover_color="#2ecc71")
        self.btn_run.grid(row=1, column=1, padx=20, pady=10, sticky="w")
        self.lbl_run_error = ctk.CTkLabel(ia_frame, text="", font=ctk.CTkFont(size=12), text_color="#e74c3c", justify="left")
        self.lbl_run_error.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")

        # --- Passo 4: Auditoria (Opcional) ---
        audit_frame = ctk.CTkFrame(container)
        audit_frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(audit_frame, text="Passo 4: Auditoria Acadêmica (Opcional)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.btn_audit = ctk.CTkButton(audit_frame, text="Carregar Gabarito (CSV)", state="disabled", command=self.run_audit, fg_color="#8e44ad", hover_color="#9b59b6")
        self.btn_audit.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        
        self.lbl_audit_results = ctk.CTkLabel(audit_frame, text="", justify="left", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_audit_results.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # --- Passo 5: Explicabilidade (XAI) ---
        xai_frame = ctk.CTkFrame(container)
        xai_frame.grid(row=5, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(xai_frame, text="Passo 5: Explicabilidade (XAI)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        sel_frame = ctk.CTkFrame(xai_frame, fg_color="transparent")
        sel_frame.grid(row=1, column=0, padx=20, pady=(0, 6), sticky="w")
        self.report_menu = ctk.CTkOptionMenu(sel_frame, values=["—"], state="disabled", width=340)
        self.report_menu.grid(row=0, column=0, padx=(0, 10), pady=6, sticky="w")
        self.btn_abrir_relatorio = ctk.CTkButton(sel_frame, text="Abrir Relatório", state="disabled", command=self._abrir_relatorio_selecionado, fg_color="#2980b9", hover_color="#3498db")
        self.btn_abrir_relatorio.grid(row=0, column=1, pady=6, sticky="w")

        self.lbl_report_hint = ctk.CTkLabel(xai_frame, text="Disponível após processar o diagnóstico.", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.lbl_report_hint.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="w")

        # --- Passo 6: Exportar Resultados ---
        export_frame = ctk.CTkFrame(container)
        export_frame.grid(row=6, column=0, padx=20, pady=5, sticky="nsew")
        ctk.CTkLabel(export_frame, text="Passo 6: Exportar Resultados", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=(10, 5), sticky="w")

        self.btn_export_csv = ctk.CTkButton(export_frame, text="Exportar CSV", state="disabled", command=self._exportar_csv, fg_color="#16a085", hover_color="#1abc9c")
        self.btn_export_csv.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.btn_export_pdf = ctk.CTkButton(export_frame, text="Exportar PDF (Resumo do Lote)", state="disabled", command=self._exportar_pdf_lote, fg_color="#16a085", hover_color="#1abc9c")
        self.btn_export_pdf.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        self.lbl_export_hint = ctk.CTkLabel(export_frame, text="Disponível após processar o diagnóstico.", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.lbl_export_hint.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")

        # --- Tabela de Preview ---
        self.preview_frame = ctk.CTkFrame(container)
        self.preview_frame.grid(row=7, column=0, padx=20, pady=10, sticky="nsew")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(1, weight=1) 

        self.tree = ttk.Treeview(self.preview_frame, show="headings", height=15)
        self.tree.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="nsew")
        bind_treeview_mousewheel(self.tree)
        # Tooltip nos cabeçalhos: explica cada biomarcador/coluna ao passar o mouse.
        self._heading_tooltip = HeadingTooltip(self.tree, descricao_coluna)

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

    def _diretorio_inicial(self) -> str:
        """
        Resolve o diretório onde os diálogos de seleção de arquivo devem abrir.

        Returns
        -------
        str
            Caminho da pasta ``data/`` do repositório (onde ficam os CSVs).
            Caso ela não exista, retorna a raiz do repositório como reserva.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, 'data')
        return data_dir if os.path.isdir(data_dir) else base_dir

    def select_file(self):
        """
        Abre o explorador de arquivos do sistema focado na seleção de arquivos CSV.

        Caso um arquivo seja selecionado, atualiza o caminho na interface e invoca
        a leitura e pré-visualização dos dados.
        """
        filetypes = (('Arquivos CSV', '*.csv'), ('Todos os arquivos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecione o arquivo CSV', initialdir=self._diretorio_inicial(), filetypes=filetypes)

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
            self.lbl_standardize_error.configure(text="")
            self.model_selector.configure(state="disabled")
            self.btn_run.configure(state="disabled")
            self.lbl_run_error.configure(text="")
            self.btn_audit.configure(state="disabled")
            self.lbl_audit_results.configure(text="")
            self._ultima_acuracia = None
            self.btn_export_csv.configure(state="disabled")
            self.btn_export_pdf.configure(state="disabled")
            self.lbl_export_hint.configure(
                text="Disponível após processar o diagnóstico.", text_color="gray")
            self._reset_relatorio()
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
                self.df_padronizado, self.df_limpo = processor.process(self.df_bruto)
                self._update_treeview_with_data(self.df_padronizado)
                
                self.model_selector.configure(state="normal")
                self.btn_run.configure(state="normal")
                self.lbl_standardize_error.configure(text="")
            except Exception as e:
                self.lbl_standardize_error.configure(text=f"Erro na padronização: {e}")

    def process_batch(self):
        """
        Obtém o modelo selecionado, submete os dados padronizados ao PredictorEngine,
        atualiza o Treeview com as predições e habilita a etapa de auditoria.
        """
        if self.df_padronizado is not None:
            try:
                modelo_escolhido = self.model_selector.get()
                self.df_resultado = self.predictor.predict(self.df_padronizado, self.df_limpo, modelo_escolhido)
                self._update_treeview_with_data(self.df_resultado)
                self.lbl_run_error.configure(text="")

                # Libera o Passo 4 após a IA rodar
                self.btn_audit.configure(state="normal")
                self.lbl_audit_results.configure(text="")
                self._ultima_acuracia = None

                # Libera o Passo 6 (exportação) após a IA rodar
                self.btn_export_csv.configure(state="normal")
                self.btn_export_pdf.configure(state="normal")
                self.lbl_export_hint.configure(text="Pronto para exportar.", text_color="#2ecc71")

                # Gera e exibe a explicabilidade da Árvore de Decisão ao final
                self._preparar_relatorio(modelo_escolhido)

                # Persiste a sessão no histórico, embutindo o relatório quando houver
                arquivo = self.file_path_var.get().replace("Arquivo selecionado: ", "")
                total = len(self.df_resultado)
                if 'Diagnóstico_IA' in self.df_resultado.columns:
                    malignos = int((self.df_resultado['Diagnóstico_IA'] == 'Maligno').sum())
                    benignos = int((self.df_resultado['Diagnóstico_IA'] == 'Benigno').sum())
                else:
                    malignos = benignos = None
                self._history_manager.save_session(
                    arquivo, modelo_escolhido, total, malignos, benignos,
                    self._relatorio_para_historico() or None,
                )
            except Exception as e:
                self.lbl_run_error.configure(text=f"Erro durante a inferência: {e}")

    def _relatorio_para_historico(self) -> dict:
        """
        Monta o dicionário de relatórios a persistir na sessão do histórico.

        Reúne os relatórios exatos (Árvore/LR/KNN/Random Forest) e, para que o
        SHAP e o UMAP possam ser reabertos depois de forma interativa, guarda os
        dados do lote necessários para reconstruí-los: o lote padronizado/bruto e
        os modelos com SHAP, e a projeção UMAP 2D já calculada. Os artefatos
        pesados (modelos, background, embedding de treino) não são duplicados —
        vêm do wisconsin.pkl ao reabrir.

        Returns
        -------
        dict
            Mapa {tipo: dados} com os relatórios exatos e, quando disponíveis,
            as chaves 'shap' (lote + modelos) e 'umap' (projeção + pacientes).
        """
        relatorio = dict(self._ultima_explicacao)  # relatórios exatos (cópia)
        fn = self.model_loader.feature_names

        # SHAP: guarda o lote uma única vez + a lista de modelos com SHAP.
        if self._shap_disponiveis:
            relatorio['shap'] = {
                'X_scaled': self.df_padronizado[fn].values.tolist(),
                'X_raw': self.df_limpo[fn].values.tolist(),
                'indices': [int(i) for i in self.df_padronizado.index],
                'modelos': list(self._shap_disponiveis),
            }

        # UMAP: guarda a projeção 2D do lote (o fundo de treino vem do .pkl).
        if self.model_loader.umap_train_2d is not None:
            from views import report_launchers
            batch_2d = report_launchers.projetar_umap(
                self.model_loader, self.df_padronizado[fn].values)
            if batch_2d is not None:
                relatorio['umap'] = {
                    'batch_2d': batch_2d.tolist(),
                    'pacientes': self._pacientes_diagnostico(),
                }

        return relatorio

    def _reset_relatorio(self):
        """Limpa o estado dos relatórios de explicabilidade e desabilita o Passo 5."""
        self._ultima_explicacao = {}
        self._shap_disponiveis = []
        self._opcoes_relatorio = {}
        self.report_menu.configure(values=["—"], state="disabled")
        self.report_menu.set("—")
        self.btn_abrir_relatorio.configure(state="disabled")
        self.lbl_report_hint.configure(
            text="Disponível após processar o diagnóstico.", text_color="gray")

    def _gerar_exato(self, tipo: str, funcao):
        """
        Gera o relatório exato de um modelo, isolando eventuais falhas.

        Se o explicador estiver indisponível (por exemplo, um .pkl regenerado
        com código antigo), o problema é registrado e o processamento segue —
        sem derrubar os demais relatórios do Passo 5.

        Parameters
        ----------
        tipo : str
            'arvore', 'logistica' ou 'knn'.
        funcao : callable
            Função sem argumentos que produz o dicionário do relatório.
        """
        try:
            self._ultima_explicacao[tipo] = funcao()
        except Exception as e:
            print(f"Explicador '{tipo}' indisponível: {e}")

    def _preparar_relatorio(self, modelo_escolhido: str):
        """
        Gera as explicações dos modelos executados e popula o menu de relatórios.

        Os explicadores exatos (Árvore, Regressão Logística e KNN) são calculados
        na hora. Os relatórios SHAP ficam disponíveis para todos os modelos que
        rodaram, mas são calculados sob demanda ao serem abertos (lazy). Havendo
        um único relatório, ele é aberto automaticamente.

        Parameters
        ----------
        modelo_escolhido : str
            Nome do modelo selecionado pelo utilizador no Passo 3.
        """
        self._reset_relatorio()

        if self.df_limpo is None or self.df_padronizado is None:
            return

        todos = modelo_escolhido == self.NOME_TODOS
        explicadores = self.model_loader.explainers

        # Cada explicador é isolado: se um estiver indisponível (ex.: .pkl antigo
        # com bug), os demais relatórios continuam funcionando.
        exp = explicadores.get('arvore')
        if exp is not None and (todos or modelo_escolhido == self.NOME_ARVORE):
            self._gerar_exato('arvore', lambda: {
                'importancias': self._importancias_arvore(exp),
                'explicacoes': exp.explain(self.df_limpo),
            })
        exp = explicadores.get('logistica')
        if exp is not None and (todos or modelo_escolhido == self.NOME_LOGISTICA):
            self._gerar_exato('logistica', lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado, self.df_limpo),
            })
        exp = explicadores.get('knn')
        if exp is not None and (todos or modelo_escolhido == self.NOME_KNN):
            self._gerar_exato('knn', lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado),
                'contexto': exp.contexto(),
            })
        exp = explicadores.get('randomforest')
        if exp is not None and (todos or modelo_escolhido == self.NOME_RF):
            self._gerar_exato('randomforest', lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado),
                'contexto': exp.contexto(),
            })
        exp = explicadores.get('svm')
        if exp is not None and (todos or modelo_escolhido == self.NOME_SVM):
            self._gerar_exato('svm', lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado),
                'contexto': exp.contexto(),
                'batch_2d': self._batch_2d_para_svm(),
            })

        # SHAP: disponível para cada modelo executado que tenha artefatos salvos.
        if self.model_loader.shap_background is not None:
            for nome_modelo, key in self._MODELO_KEY.items():
                if (todos or modelo_escolhido == nome_modelo) \
                        and key in self.model_loader.shap_importances:
                    self._shap_disponiveis.append(key)

        # Monta as opções do menu: exatos primeiro, depois SHAP.
        opcoes = {}
        for tipo in ('arvore', 'logistica', 'knn', 'randomforest', 'svm'):
            if tipo in self._ultima_explicacao:
                opcoes[self._ROTULOS_RELATORIO[tipo]] = tipo
        for key in self._shap_disponiveis:
            tipo = f'shap_{key}'
            opcoes[self._ROTULOS_RELATORIO[tipo]] = tipo

        # Mapa populacional UMAP: independe do modelo, exige apenas o embedding.
        if self.model_loader.umap_train_2d is not None:
            opcoes[self._ROTULOS_RELATORIO['umap']] = 'umap'

        self._opcoes_relatorio = opcoes
        if not opcoes:
            return

        labels = list(opcoes.keys())
        self.report_menu.configure(values=labels, state="normal")
        self.report_menu.set(labels[0])
        self.btn_abrir_relatorio.configure(state="normal")

        if len(labels) == 1:
            self.lbl_report_hint.configure(text="Relatório pronto.", text_color="#2ecc71")
            self._abrir_relatorio(opcoes[labels[0]])
        else:
            self.lbl_report_hint.configure(
                text=f"{len(labels)} relatórios prontos — escolha e clique em Abrir.",
                text_color="#2ecc71")

    def _batch_2d_para_svm(self):
        """
        Projeta o lote no embedding UMAP do treino, para o mapa do relatório SVM.

        Reaproveita a mesma projeção usada pelo Mapa Populacional — os vetores
        de suporte plotados no relatório do SVM vêm desse mesmo embedding.

        Returns
        -------
        list ou None
            Lista de posições 2D (n_lote × 2), ou None se o embedding UMAP do
            treino não estiver disponível (.pkl sem esse artefato).
        """
        from views import report_launchers

        fn = self.model_loader.feature_names
        batch_2d = report_launchers.projetar_umap(
            self.model_loader, self.df_padronizado[fn].values)
        return batch_2d.tolist() if batch_2d is not None else None

    def _importancias_arvore(self, exp) -> list:
        """
        Calcula a importância global da Árvore de Decisão em Python puro.

        Reimplementa o cálculo do sklearn (redução de impureza por atributo) a
        partir dos arrays da árvore, evitando a propriedade nativa
        ``feature_importances_`` — que, neste ambiente (numpy 2.x), provoca uma
        falha de memória intermitente dentro do stack gráfico do app.

        Parameters
        ----------
        exp : DecisionTreeExplainer
            Explicador da árvore, do qual se lê a estrutura interna.

        Returns
        -------
        list[tuple[str, float]]
            Pares (característica, importância) com importância > 0, do maior
            para o menor (top 10).
        """
        t = exp._tree
        feat, cl, cr = t.feature, t.children_left, t.children_right
        wn, impureza = t.weighted_n_node_samples, t.impurity

        imp = np.zeros(len(exp.feature_names))
        for node in range(len(feat)):
            esq = cl[node]
            if esq != -1:  # nó de decisão (não folha)
                dir_ = cr[node]
                imp[feat[node]] += (wn[node] * impureza[node]
                                    - wn[esq] * impureza[esq]
                                    - wn[dir_] * impureza[dir_])
        total = imp.sum()
        if total > 0:
            imp = imp / total

        pares = [(exp.feature_names[i], float(imp[i]))
                 for i in range(len(imp)) if imp[i] > 0]
        pares.sort(key=lambda p: p[1], reverse=True)
        return pares[:10]

    def _abrir_relatorio_selecionado(self):
        """Abre o relatório atualmente selecionado no menu do Passo 5."""
        tipo = self._opcoes_relatorio.get(self.report_menu.get())
        if tipo:
            self._abrir_relatorio(tipo)

    def _abrir_relatorio(self, tipo: str):
        """
        Abre (ou recria) a janela do relatório indicado (exato ou SHAP).

        Parameters
        ----------
        tipo : str
            'arvore'/'logistica'/'knn' (exatos) ou 'shap_<modelo>'.
        """
        janela = self._report_windows.get(tipo)
        if janela is not None and janela.winfo_exists():
            janela.destroy()

        if tipo == 'umap':
            self._report_windows['umap'] = self._abrir_umap()
        elif tipo.startswith('shap_'):
            self._report_windows[tipo] = self._abrir_shap(tipo[len('shap_'):])
        else:
            dados = self._ultima_explicacao.get(tipo)
            if not dados:
                return
            self._report_windows[tipo] = self._CLASSES_RELATORIO[tipo](self, **dados)

    def _abrir_shap(self, key: str):
        """
        Abre a janela SHAP do modelo a partir do lote atual.

        Delega a construção ao módulo compartilhado ``report_launchers`` (o mesmo
        usado pelo histórico), reaproveitando o cache de explicadores SHAP.

        Parameters
        ----------
        key : str
            Chave do modelo ('dt', 'rf', 'lr', 'svm' ou 'knn').

        Returns
        -------
        ShapReportWindow
            A janela de relatório SHAP recém-criada.
        """
        from views import report_launchers

        fn = self.model_loader.feature_names
        if key not in self._shap_cache:
            self.lbl_report_hint.configure(text="Preparando SHAP…", text_color="gray")
            self.update_idletasks()

        return report_launchers.abrir_shap(
            self, self.model_loader, key,
            self.df_padronizado[fn].values, self.df_limpo[fn].values,
            list(self.df_padronizado.index), self._shap_cache)

    def _abrir_umap(self):
        """
        Projeta o lote no embedding UMAP do treino e abre o mapa populacional.

        Returns
        -------
        UmapMapWindow ou None
            A janela do mapa recém-criada, ou None se o embedding faltar.
        """
        from views import report_launchers

        fn = self.model_loader.feature_names
        batch_2d = report_launchers.projetar_umap(
            self.model_loader, self.df_padronizado[fn].values)
        if batch_2d is None:
            self.lbl_report_hint.configure(
                text="Mapa UMAP indisponível: regenere o wisconsin.pkl pelo notebook.",
                text_color="#e74c3c")
            return None

        return report_launchers.abrir_umap(
            self, self.model_loader, batch_2d, self._pacientes_diagnostico())

    def _pacientes_diagnostico(self) -> list:
        """
        Monta a lista {'indice', 'classe'} do lote a partir do resultado da IA.

        Para um único modelo usa a coluna 'Diagnóstico_IA'; no modo de comparação
        ('Todos') usa o voto da maioria entre as colunas dos modelos.

        Returns
        -------
        list[dict]
            Um item por paciente, com índice e classe prevista.
        """
        df = self.df_resultado
        indices = list(df.index)
        if 'Diagnóstico_IA' in df.columns:
            classes = list(df['Diagnóstico_IA'])
        else:
            colunas_ia = [c for c in df.columns if c.startswith('IA_')]
            classes = [
                'Maligno' if (linha == 'Maligno').sum() * 2 >= len(colunas_ia) else 'Benigno'
                for _, linha in df[colunas_ia].iterrows()
            ]
        return [{'indice': int(indices[i]), 'classe': classes[i]} for i in range(len(indices))]

    def run_audit(self):
        
        """Abre o CSV com diagnósticos reais, adiciona à tabela e calcula a acurácia."""
        filetypes = (('Arquivos CSV', '*.csv'), ('Todos os arquivos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecione o arquivo Gabarito', initialdir=self._diretorio_inicial(), filetypes=filetypes)

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

                # 3. Calcula a acurácia por modelo
                acuracia = {}
                colunas_ia = [col for col in self.df_resultado.columns if col.startswith('IA_')]

                if colunas_ia:
                    for col in colunas_ia:
                        acertos = (self.df_resultado[col] == self.df_resultado['Diagnóstico_Real']).sum()
                        acuracia[col.replace('IA_', '')] = round((acertos / len(gabarito_text)) * 100, 2)
                else:
                    acertos = (self.df_resultado['Diagnóstico_IA'] == self.df_resultado['Diagnóstico_Real']).sum()
                    acuracia['Modelo Selecionado'] = round((acertos / len(gabarito_text)) * 100, 2)

                resultados_texto = "Acurácia no Lote:\n" + "\n".join(
                    f"   • {k}: {v:.2f}%" for k, v in acuracia.items()
                )
                self.lbl_audit_results.configure(text=resultados_texto, text_color="#2ecc71")
                self._ultima_acuracia = acuracia

                # Atualiza a acurácia na entrada mais recente do histórico
                self._history_manager.update_last_accuracy(acuracia)

            except Exception as e:
                self.lbl_audit_results.configure(text=f"Erro na auditoria: {e}", text_color="#e74c3c")

    @staticmethod
    def _formatar_importancias(tipo: str, dados: dict) -> list:
        """
        Formata as importâncias de um explicador como pares (nome, texto).

        A Regressão Logística expõe coeficientes assinados (dicts com
        'feature'/'coeficiente'/'direcao'), enquanto os demais expõem
        importância por permutação/Gini (tuplas (nome, fração de 0 a 1) —
        cada um com sua própria unidade, por isso a formatação é decidida
        aqui e não no gerador de PDF.

        Parameters
        ----------
        tipo : str
            'arvore', 'logistica', 'knn', 'randomforest' ou 'svm'.
        dados : dict
            Relatório do explicador, com a chave 'importancias'.

        Returns
        -------
        list[tuple[str, str]]
            Pares (biomarcador, valor já formatado como texto).
        """
        itens = dados.get('importancias', [])
        if tipo == 'logistica':
            return [(d['feature'], f"{d['coeficiente']:+.2f} ({d['direcao']})") for d in itens]
        return [(nome, f"{valor * 100:.2f}%") for nome, valor in itens]

    def _nome_base_export(self) -> str:
        """Monta um nome de arquivo sugerido para a exportação (sem extensão)."""
        modelo = self.model_selector.get().lower().replace(" ", "_")
        modelo = modelo.replace("(", "").replace(")", "")
        return f"diagnostico_{modelo}"

    def _exportar_csv(self):
        """Exporta o DataFrame de resultado (Passo 3/4) para um arquivo CSV."""
        if self.df_resultado is None:
            return
        caminho = filedialog.asksaveasfilename(
            title="Exportar CSV", initialdir=resolve_reports_dir(),
            initialfile=f"{self._nome_base_export()}.csv",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"), ("Todos os arquivos", "*.*")),
        )
        if not caminho:
            return
        try:
            self.df_resultado.to_csv(caminho, index=True, index_label="paciente")
            self.lbl_export_hint.configure(text=f"CSV salvo em: {caminho}", text_color="#2ecc71")
        except Exception as e:
            self.lbl_export_hint.configure(text=f"Erro ao exportar CSV: {e}", text_color="#e74c3c")

    def _exportar_pdf_lote(self):
        """Exporta o resumo da sessão (metadados, importâncias e diagnóstico por paciente) em PDF."""
        if self.df_resultado is None:
            return
        caminho = filedialog.asksaveasfilename(
            title="Exportar PDF", initialdir=resolve_reports_dir(),
            initialfile=f"{self._nome_base_export()}.pdf",
            defaultextension=".pdf",
            filetypes=(("PDF", "*.pdf"), ("Todos os arquivos", "*.*")),
        )
        if not caminho:
            return
        try:
            total = len(self.df_resultado)
            if 'Diagnóstico_IA' in self.df_resultado.columns:
                malignos = int((self.df_resultado['Diagnóstico_IA'] == 'Maligno').sum())
                benignos = total - malignos
            else:
                malignos = benignos = None
            meta = {
                'arquivo': self.file_path_var.get().replace("Arquivo selecionado: ", ""),
                'modelo': self.model_selector.get(),
                'total': total, 'malignos': malignos, 'benignos': benignos,
            }
            nomes_modelo = {
                'arvore': self.NOME_ARVORE, 'logistica': self.NOME_LOGISTICA,
                'knn': self.NOME_KNN, 'randomforest': self.NOME_RF, 'svm': self.NOME_SVM,
            }
            importancias_por_modelo = {
                nomes_modelo.get(tipo, tipo): self._formatar_importancias(tipo, dados)
                for tipo, dados in self._ultima_explicacao.items()
            }
            export_batch_report(
                caminho, meta, self.df_resultado,
                importancias_por_modelo, self._ultima_acuracia,
            )
            self.lbl_export_hint.configure(text=f"PDF salvo em: {caminho}", text_color="#2ecc71")
        except Exception as e:
            self.lbl_export_hint.configure(text=f"Erro ao exportar PDF: {e}", text_color="#e74c3c")

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