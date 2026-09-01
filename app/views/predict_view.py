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
from core.committee import explicar as explicar_comite
from core.decision import (
    COLUNA_ZONA,
    ROTULO_BENIGNO,
    ROTULO_MALIGNO,
    ROTULO_REVISAR,
    ZONA_LIMITROFE,
    aplicar_a_explicacoes,
)
from core.history_manager import HistoryManager
from core.inference import ModelLoader
from core.metrics import analise_critica, avaliar_modelos, ressalvas_do_lote
from core.predictor import PredictorEngine
from utils.ui import ScrollableFrame, bind_treeview_mousewheel, HeadingTooltip
from utils.pdf_report import export_batch_report, resolve_reports_dir
from views.report_common import COR_BENIGNO, COR_MALIGNO, COR_REVISAR
from views.report_window import ReportWindow
from views.report_window_comite import ComiteReportWindow
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
        'comite': ComiteReportWindow,
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

    # Colunas da tabela de desempenho do Passo 4: (id, cabeçalho, largura em px).
    # A matriz de confusão (VP/FN/FP/VN) fica à direita porque é a origem de todas
    # as métricas — é ela que permite conferir os percentuais em casos pequenos.
    _COLUNAS_METRICAS = [
        ("modelo", "Modelo", 170),
        ("cobertura", "Cobertura", 90),
        ("acuracia", "Acurácia", 90),
        ("sensibilidade", "Sensibilidade", 110),
        ("especificidade", "Especificidade", 115),
        ("precisao", "Precisão", 90),
        ("f1", "F1", 80),
        ("vp", "VP", 45),
        ("fn", "FN", 45),
        ("fp", "FP", 45),
        ("vn", "VN", 45),
    ]

    # Definição de cada métrica, exibida como tooltip no cabeçalho da tabela.
    _DESCRICAO_METRICA = {
        "modelo": "Modelo de IA avaliado contra o gabarito deste lote.",
        "cobertura": "Fração do lote em que o modelo aceitou decidir. Abaixo de 100%, todas as "
                     "outras colunas descrevem apenas essa parte. Os casos devolvidos para "
                     "revisão não entram na matriz de confusão, pois não houve decisão a "
                     "pontuar. Decidir menos torna as demais métricas mais fáceis.",
        "acuracia": "Proporção de acertos no lote inteiro. Isolada, engana: num lote "
                    "majoritariamente benigno, um modelo que nunca acusa malignidade já "
                    "exibe acurácia alta.",
        "sensibilidade": "Dos tumores realmente malignos, quantos o modelo detectou "
                         "(revocação da classe Maligno). É a métrica crítica em rastreio: "
                         "o que ela não pega vira falso negativo.",
        "especificidade": "Dos tumores realmente benignos, quantos o modelo poupou de um "
                          "alarme falso (revocação da classe Benigno).",
        "precisao": "Dos pacientes que o modelo apontou como malignos, quantos eram de fato "
                    "malignos (valor preditivo positivo). Depende da prevalência do lote.",
        "f1": "Média harmônica entre precisão e sensibilidade. Resume num número só o quanto o "
              "modelo pega malignos sem alarmar benignos.",
        "vp": "Verdadeiros positivos: malignos corretamente identificados.",
        "fn": "Falsos negativos: malignos classificados como benignos. É o erro mais caro, "
              "porque o câncer passa sem nenhum sinal de alerta.",
        "fp": "Falsos positivos: benignos classificados como malignos. Custa exames "
              "adicionais e ansiedade, mas não deixa doença sem tratar.",
        "vn": "Verdadeiros negativos: benignos corretamente identificados.",
    }

    # Rótulos exibidos no menu de relatórios do Passo 5.
    _ROTULOS_RELATORIO = {
        'comite': "Comitê: concordância dos membros",
        'arvore': "Árvore de Decisão: regras",
        'logistica': "Regressão Logística: contribuições",
        'knn': "KNN: vizinhos",
        'randomforest': "Random Forest: consenso das árvores",
        'svm': "SVM: vetores de suporte",
        'shap_dt': "Árvore de Decisão: SHAP",
        'shap_rf': "Random Forest: SHAP",
        'shap_lr': "Regressão Logística: SHAP",
        'shap_svm': "SVM: SHAP",
        'shap_knn': "KNN: SHAP",
        'umap': "Mapa Populacional: UMAP",
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
        self._ultimas_metricas = {}
        self._probabilidades = {}
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
        modelos_disponiveis = [self.NOME_TODOS] + self.predictor.modelos_disponiveis()
        self.model_selector = ctk.CTkOptionMenu(ia_frame, values=modelos_disponiveis, state="disabled",
                                                command=self._ao_trocar_modelo)
        self.model_selector.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.btn_run = ctk.CTkButton(ia_frame, text="Processar Diagnóstico", state="disabled", command=self.process_batch, fg_color="#27ae60", hover_color="#2ecc71")
        self.btn_run.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # Opção de recusa: LIGADA por padrão. O erro que este sistema existe para
        # evitar é o falso negativo; diante de uma certeza que não separa as
        # classes, devolver o caso é mais seguro que arriscar um palpite. Quem
        # opera pode desligar — mas aí é escolha explícita, não silêncio do app.
        self.var_adiar = ctk.BooleanVar(value=True)
        # Preferência do usuário, preservada quando um modelo sem faixa calibrada
        # força a recusa a desligar: ao voltar para um modelo que a suporta, o
        # padrão seguro volta com ele.
        self._preferencia_adiar = True
        self.chk_adiar = ctk.CTkCheckBox(
            ia_frame, text="Adiar casos incertos em vez de decidir (padrão)",
            variable=self.var_adiar, command=self._ao_alternar_recusa,
            font=ctk.CTkFont(size=12))
        self.chk_adiar.grid(row=1, column=2, padx=(0, 20), pady=10, sticky="w")

        # Régua de decisão do modelo em vigor: as faixas de certeza e o que o
        # sistema faz em cada uma. Sem ela, uma certeza de 25% rotulada como
        # "Maligno" pareceria erro em vez de decisão deliberada — e "por que este
        # limiar?" ficaria sem resposta na tela.
        self.frm_regua = ctk.CTkFrame(ia_frame, fg_color="transparent")
        self.frm_regua.grid(row=2, column=0, columnspan=3, padx=20, pady=(0, 6), sticky="ew")
        # A folga do painel vai toda para a última coluna: sem isso, os textos
        # longos (a justificativa, que ocupa as três) esticariam também as
        # colunas do rótulo e da faixa, abrindo um vão no meio de cada linha.
        self.frm_regua.grid_columnconfigure(3, weight=1)
        self._ao_trocar_modelo(self.model_selector.get())

        # Vazio, este rótulo ainda ocuparia a altura de uma linha, abrindo um vão
        # entre a régua e os avisos que a seguem — por isso sai da grade quando
        # não há erro a mostrar (ver _mostrar_erro_inferencia).
        self.lbl_run_error = ctk.CTkLabel(ia_frame, text="", font=ctk.CTkFont(size=12), text_color="#e74c3c", justify="left")
        self.lbl_run_error.grid(row=3, column=0, columnspan=3, padx=20, pady=(0, 10), sticky="w")
        self.lbl_run_error.grid_remove()
        # Aviso de perfis atípicos (fora da distribuição de treino), preenchido após a inferência.
        self.lbl_ood = ctk.CTkLabel(ia_frame, text="", font=ctk.CTkFont(size=12), justify="left", wraplength=760)
        self.lbl_ood.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 4), sticky="w")
        # Aviso de decisões limítrofes (certeza calibrada perto do limiar de operação).
        self.lbl_limitrofe = ctk.CTkLabel(ia_frame, text="", font=ctk.CTkFont(size=12), justify="left", wraplength=760)
        self.lbl_limitrofe.grid(row=5, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")

        # --- Passo 4: Auditoria (Opcional) ---
        audit_frame = ctk.CTkFrame(container)
        audit_frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")
        audit_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(audit_frame, text="Passo 4: Auditoria Acadêmica (Opcional)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        self.btn_audit = ctk.CTkButton(audit_frame, text="Carregar Gabarito (CSV)", state="disabled", command=self.run_audit, fg_color="#8e44ad", hover_color="#9b59b6")
        self.btn_audit.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.lbl_audit_results = ctk.CTkLabel(audit_frame, text="", justify="left", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_audit_results.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # Tabela de desempenho por modelo (acurácia, sensibilidade, especificidade,
        # F1 e a matriz de confusão que os origina). Só aparece após a auditoria.
        self.tree_metricas = ttk.Treeview(
            audit_frame, columns=[c for c, _, _ in self._COLUNAS_METRICAS],
            show="headings", height=1)
        for coluna, titulo, largura in self._COLUNAS_METRICAS:
            self.tree_metricas.heading(coluna, text=titulo)
            self.tree_metricas.column(coluna, width=largura, anchor="center", stretch=False)
        self.tree_metricas.column("modelo", anchor="w")
        self.tree_metricas.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 8), sticky="ew")
        self.tree_metricas.grid_remove()
        # Tooltip nos cabeçalhos: a definição de cada métrica ao passar o mouse.
        self._metricas_tooltip = HeadingTooltip(self.tree_metricas, self._DESCRICAO_METRICA.get)

        # Leitura crítica por modelo: pontos fortes, ressalvas e veredito.
        self.txt_audit_critica = ctk.CTkTextbox(
            audit_frame, wrap="word", height=300, activate_scrollbars=True,
            font=ctk.CTkFont(size=12))
        self.txt_audit_critica.grid(row=3, column=0, columnspan=2, padx=20, pady=(0, 12), sticky="ew")
        self.txt_audit_critica.grid_remove()

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

        # Legenda das colunas de saída. Elas respondem a perguntas diferentes
        # (o que foi decidido, com quanta evidência, quão firme, e se a evidência
        # vale para este paciente) e, lado a lado, eram lidas como variações da
        # mesma coisa. O tooltip de cabeçalho continua existindo para o detalhe;
        # a legenda existe porque ninguém descobre um tooltip sem procurá-lo.
        self.lbl_legenda = ctk.CTkLabel(
            self.preview_frame, text="", font=ctk.CTkFont(size=11),
            text_color="gray", justify="left", wraplength=1000)
        self.lbl_legenda.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.lbl_legenda.grid_remove()

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
            self._mostrar_erro_inferencia("")
            self.btn_audit.configure(state="disabled")
            self._limpar_auditoria()
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
                self._mostrar_erro_inferencia("")
                self._atualizar_aviso_ood()
                self._atualizar_aviso_limitrofe()
                self._atualizar_legenda(modelo_escolhido)

                # Libera o Passo 4 após a IA rodar
                self.btn_audit.configure(state="normal")
                self._limpar_auditoria()

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
                    diagnosticos = self.df_resultado['Diagnóstico_IA']
                    malignos = int((diagnosticos == 'Maligno').sum())
                    benignos = int((diagnosticos == 'Benigno').sum())
                    adiados = int((diagnosticos == ROTULO_REVISAR).sum())
                else:
                    malignos = benignos = None
                    adiados = 0
                self._history_manager.save_session(
                    arquivo, modelo_escolhido, total, malignos, benignos,
                    self._relatorio_para_historico() or None, adiados,
                )
            except Exception as e:
                self._mostrar_erro_inferencia(f"Erro durante a inferência: {e}")

    def _atualizar_aviso_ood(self):
        """
        Atualiza o aviso de perfis atípicos (fora da distribuição de treino).

        Lê a coluna ``Perfil`` do resultado (preenchida pelo PredictorEngine) e
        resume quantos pacientes têm perfil atípico — casos em que a previsão
        extrapola para uma região pouco vista no treino e é menos confiável.
        """
        df = self.df_resultado
        if df is None or 'Perfil' not in df.columns:
            self.lbl_ood.configure(text="")
            return

        total = len(df)
        atipicos = int((df['Perfil'] == 'Atípico').sum())
        if atipicos:
            self.lbl_ood.configure(
                text=(f"⚠ {atipicos} de {total} paciente(s) com perfil atípico (fora da distribuição "
                      f"de treino). Interprete essas previsões com cautela."),
                text_color="#e67e22",
            )
        else:
            self.lbl_ood.configure(
                text=f"✓ Todos os {total} perfis dentro da distribuição de treino.",
                text_color="#2ecc71",
            )

    def _ao_trocar_modelo(self, modelo: str):
        """
        Redesenha a régua de decisão do modelo selecionado no Passo 3.

        Parameters
        ----------
        modelo : str
            Nome escolhido no seletor.
        """
        # A recusa exige faixa calibrada; nem todo modelo tem uma (o KNN só
        # consegue não errar adiando dois terços do lote — ver o script de
        # calibração), e a Árvore não tem probabilidade para comparar. Quando o
        # modelo não a suporta, a recusa desliga sem apagar a preferência.
        disponivel = self._recusa_disponivel(modelo)
        self.chk_adiar.configure(state="normal" if disponivel else "disabled")
        self.var_adiar.set(self._preferencia_adiar and disponivel)
        self.predictor.politica.adiar_incertos = bool(self.var_adiar.get())

        self._desenhar_regua(modelo, recusa_disponivel=disponivel)

    def _recusa_disponivel(self, modelo: str) -> bool:
        """True se o modelo escolhido pode adiar casos incertos."""
        politica = self.predictor.politica
        if modelo == self.NOME_TODOS:
            return any(politica.pode_adiar(m) for m in self.model_loader.models)
        return politica.pode_adiar(modelo)

    def _ao_alternar_recusa(self):
        """Registra a escolha do usuário sobre a recusa e redesenha a régua."""
        self._preferencia_adiar = bool(self.var_adiar.get())
        self._ao_trocar_modelo(self.model_selector.get())

    # --- Régua de decisão (Passo 3) ---------------------------------------

    # Cor de cada faixa da régua: a mesma que os relatórios usam para a classe,
    # para que "verde = liberado, laranja = não decidido, vermelho = acusado"
    # signifique o mesmo em todas as telas.
    _COR_FAIXA = {
        ROTULO_BENIGNO: COR_BENIGNO,
        ROTULO_MALIGNO: COR_MALIGNO,
        ROTULO_REVISAR: COR_REVISAR,
        ZONA_LIMITROFE: COR_REVISAR,
    }

    def _desenhar_regua(self, modelo: str, recusa_disponivel: bool):
        """
        Monta o painel que mostra, por extenso, como o modelo decide.

        São duas perguntas, e a tela responde às duas: *em que faixa de certeza
        cada resposta é dada* (a régua propriamente dita) e *por que os cortes
        estão ali* (a justificativa embaixo). Para "Todos (Comparação)" a régua
        vira uma tabela — um modelo por linha —, que é a forma de comparar os
        limiares lado a lado.

        Parameters
        ----------
        modelo : str
            Nome escolhido no seletor.
        recusa_disponivel : bool
            Se o modelo suporta adiar casos incertos; quando não, a tela diz
            por quê, para a caixa desabilitada não parecer defeito.
        """
        for filho in self.frm_regua.winfo_children():
            filho.destroy()

        if modelo == self.NOME_TODOS:
            linha = self._regua_comparativa()
        elif modelo in self.predictor.MODELOS_SEM_CERTEZA:
            linha = self._regua_sem_certeza(modelo)
        else:
            linha = self._regua_de_um_modelo(modelo)

        if not recusa_disponivel and modelo not in self.predictor.MODELOS_SEM_CERTEZA:
            self._nota_regua(
                linha,
                "Adiar casos incertos não está disponível aqui: a calibração não encontrou, "
                "para este modelo, uma faixa que valha a pena adiar (ver "
                "scripts/calibrar_limiares.py). Ele decide todos os casos.")

    def _titulo_regua(self, linha: int, texto: str) -> int:
        """Escreve o título do painel e devolve a próxima linha livre."""
        ctk.CTkLabel(self.frm_regua, text=texto,
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=linha, column=0, columnspan=4, pady=(0, 2), sticky="w")
        return linha + 1

    def _nota_regua(self, linha: int, texto: str) -> int:
        """Escreve um parágrafo cinza abaixo da régua e devolve a próxima linha."""
        ctk.CTkLabel(self.frm_regua, text=texto, font=ctk.CTkFont(size=11),
                     text_color="gray", justify="left", wraplength=900).grid(
            row=linha, column=0, columnspan=4, pady=(4, 0), sticky="w")
        return linha + 1

    def _regua_de_um_modelo(self, modelo: str) -> int:
        """Desenha a régua de um modelo só (uma linha por faixa de certeza)."""
        politica = self.predictor.politica
        linha = self._titulo_regua(0, f"Régua de decisão: {modelo}")

        for faixa in politica.regua(modelo):
            cor = self._COR_FAIXA.get(faixa['rotulo'], "gray")
            ctk.CTkLabel(self.frm_regua, text=f"■ {faixa['rotulo']}", text_color=cor,
                         font=ctk.CTkFont(size=12, weight="bold"), anchor="w",
                         width=110).grid(row=linha, column=0, sticky="w")
            ctk.CTkLabel(self.frm_regua, text=faixa['faixa'], font=ctk.CTkFont(size=12),
                         anchor="w", width=150).grid(row=linha, column=1, sticky="w")
            ctk.CTkLabel(self.frm_regua, text=faixa['efeito'], font=ctk.CTkFont(size=11),
                         text_color="gray", anchor="w", justify="left",
                         wraplength=620).grid(row=linha, column=2, sticky="w")
            linha += 1

        return self._nota_regua(linha, politica.justificativa(modelo))

    def _regua_comparativa(self) -> int:
        """
        Desenha os cortes de todos os modelos numa tabela, para comparação.

        No modo "Todos" não há uma régua só: cada modelo tem a sua, e é
        exatamente a diferença entre elas que a comparação existe para mostrar.
        """
        politica = self.predictor.politica
        linha = self._titulo_regua(0, "Limiares de decisão: um por modelo")

        cabecalho = ("Modelo", "Decide Benigno", "Faixa incerta", "Decide Maligno")
        for coluna, texto in enumerate(cabecalho):
            ctk.CTkLabel(self.frm_regua, text=texto, font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="gray", anchor="w",
                         width=(190 if coluna == 0 else 150)).grid(
                row=linha, column=coluna, sticky="w")
        linha += 1

        for nome in self.model_loader.models:
            celulas = self._celulas_da_regua(nome)
            ctk.CTkLabel(self.frm_regua, text=nome, font=ctk.CTkFont(size=12),
                         anchor="w").grid(row=linha, column=0, sticky="w")
            for coluna, (texto, cor) in enumerate(celulas, start=1):
                ctk.CTkLabel(self.frm_regua, text=texto, font=ctk.CTkFont(size=12),
                             text_color=cor, anchor="w").grid(
                    row=linha, column=coluna, sticky="w")
            linha += 1

        criterio = ("Cada modelo tem o seu corte porque cada um calibra as probabilidades à sua "
                    "maneira; o critério é o mesmo para todos, e está descrito ao selecionar um "
                    "modelo específico." if politica.calibrada else
                    "Sem data/limiares.json, todos os modelos operam no corte padrão de 50% do "
                    "scikit-learn.")
        return self._nota_regua(linha, criterio)

    def _celulas_da_regua(self, modelo: str) -> list:
        """
        As três células (Benigno, faixa incerta, Maligno) de um modelo na tabela.

        Returns
        -------
        list[tuple[str, str]]
            Pares (texto, cor) na ordem das colunas.
        """
        politica = self.predictor.politica
        if modelo in self.predictor.MODELOS_SEM_CERTEZA:
            return [("—", "gray"), ("sem limiar: decide por regras", "gray"), ("—", "gray")]

        faixas = {f['rotulo']: f for f in politica.regua(modelo)}
        incerta = faixas.get(ROTULO_REVISAR) or faixas.get(ZONA_LIMITROFE)
        sufixo = f" ({incerta['rotulo']})" if incerta else ""
        return [
            (faixas[ROTULO_BENIGNO]['faixa'] if ROTULO_BENIGNO in faixas else "—", COR_BENIGNO),
            (f"{incerta['faixa']}{sufixo}" if incerta else "—", COR_REVISAR),
            (faixas[ROTULO_MALIGNO]['faixa'] if ROTULO_MALIGNO in faixas else "—", COR_MALIGNO),
        ]

    def _regua_sem_certeza(self, modelo: str) -> int:
        """Painel do modelo que não tem probabilidade para comparar a limiar algum."""
        linha = self._titulo_regua(0, f"Régua de decisão: {modelo}")
        return self._nota_regua(
            linha,
            f"A {modelo} não tem limiar ajustável. Suas folhas são puras, então a "
            f"probabilidade sai sempre 0% ou 100% e não há o que deslocar. Ela decide pela "
            f"sequência de regras que aprendeu, e por isso não recebe as colunas de certeza "
            f"e de zona, nem entra no comitê.")

    def _mostrar_erro_inferencia(self, texto: str):
        """Exibe (ou esconde, quando ``texto`` é vazio) o erro do Passo 3."""
        self.lbl_run_error.configure(text=texto)
        if texto:
            self.lbl_run_error.grid()
        else:
            self.lbl_run_error.grid_remove()

    def _atualizar_legenda(self, modelo: str):
        """
        Escreve, acima da tabela, o que cada coluna de saída significa.

        Parameters
        ----------
        modelo : str
            Nome escolhido no seletor, para citar os cortes em vigor.
        """
        df = self.df_resultado
        if df is None:
            self.lbl_legenda.grid_remove()
            return

        if 'Diagnóstico_IA' not in df.columns:
            # Modo de comparação: uma coluna por modelo, cada uma com o corte do
            # seu próprio modelo — que é justamente o que a tabela acima lista.
            self.lbl_legenda.configure(
                text=("Colunas:   IA_XXX: o que cada modelo entregaria para o paciente, cada um "
                      "pelo seu próprio limiar (a tabela do Passo 3 lista todos)   ·   "
                      "Perfil: se o paciente se parece com os casos de treino "
                      "(\"Atípico\" quer dizer que o modelo está extrapolando)"))
            self.lbl_legenda.grid()
            return

        partes = ["Diagnóstico_IA: o que o sistema entrega para o paciente"]
        if 'Certeza_Maligno(%)' in df.columns:
            regra = self.predictor.politica.regra(modelo)
            partes.append(f"Certeza_Maligno(%): a evidência, lida nesta régua: {regra}")
        if COLUNA_ZONA in df.columns:
            partes.append(f"{COLUNA_ZONA}: em qual dessas faixas a certeza caiu")
        if 'Perfil' in df.columns:
            partes.append("Perfil: se o paciente se parece com os casos de treino "
                          "(\"Atípico\" quer dizer que o modelo está extrapolando)")

        self.lbl_legenda.configure(text="Colunas:   " + "   ·   ".join(partes))
        self.lbl_legenda.grid()

    def _atualizar_aviso_limitrofe(self):
        """
        Resume, em uma linha, quantos casos caíram na faixa incerta da régua.

        Lê a coluna de zona do resultado (preenchida pelo PredictorEngine) e
        conta os que o modelo devolveu ou marcou como limítrofes. O texto cita a
        faixa em números, para que o total apareça ao lado do critério que o
        produziu. Quando o modelo não fornece certeza (ex.: Árvore de Decisão),
        o aviso fica vazio.
        """
        df = self.df_resultado
        if df is None or COLUNA_ZONA not in df.columns:
            self.lbl_limitrofe.configure(text="")
            return

        modelo = self.model_selector.get()
        inferior, superior = self.predictor.politica.faixa_incerta(modelo)
        faixa = f"{inferior * 100:.1f}% a {superior * 100:.1f}%"

        total = len(df)
        adiados = int((df[COLUNA_ZONA] == ROTULO_REVISAR).sum())
        if adiados:
            # Com a recusa ligada, o adiamento substitui o aviso de limítrofe:
            # não há decisão a rever, e sim decisão nenhuma.
            self.lbl_limitrofe.configure(
                text=(f"⏸ {adiados} de {total} caso(s) devolvido(s) para revisão. A certeza ficou "
                      f"dentro da faixa incerta ({faixa}), onde o modelo não decide. Os outros "
                      f"{total - adiados} receberam diagnóstico."),
                text_color="#e67e22",
            )
            return

        limitrofes = int((df[COLUNA_ZONA] == ZONA_LIMITROFE).sum())
        if limitrofes:
            self.lbl_limitrofe.configure(
                text=(f"⚠ {limitrofes} de {total} caso(s) limítrofe(s). A certeza ficou dentro de "
                      f"{faixa}, perto demais do limiar para a decisão ser firme. "
                      f"Recomenda-se revisão."),
                text_color="#e67e22",
            )
        else:
            self.lbl_limitrofe.configure(
                text=(f"✓ Nenhum dos {total} casos caiu na faixa incerta ({faixa}). "
                      f"Todas as decisões saíram com folga."),
                text_color="#2ecc71",
            )

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
        self._probabilidades = {}
        self.report_menu.configure(values=["—"], state="disabled")
        self.report_menu.set("—")
        self.btn_abrir_relatorio.configure(state="disabled")
        self.lbl_report_hint.configure(
            text="Disponível após processar o diagnóstico.", text_color="gray")

    def _modelos_explicados(self, modelo_escolhido: str) -> set:
        """
        Modelos cujos relatórios devem ser preparados para a escolha do Passo 3.

        Um único modelo explica a si mesmo; "Todos (Comparação)" explica os
        cinco; e o comitê explica cada um dos seus membros — é assim que a
        decisão do voto suave fica auditável, já que o comitê não tem
        explicador próprio: quem decidiu foram os membros.

        Parameters
        ----------
        modelo_escolhido : str
            Nome selecionado no Passo 3.

        Returns
        -------
        set[str]
            Nomes dos modelos a explicar.
        """
        if modelo_escolhido == self.NOME_TODOS:
            return set(self.model_loader.models)
        if modelo_escolhido == self.predictor.NOME_COMITE:
            # O comitê explica a si mesmo (a concordância entre os membros) e
            # empresta os relatórios individuais para detalhar cada membro.
            return set(self.predictor.membros_comite()) | {self.predictor.NOME_COMITE}
        return {modelo_escolhido}

    def _gerar_exato(self, tipo: str, nome_modelo: str, funcao):
        """
        Gera o relatório exato de um modelo, isolando eventuais falhas.

        Após gerar, alinha as explicações à decisão que a tabela do Passo 3
        exibe (classe, probabilidade e marcação de limítrofe vindas da
        probabilidade calibrada e do limiar de operação) — sem isso, relatório
        e tabela discordariam em toda a faixa entre o limiar e 50%.

        Se o explicador estiver indisponível (por exemplo, um .pkl regenerado
        com código antigo), o problema é registrado e o processamento segue —
        sem derrubar os demais relatórios do Passo 5.

        Parameters
        ----------
        tipo : str
            'arvore', 'logistica', 'knn', 'randomforest' ou 'svm'.
        nome_modelo : str
            Nome do modelo correspondente, para resolver o limiar de operação.
        funcao : callable
            Função sem argumentos que produz o dicionário do relatório.
        """
        try:
            dados = funcao()
            aplicar_a_explicacoes(
                dados.get('explicacoes', []), self._probabilidades.get(nome_modelo),
                self.predictor.politica, nome_modelo)
            self._ultima_explicacao[tipo] = dados
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

        alvos = self._modelos_explicados(modelo_escolhido)
        explicadores = self.model_loader.explainers

        # Probabilidades calibradas de cada modelo explicado: é com elas que as
        # janelas do Passo 5 exibem a mesma decisão da tabela do Passo 3.
        self._probabilidades = self.predictor.probabilidades_calibradas(
            self.df_padronizado, self.df_limpo, alvos)

        # Cada explicador é isolado: se um estiver indisponível (ex.: .pkl antigo
        # com bug), os demais relatórios continuam funcionando.
        # O comitê não vem do .pkl: é explicado a partir das probabilidades dos
        # membros, que acabaram de ser calculadas acima.
        comite = self.predictor.NOME_COMITE
        if comite in alvos:
            membros = self.predictor.membros_comite()
            self._gerar_exato('comite', comite, lambda: explicar_comite(
                {m: self._probabilidades[m] for m in membros},
                list(self.df_padronizado.index), self.predictor.politica, comite))

        exp = explicadores.get('arvore')
        if exp is not None and self.NOME_ARVORE in alvos:
            self._gerar_exato('arvore', self.NOME_ARVORE, lambda: {
                'importancias': self._importancias_arvore(exp),
                'explicacoes': exp.explain(self.df_limpo),
            })
        exp = explicadores.get('logistica')
        if exp is not None and self.NOME_LOGISTICA in alvos:
            self._gerar_exato('logistica', self.NOME_LOGISTICA, lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado, self.df_limpo),
            })
        exp = explicadores.get('knn')
        if exp is not None and self.NOME_KNN in alvos:
            self._gerar_exato('knn', self.NOME_KNN, lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado),
                'contexto': exp.contexto(),
            })
        exp = explicadores.get('randomforest')
        if exp is not None and self.NOME_RF in alvos:
            self._gerar_exato('randomforest', self.NOME_RF, lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado),
                'contexto': exp.contexto(),
            })
        exp = explicadores.get('svm')
        if exp is not None and self.NOME_SVM in alvos:
            self._gerar_exato('svm', self.NOME_SVM, lambda: {
                'importancias': exp.global_importances(top_n=10),
                'explicacoes': exp.explain(self.df_padronizado),
                'contexto': exp.contexto(),
                'batch_2d': self._batch_2d_para_svm(),
            })

        # SHAP: disponível para cada modelo executado que tenha artefatos salvos.
        if self.model_loader.shap_background is not None:
            for nome_modelo, key in self._MODELO_KEY.items():
                if nome_modelo in alvos and key in self.model_loader.shap_importances:
                    self._shap_disponiveis.append(key)

        # Monta as opções do menu: exatos primeiro, depois SHAP.
        opcoes = {}
        for tipo in ('comite', 'arvore', 'logistica', 'knn', 'randomforest', 'svm'):
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
                text=f"{len(labels)} relatórios prontos. Escolha um e clique em Abrir.",
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

        # Ganho de entropia bruto por atributo: soma, em cada nó de decisão, da
        # redução ponderada de entropia (impureza) do split. Como criterion='entropy',
        # impureza[node] é a entropia do nó; este acumulado é o ganho de informação
        # total que o atributo trouxe à árvore.
        ganho = np.zeros(len(exp.feature_names))
        for node in range(len(feat)):
            esq = cl[node]
            if esq != -1:  # nó de decisão (não folha)
                dir_ = cr[node]
                ganho[feat[node]] += (wn[node] * impureza[node]
                                      - wn[esq] * impureza[esq]
                                      - wn[dir_] * impureza[dir_])
        total = ganho.sum() or 1.0

        # (atributo, participação normalizada, ganho de entropia bruto)
        trios = [(exp.feature_names[i], float(ganho[i] / total), float(ganho[i]))
                 for i in range(len(ganho)) if ganho[i] > 0]
        trios.sort(key=lambda t: t[1], reverse=True)
        return trios[:10]

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
        Monta a lista {'indice', 'classe', ...} do lote a partir do resultado da IA.

        Para um único modelo usa a coluna 'Diagnóstico_IA'; no modo de comparação
        ('Todos') usa o voto da maioria entre as colunas dos modelos. Quando
        presentes, os sinais de confiabilidade (certeza calibrada, perfil atípico
        e decisão limítrofe) são incluídos para enriquecer o mapa interativo.

        Returns
        -------
        list[dict]
            Um item por paciente, com índice, classe prevista e, quando houver,
            'certeza'/'perfil'/'decisao'.
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

        certezas = df['Certeza_Maligno(%)'] if 'Certeza_Maligno(%)' in df.columns else None
        perfis = df['Perfil'] if 'Perfil' in df.columns else None
        decisoes = df[COLUNA_ZONA] if COLUNA_ZONA in df.columns else None

        pacientes = []
        for i in range(len(indices)):
            p = {'indice': int(indices[i]), 'classe': classes[i]}
            if certezas is not None:
                p['certeza'] = f"{certezas.iloc[i]:.2f}%"
            if perfis is not None:
                p['perfil'] = perfis.iloc[i]
            if decisoes is not None:
                p['decisao'] = decisoes.iloc[i]
            pacientes.append(p)
        return pacientes

    def run_audit(self):

        """Abre o CSV com diagnósticos reais, adiciona à tabela e avalia os modelos."""
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
                self._atualizar_legenda(self.model_selector.get())

                # 3. Avalia cada modelo contra o gabarito (matriz de confusão e derivadas)
                metricas = avaliar_modelos(
                    self.df_resultado, nome_modelo=self.model_selector.get())
                if not metricas:
                    self.lbl_audit_results.configure(
                        text="Erro: o lote não tem coluna de diagnóstico para auditar.",
                        text_color="#e74c3c")
                    return

                self._ultimas_metricas = metricas
                self._preencher_tabela_metricas(metricas)
                self._preencher_critica(metricas)

                # A composição do lote é a mesma para todos os modelos — basta um.
                resumo = metricas[next(iter(metricas))]
                self.lbl_audit_results.configure(
                    text=(f"Gabarito conferido: {resumo['n']} pacientes "
                          f"({resumo['vp'] + resumo['fn']} malignos, "
                          f"{resumo['vn'] + resumo['fp']} benignos)."),
                    text_color="#2ecc71")

                # A acurácia (só ela) segue para o histórico e o PDF, cujo formato
                # é {modelo: percentual} — as demais métricas ficam nesta tela.
                acuracia = {nome: round(m['acuracia'], 2) for nome, m in metricas.items()}
                self._ultima_acuracia = acuracia
                self._history_manager.update_last_accuracy(acuracia)

            except Exception as e:
                self.lbl_audit_results.configure(text=f"Erro na auditoria: {e}", text_color="#e74c3c")

    def _preencher_tabela_metricas(self, metricas_por_modelo: dict):
        """
        Preenche e exibe a tabela de desempenho do Passo 4.

        Parameters
        ----------
        metricas_por_modelo : dict
            Saída de ``core.metrics.avaliar_modelos`` — já ordenada da maior
            para a menor sensibilidade.
        """
        self.tree_metricas.delete(*self.tree_metricas.get_children())

        for nome, m in metricas_por_modelo.items():
            self.tree_metricas.insert("", "end", values=(
                nome, self._pct(m.get('cobertura')),
                self._pct(m['acuracia']), self._pct(m['sensibilidade']),
                self._pct(m['especificidade']), self._pct(m['precisao']), self._pct(m['f1']),
                m['vp'], m['fn'], m['fp'], m['vn'],
            ))

        self.tree_metricas.configure(height=max(len(metricas_por_modelo), 1))
        self.tree_metricas.grid()

    @staticmethod
    def _pct(valor) -> str:
        """Formata uma proporção percentual, marcando as indefinidas com '—'."""
        return "—" if valor is None else f"{valor:.1f}%"

    def _preencher_critica(self, metricas_por_modelo: dict):
        """
        Escreve a leitura crítica de cada modelo na caixa de texto do Passo 4.

        Para cada modelo: pontos fortes (do algoritmo e do que ele fez neste
        lote), ressalvas (erros cometidos, assimetria entre os dois tipos de
        erro, incerteza amostral) e um veredito de uso. Ao final, as ressalvas
        metodológicas que valem para a auditoria inteira.

        Parameters
        ----------
        metricas_por_modelo : dict
            Saída de ``core.metrics.avaliar_modelos``.
        """
        linhas = []
        for nome, m in metricas_por_modelo.items():
            critica = analise_critica(nome, m, metricas_por_modelo)
            linhas.append(f"■ {nome}")
            cobertura = (f"   decidiu {m['adiados'] and self._pct(m['cobertura']) or '100.0%'} "
                         f"do lote  ·  " if m.get('adiados') else "   ")
            linhas.append(f"{cobertura}acurácia {self._pct(m['acuracia'])}"
                          f"  ·  sensibilidade {self._pct(m['sensibilidade'])}"
                          f"  ·  especificidade {self._pct(m['especificidade'])}"
                          f"  ·  F1 {self._pct(m['f1'])}")
            if m['ic_sensibilidade']:
                linhas.append(f"   IC 95% da sensibilidade: "
                              f"[{m['ic_sensibilidade'][0]:.1f}%, {m['ic_sensibilidade'][1]:.1f}%]"
                              f"  ·  da especificidade: "
                              f"[{m['ic_especificidade'][0]:.1f}%, {m['ic_especificidade'][1]:.1f}%]")

            linhas.append("   Pontos fortes:")
            linhas.extend(f"      • {t}" for t in critica['fortes'])
            linhas.append("   Ressalvas:")
            linhas.extend(f"      • {t}" for t in critica['ressalvas'])
            linhas.append(f"   Veredito: {critica['veredito']}")
            linhas.append("")

        ressalvas = ressalvas_do_lote(metricas_por_modelo, self.predictor.politica)
        if ressalvas:
            linhas.append("■ Limites desta auditoria")
            linhas.extend(f"      • {t}" for t in ressalvas)

        self.txt_audit_critica.configure(state="normal")
        self.txt_audit_critica.delete("1.0", "end")
        self.txt_audit_critica.insert("1.0", "\n".join(linhas))
        self.txt_audit_critica.configure(state="disabled")
        self.txt_audit_critica.grid()

    def _limpar_auditoria(self):
        """Esconde a tabela e a leitura crítica, zerando o estado do Passo 4."""
        self._ultima_acuracia = None
        self._ultimas_metricas = {}
        self.lbl_audit_results.configure(text="")
        self.tree_metricas.delete(*self.tree_metricas.get_children())
        self.tree_metricas.grid_remove()
        self.txt_audit_critica.configure(state="normal")
        self.txt_audit_critica.delete("1.0", "end")
        self.txt_audit_critica.configure(state="disabled")
        self.txt_audit_critica.grid_remove()

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
        if tipo == 'arvore':
            # (nome, participação, ganho_entropia) — mostra a redução de entropia.
            return [(item[0], f"Δentropia {item[2]:.3f}" if len(item) > 2
                     else f"{item[1] * 100:.2f}%") for item in itens]
        return [(item[0], f"{item[1] * 100:.2f}%") for item in itens]

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
            adiados = 0
            if 'Diagnóstico_IA' in self.df_resultado.columns:
                diagnosticos = self.df_resultado['Diagnóstico_IA']
                malignos = int((diagnosticos == 'Maligno').sum())
                # Contado explicitamente, e não por diferença: com a recusa
                # ligada há uma terceira saída, e "o resto é benigno" liberaria
                # no relatório pacientes que o modelo não chegou a avaliar.
                benignos = int((diagnosticos == 'Benigno').sum())
                adiados = int((diagnosticos == ROTULO_REVISAR).sum())
            else:
                malignos = benignos = None
            modelo = self.model_selector.get()
            meta = {
                'arquivo': self.file_path_var.get().replace("Arquivo selecionado: ", ""),
                'modelo': modelo,
                'total': total, 'malignos': malignos, 'benignos': benignos,
                'adiados': adiados,
            }
            # O ponto de operação viaja junto: fora do app, uma linha "Maligno,
            # certeza 25%" só é conferível se o corte que a gerou estiver na
            # mesma folha. Modelos sem probabilidade (Árvore) não têm régua.
            if modelo not in self.predictor.MODELOS_SEM_CERTEZA and modelo != self.NOME_TODOS:
                politica = self.predictor.politica
                meta['regua'] = politica.regua(modelo)
                meta['justificativa'] = politica.justificativa(modelo)
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
        # A legenda descreve as colunas de saída; ela volta (via
        # _atualizar_legenda) quando o que se está mostrando é o resultado da IA.
        self.lbl_legenda.grid_remove()

        self.tree.delete(*self.tree.get_children())
        colunas = list(df.columns)
        self.tree["columns"] = colunas
        
        for col in colunas:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, minwidth=130, stretch=False, anchor="center") 
            
        for indice, linha in df.iterrows():
            valores = [round(val, 4) if isinstance(val, float) else val for val in linha]
            self.tree.insert("", "end", values=valores)