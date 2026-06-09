"""
Módulo contendo a interface do Dashboard inicial e informações de sistema.
"""

import customtkinter as ctk

class DashboardView(ctk.CTkFrame):
    """
    Frame responsável por exibir o status do sistema e o guia de uso.

    Apresenta cartões com o estado atual dos motores de IA, da base de
    conhecimento e fornece instruções rápidas de navegação.
    """

    def __init__(self, master, **kwargs):
        """
        Inicializa o frame do Dashboard e configura o layout de grelha.
        """
        super().__init__(master, corner_radius=10, fg_color="transparent", **kwargs)
        
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(3, weight=1)
        
        self._setup_ui()

    def _setup_ui(self):
        """
        Constrói os elementos visuais, cartões de status e o guia de instruções.
        """
        # Título e Subtítulo
        title = ctk.CTkLabel(self, text="Painel de Controlo XAI", font=ctk.CTkFont(size=28, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 5), sticky="w")
        
        subtitle = ctk.CTkLabel(self, text="Sistema de Apoio à Decisão Clínica com Inteligência Artificial Explicável", font=ctk.CTkFont(size=14), text_color="gray")
        subtitle.grid(row=1, column=0, columnspan=3, padx=20, pady=(0, 20), sticky="w")

        # --- Status do Sistema (Cards) ---
        card1 = ctk.CTkFrame(self, fg_color=("gray85", "#1f538d"))
        card1.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card1, text="🧠 Modelos de IA", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(card1, text="5 Algoritmos Carregados\n(RF, SVM, LR, KNN, DT)", font=ctk.CTkFont(size=13)).pack(pady=(0, 15))

        card2 = ctk.CTkFrame(self, fg_color=("gray85", "#27ae60"))
        card2.grid(row=2, column=1, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card2, text="🔬 Variáveis Clínicas", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(card2, text="30 Biomarcadores Ativos\n(Extração via FNA)", font=ctk.CTkFont(size=13)).pack(pady=(0, 15))

        card3 = ctk.CTkFrame(self, fg_color=("gray85", "#8e44ad"))
        card3.grid(row=2, column=2, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card3, text="📐 Padronização", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(card3, text="Z-Score Operacional\n(Ajustado à base WDBC)", font=ctk.CTkFont(size=13)).pack(pady=(0, 15))

        # --- Área de Conteúdo Inferior (Guia e Gráficos) ---
        info_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray15"))
        info_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=20, sticky="nsew")
        
        info_frame.grid_columnconfigure(0, weight=1)
        info_frame.grid_columnconfigure(1, weight=2) # Dá um pouco mais de espaço ao gráfico
        info_frame.grid_rowconfigure(0, weight=1)

        # Coluna Esquerda: Fluxo de Uso
        guia_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        guia_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        ctk.CTkLabel(guia_frame, text="Como utilizar o sistema:", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", pady=(0, 15))
        
        passos = [
            "1. Navegue até 'Novo Diagnóstico' no menu.",
            "2. Importe um lote de dados brutos (CSV).",
            "3. Aplique a padronização matemática.",
            "4. Selecione a IA ou realize um consenso.",
            "5. Execute a auditoria ou analise os dados."
        ]
        for passo in passos:
            ctk.CTkLabel(guia_frame, text=passo, font=ctk.CTkFont(size=14), justify="left").pack(anchor="w", pady=5)

        # Coluna Direita: Placeholder de Integração Futura (SHAP)
        shap_frame = ctk.CTkFrame(info_frame, fg_color=("gray80", "gray20"), corner_radius=8)
        shap_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(shap_frame, text="Impacto Global das Variáveis (SHAP)", font=ctk.CTkFont(size=16, weight="bold")).place(relx=0.5, rely=0.3, anchor="center")
        ctk.CTkLabel(shap_frame, text="[ Área reservada para Explicabilidade ]\n\nAqui será renderizado o resumo hierárquico\ndas características celulares mais relevantes\npara a tomada de decisão da IA.", text_color="gray", justify="center").place(relx=0.5, rely=0.6, anchor="center")