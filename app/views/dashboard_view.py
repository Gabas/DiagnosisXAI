# app/views/dashboard_view.py
import customtkinter as ctk

class DashboardView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, corner_radius=10, fg_color="transparent", **kwargs)
        
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        self._setup_ui()

    def _setup_ui(self):
        title = ctk.CTkLabel(self, text="Visão Geral dos Dados", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 20), sticky="w")

        # Cards de Métricas
        card1 = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        card1.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card1, text="Total de Amostras", font=ctk.CTkFont(size=14)).pack(pady=(15, 5))
        ctk.CTkLabel(card1, text="569", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(0, 15))

        card2 = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        card2.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card2, text="Casos Benignos", font=ctk.CTkFont(size=14)).pack(pady=(15, 5))
        ctk.CTkLabel(card2, text="357", font=ctk.CTkFont(size=28, weight="bold"), text_color="#2ecc71").pack(pady=(0, 15))

        card3 = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        card3.grid(row=1, column=2, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(card3, text="Casos Malignos", font=ctk.CTkFont(size=14)).pack(pady=(15, 5))
        ctk.CTkLabel(card3, text="212", font=ctk.CTkFont(size=28, weight="bold"), text_color="#e74c3c").pack(pady=(0, 15))

        # Espaço para Gráficos
        chart_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray15"))
        chart_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=20, sticky="nsew")
        ctk.CTkLabel(chart_frame, text="[ Área reservada para renderização do gráfico UMAP ]", text_color="gray").place(relx=0.5, rely=0.5, anchor="center")