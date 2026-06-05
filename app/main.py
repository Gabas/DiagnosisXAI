"""
Ponto de entrada principal do aplicativo DiagnosisXAI.

Este módulo inicializa as configurações globais de interface do 
CustomTkinter e executa o loop principal da janela da aplicação.
"""

import customtkinter as ctk
from views.main_window import MainWindow

def main():
    """
    Inicializa e executa a aplicação gráfica.

    Configura o CustomTkinter para usar o modo escuro ("Dark") e 
    o tema de cores verde ("green"), instancia a janela principal 
    e inicia o loop de eventos principal do Tkinter.
    """
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("green") 
    
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()