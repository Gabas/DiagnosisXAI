"""
Ponto de entrada principal do aplicativo DiagnosisXAI.

Este módulo inicializa as configurações globais de interface do 
CustomTkinter e executa o loop principal da janela da aplicação.
"""

import tkinter as tk
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
    _ajustar_escala_para_tela()

    app = MainWindow()
    app.mainloop()


def _ajustar_escala_para_tela():
    """
    Reduz a escala dos widgets em telas menores (ex.: notebooks).

    O layout foi desenhado olhando para um monitor grande (1920x1080). Nessa
    resolução os elementos cabem bem; num notebook com tela menor (altura
    ~768-900px), os mesmos elementos ocupam proporcionalmente mais espaço e
    forçam bem mais rolagem. Reduzir a escala global do customtkinter antes
    de criar qualquer janela compensa isso, mantendo as proporções do layout.
    """
    sonda = tk.Tk()
    sonda.withdraw()
    altura = sonda.winfo_screenheight()
    sonda.destroy()

    if altura <= 800:
        ctk.set_widget_scaling(0.8)
    elif altura <= 900:
        ctk.set_widget_scaling(0.9)

if __name__ == "__main__":
    main()