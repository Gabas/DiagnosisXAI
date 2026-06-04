# app/main.py
import customtkinter as ctk
from views.main_window import MainWindow

def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("green") 
    
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()