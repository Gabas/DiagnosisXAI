"""
Componentes e utilitários de interface reutilizáveis.

Concentra correções de comportamento da interface que precisam ser
compartilhadas entre várias telas — em especial o suporte à roda do mouse
no Linux, ausente no customtkinter 5.2.2.
"""

import sys
import customtkinter as ctk


class ScrollableFrame(ctk.CTkScrollableFrame):
    """
    CTkScrollableFrame com suporte à roda do mouse no Linux (X11).

    O customtkinter 5.2.2 vincula apenas o evento ``<MouseWheel>``, emitido
    no Windows e no macOS. No Linux/X11 a rolagem por roda gera os eventos
    ``<Button-4>`` (cima) e ``<Button-5>`` (baixo), que ficavam sem tratamento —
    deixando a barra de rolagem inerte à roda. Esta subclasse acrescenta esses
    vínculos sem alterar o comportamento nas demais plataformas.
    """

    def __init__(self, *args, **kwargs):
        """Inicializa o frame rolável e registra a roda do mouse no Linux."""
        super().__init__(*args, **kwargs)
        if sys.platform.startswith("win") or sys.platform == "darwin":
            return
        self.bind_all("<Button-4>", self._linux_mouse_wheel, add="+")
        self.bind_all("<Button-5>", self._linux_mouse_wheel, add="+")

    def _linux_mouse_wheel(self, event):
        """
        Rola o conteúdo quando a roda é girada sobre esta área.

        Replica a verificação de propriedade usada internamente pelo
        customtkinter: só atua se o cursor está sobre um widget pertencente
        a este canvas e se ainda há conteúdo a rolar.
        """
        if not self.check_if_master_is_canvas(event.widget):
            return
        if self._parent_canvas.yview() == (0.0, 1.0):
            return
        self._parent_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")


def bind_treeview_mousewheel(tree, rows: int = 3):
    """
    Habilita a rolagem por roda do mouse em um ``ttk.Treeview``.

    Interrompe a propagação do evento (``return "break"``) para que, quando o
    cursor estiver sobre a tabela, role as linhas da própria tabela em vez da
    página que a contém.

    Parameters
    ----------
    tree : tkinter.ttk.Treeview
        Tabela que receberá o suporte à roda do mouse.
    rows : int, optional
        Número de linhas roladas por entalhe da roda (padrão 3).
    """
    def _linux(event):
        tree.yview_scroll(-rows if event.num == 4 else rows, "units")
        return "break"

    def _win_mac(event):
        tree.yview_scroll(-rows if event.delta > 0 else rows, "units")
        return "break"

    tree.bind("<Button-4>", _linux, add="+")
    tree.bind("<Button-5>", _linux, add="+")
    tree.bind("<MouseWheel>", _win_mac, add="+")
