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

    O customtkinter 5.x vincula apenas o evento ``<MouseWheel>`` (emitido no
    Windows e no macOS). No Linux/X11 a rolagem por roda gera ``<Button-4>``
    (cima) e ``<Button-5>`` (baixo), que ficavam sem tratamento — deixando a
    barra de rolagem inerte à roda. Esta subclasse acrescenta esses vínculos.

    A partir do customtkinter 6.0, o próprio widget já trata ``<Button-4>``/
    ``<Button-5>`` nativamente; nesse caso o workaround é desativado para
    evitar rolagem dupla. A detecção é por presença do método interno
    ``check_if_master_is_canvas`` (existente apenas nas versões 5.x).
    """

    def __init__(self, *args, **kwargs):
        """Inicializa o frame rolável e registra a roda do mouse no Linux (5.x)."""
        super().__init__(*args, **kwargs)
        if sys.platform.startswith("win") or sys.platform == "darwin":
            return
        # customtkinter >= 6.0 já trata a roda no Linux — nada a fazer.
        if not hasattr(self, "check_if_master_is_canvas"):
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


def responsive_geometry(window, width: int, height: int, margin: float = 0.9,
                         min_width: int = 480, min_height: int = 400):
    """
    Redimensiona e centraliza a janela dentro dos limites reais da tela.

    As janelas do app foram dimensionadas olhando para um monitor grande
    (1920x1080). Um ``geometry("1060x860")`` fixo abre maior que a tela
    inteira num notebook menor, empurrando parte do conteúdo para fora da
    área visível e exigindo rolagem extra (ou o gerenciador de janelas
    cortando a janela). Aqui, o tamanho "ideal" passado é usado como teto:
    a janela nunca ultrapassa uma fração da tela disponível.

    Parameters
    ----------
    window : ctk.CTk ou ctk.CTkToplevel
        Janela a redimensionar.
    width, height : int
        Tamanho ideal (o que já funciona bem em telas grandes).
    margin : float, optional
        Fração da tela a ocupar no máximo (padrão 90%) — deixa espaço para
        barra de tarefas/decorações do gerenciador de janelas.
    min_width, min_height : int, optional
        Tamanho mínimo absoluto, para a janela permanecer utilizável mesmo
        em telas muito pequenas.
    """
    window.update_idletasks()
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()

    largura = max(min_width, min(width, int(screen_w * margin)))
    altura = max(min_height, min(height, int(screen_h * margin)))

    x = max(0, (screen_w - largura) // 2)
    y = max(0, (screen_h - altura) // 2)

    window.geometry(f"{largura}x{altura}+{x}+{y}")
    window.minsize(min(min_width, largura), min(min_height, altura))


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
