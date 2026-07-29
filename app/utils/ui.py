"""
Componentes e utilitários de interface reutilizáveis.

Concentra correções de comportamento da interface que precisam ser
compartilhadas entre várias telas — em especial o suporte à roda do mouse
no Linux, ausente no customtkinter 5.2.2.
"""

import sys
import tkinter as tk
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


def adicionar_barra_zoom(canvas, parent):
    """
    Adiciona a barra de navegação do matplotlib (zoom/pan/home/salvar) a um gráfico.

    Retorna a barra já criada com ``pack_toolbar=False`` para poder ser posicionada
    por ``grid`` pelo chamador. Dá ao usuário zoom e deslocamento interativos sobre
    uma figura embutida via ``FigureCanvasTkAgg``.

    Parameters
    ----------
    canvas : matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
        Canvas do gráfico a controlar.
    parent : tkinter widget
        Frame onde a barra será inserida.

    Returns
    -------
    matplotlib.backends.backend_tkagg.NavigationToolbar2Tk
        A barra criada (posicione-a com ``.grid(...)``).
    """
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

    barra = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    barra.update()
    try:
        barra.configure(background="#2b2b2b")
        for filho in barra.winfo_children():
            filho.configure(background="#2b2b2b")
    except Exception:
        pass  # estilo é cosmético; a funcionalidade não depende dele
    return barra


class HeadingTooltip:
    """
    Tooltip flutuante para os cabeçalhos de um ``ttk.Treeview``.

    O ``ttk.Treeview`` não expõe um widget por cabeçalho, então não dá para
    vincular um tooltip diretamente a cada coluna. Esta classe observa o
    movimento do mouse sobre a tabela, detecta quando o cursor está sobre a
    região de cabeçalho (``identify_region``), descobre a coluna
    (``identify_column``) e exibe, após um pequeno atraso, uma janelinha com o
    texto retornado por ``descricao(nome_coluna)``.

    Parameters
    ----------
    tree : tkinter.ttk.Treeview
        Tabela cujos cabeçalhos receberão tooltips.
    descricao : callable
        Função ``nome_coluna -> str | None``. Quando retorna ``None`` (coluna
        sem descrição conhecida), nenhum tooltip é exibido.
    delay : int, optional
        Atraso, em milissegundos, antes de mostrar o tooltip (padrão 450).
    wraplength : int, optional
        Largura máxima do texto, em pixels, antes de quebrar linha (padrão 340).
    """

    def __init__(self, tree, descricao, delay: int = 450, wraplength: int = 340):
        """Registra os vínculos de mouse na tabela e guarda a função de descrição."""
        self.tree = tree
        self.descricao = descricao
        self.delay = delay
        self.wraplength = wraplength
        self._tip = None
        self._after_id = None
        self._coluna_atual = None
        tree.bind("<Motion>", self._on_motion, add="+")
        tree.bind("<Leave>", self._on_leave, add="+")

    def _nome_coluna(self, event):
        """Nome da coluna sob o cursor se ele estiver sobre um cabeçalho, senão None."""
        if self.tree.identify_region(event.x, event.y) != "heading":
            return None
        col_id = self.tree.identify_column(event.x)  # ex.: '#1'
        try:
            idx = int(col_id[1:]) - 1
        except ValueError:
            return None
        colunas = self.tree["columns"]
        if 0 <= idx < len(colunas):
            return colunas[idx]
        return None

    def _on_motion(self, event):
        """Agenda/atualiza o tooltip conforme o cabeçalho sob o cursor muda."""
        nome = self._nome_coluna(event)
        if nome == self._coluna_atual:
            return
        self._hide()
        self._coluna_atual = nome
        texto = self.descricao(nome) if nome is not None else None
        if texto:
            self._cancel()
            x, y = event.x_root + 14, event.y_root + 20
            self._after_id = self.tree.after(self.delay, lambda: self._show(texto, x, y))

    def _show(self, texto: str, x: int, y: int):
        """Cria a janelinha do tooltip na posição indicada."""
        self._tip = tk.Toplevel(self.tree)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        self._tip.attributes("-topmost", True)
        tk.Label(
            self._tip, text=texto, justify="left", wraplength=self.wraplength,
            background="#1e1e1e", foreground="#e6e6e6", relief="solid", borderwidth=1,
            padx=8, pady=6,
        ).pack()

    def _hide(self):
        """Cancela o agendamento pendente e destrói o tooltip visível, se houver."""
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _cancel(self):
        """Cancela um agendamento de exibição ainda não disparado."""
        if self._after_id is not None:
            self.tree.after_cancel(self._after_id)
            self._after_id = None

    def _on_leave(self, _event):
        """Esconde o tooltip quando o cursor deixa a tabela."""
        self._hide()
        self._coluna_atual = None
