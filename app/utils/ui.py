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


# Altura de tela para a qual o layout dos relatórios foi desenhado. Serve de
# referência para encolher figuras, listas e tabelas em telas menores — não é um
# mínimo: telas maiores simplesmente não recebem ampliação.
ALTURA_REFERENCIA = 1200

# Piso do encolhimento: abaixo disso os gráficos ficariam ilegíveis, e é a
# rolagem (não a redução) que passa a garantir o acesso ao conteúdo.
FATOR_MINIMO = 0.6


def fator_tela(window, altura_tela: int = None) -> float:
    """
    Quanto o conteúdo desta janela deve encolher para caber na tela atual.

    Parameters
    ----------
    window : tkinter widget
        Qualquer widget da janela (usado só para consultar a tela).
    altura_tela : int ou None, optional
        Altura da tela em pixels. Informe apenas em testes; em uso normal é
        lida do próprio sistema.

    Returns
    -------
    float
        Fator entre ``FATOR_MINIMO`` e 1.0.
    """
    altura = altura_tela or window.winfo_screenheight()
    return max(FATOR_MINIMO, min(1.0, altura / ALTURA_REFERENCIA))


def figura_responsiva(window, largura: float, altura: float,
                      altura_tela: int = None) -> tuple:
    """
    Adapta o ``figsize`` de um gráfico matplotlib à tela.

    Uma figura tem tamanho fixo em polegadas: num notebook de 768px de altura,
    os 4,2 polegadas (420px) usados no projeto consomem sozinhos mais de metade
    da janela e empurram a área mestre-detalhe para fora dela.

    Parameters
    ----------
    window : tkinter widget
        Janela que hospedará o gráfico.
    largura, altura : float
        Tamanho ideal em polegadas (o que funciona bem em tela grande).
    altura_tela : int ou None, optional
        Apenas para testes.

    Returns
    -------
    tuple[float, float]
        ``figsize`` já reduzido. A largura encolhe menos que a altura: a falta
        de espaço é sobretudo vertical, e achatar a largura prejudicaria os
        rótulos do eixo x.
    """
    fator = fator_tela(window, altura_tela)
    return (largura * (1.0 + fator) / 2, altura * fator)


def itens_visiveis(window, padrao: int, minimo: int = 5,
                   altura_tela: int = None) -> int:
    """
    Quantos itens de uma lista vertical (ex.: barras de importância) exibir.

    Parameters
    ----------
    window : tkinter widget
        Janela que hospedará a lista.
    padrao : int
        Quantidade ideal, usada em telas grandes.
    minimo : int, optional
        Piso — abaixo disso o ranking deixaria de ser informativo.
    altura_tela : int ou None, optional
        Apenas para testes.

    Returns
    -------
    int
        Quantidade a exibir nesta tela.
    """
    return max(minimo, round(padrao * fator_tela(window, altura_tela)))


def ajustar_ao_conteudo(window, conteudo=None, min_width: int = 480,
                        min_height: int = 400, margem: int = 24):
    """
    Redimensiona a janela para o tamanho que o conteúdo realmente pede.

    Complementa :func:`responsive_geometry`, que só conhece um tamanho "ideal"
    fixado no código — e que pode ficar aquém do conteúdo real (o relatório do
    SVM, por exemplo, pedia 1084px de largura numa janela aberta com 1060, e a
    diferença era simplesmente cortada). Aqui o tamanho pedido é medido depois
    de construída a interface e continua limitado pela tela; o que exceder fica
    acessível pela rolagem.

    Chame depois de montar todo o conteúdo.

    Parameters
    ----------
    window : ctk.CTk ou ctk.CTkToplevel
        Janela a ajustar.
    conteudo : tkinter widget ou None, optional
        Widget a medir. Informe o corpo rolável quando houver um: uma área de
        rolagem anuncia o tamanho *dela*, não o do que carrega dentro, e medir
        a janela devolveria algo como 223x212 em vez do conteúdo real.
    min_width, min_height : int, optional
        Tamanho mínimo utilizável.
    margem : int, optional
        Folga somada ao tamanho pedido, para as bordas do gerenciador de janelas.
    """
    window.update_idletasks()
    alvo = conteudo if conteudo is not None else window
    responsive_geometry(
        window,
        alvo.winfo_reqwidth() + margem,
        alvo.winfo_reqheight() + margem,
        min_width=min_width, min_height=min_height,
    )


def quebra_automatica(label, margem: int = 40, minimo: int = 200):
    """
    Faz um rótulo quebrar a linha na largura que ele realmente tem.

    ``wraplength`` é fixado em pixels, então um texto ajustado para uma janela
    larga continua exigindo aquela largura numa janela estreita — e o excedente
    é cortado horizontalmente, já que as telas do app só rolam na vertical.
    Aqui a quebra passa a acompanhar a largura do widget a cada redimensionamento.

    Parameters
    ----------
    label : ctk.CTkLabel
        Rótulo de texto longo, posicionado com ``fill="x"`` ou ``sticky="ew"``.
    margem : int, optional
        Pixels descontados da largura (paddings internos do contêiner).
    minimo : int, optional
        Largura mínima de quebra, para o texto não virar uma coluna de palavras
        soltas em janelas muito estreitas.

    Returns
    -------
    O próprio rótulo, para permitir encadeamento com ``.pack()``/``.grid()``.

    Notes
    -----
    O vínculo é feito no contêiner, e não no próprio rótulo, porque
    ``CTkLabel.bind`` repassa o evento ao rótulo *interno* do customtkinter:
    a largura que chegaria no evento seria a do texto já quebrado, e não a do
    espaço disponível — realimentando a quebra a cada evento. A largura é lida
    do widget depois que o gerenciador de geometria termina (``after_idle``),
    quando ela já reflete o novo tamanho.
    """
    def _aplicar():
        try:
            largura = max(minimo, label.winfo_width() - margem)
            if label.cget("wraplength") != largura:
                label.configure(wraplength=largura)
        except tk.TclError:
            pass  # rótulo destruído entre o evento e a aplicação

    def _ao_redimensionar(_evento=None):
        try:
            label.after_idle(_aplicar)
        except tk.TclError:
            pass

    label.master.bind("<Configure>", _ao_redimensionar, add="+")
    label.after_idle(_aplicar)
    return label


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
