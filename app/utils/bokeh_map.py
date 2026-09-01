"""
Gera o Mapa Populacional interativo (Bokeh) como um HTML autocontido.

Complementa o mapa estático (matplotlib) da ``UmapMapWindow``: a versão
interativa abre no navegador e permite **hover** (ver diagnóstico de cada ponto
de treino e, para os pacientes do lote, classe/certeza/perfil/decisão), além de
zoom, pan e legenda clicável para ligar/desligar grupos.

Bokeh renderiza para HTML/JavaScript — não há como embuti-lo numa janela
Tkinter como o matplotlib. Por isso o mapa é salvo em arquivo e aberto no
navegador. Os recursos do BokehJS são embutidos (``INLINE``) para o arquivo
funcionar offline, sem depender de CDN.
"""

import os
import tempfile

import numpy as np
from bokeh.embed import file_html
from bokeh.models import BoxAnnotation, HoverTool, Span
from bokeh.plotting import ColumnDataSource, figure
from bokeh.resources import INLINE

COR_MALIGNO = "#e74c3c"
COR_BENIGNO = "#2ecc71"
COR_REVISAR = "#e67e22"
COR_FUNDO = "#2b2b2b"


def _cor(classe) -> str:
    """
    Cor do ponto conforme a decisão do modelo sobre aquele paciente.

    Com a recusa ligada existe uma terceira saída ("Revisar"), que não pode
    herdar a cor de Benigno: no mapa, um caso não decidido pareceria liberado.
    """
    if classe == "Maligno":
        return COR_MALIGNO
    return COR_REVISAR if classe == "Revisar" else COR_BENIGNO


def _tema_escuro(fig):
    """Aplica o tema escuro do app a uma figura Bokeh."""
    fig.background_fill_color = fig.border_fill_color = COR_FUNDO
    fig.outline_line_color = "gray"
    fig.title.text_color = "white"
    fig.xaxis.axis_label_text_color = fig.yaxis.axis_label_text_color = "gray"
    fig.xaxis.major_label_text_color = fig.yaxis.major_label_text_color = "gray"
    fig.xaxis.axis_line_color = fig.yaxis.axis_line_color = "gray"
    fig.grid.grid_line_color = "#3a3a3a"


def _salvar(fig, titulo, caminho_saida):
    """Serializa a figura como HTML autocontido (BokehJS INLINE) e grava em disco."""
    html = file_html(fig, INLINE, titulo)
    if caminho_saida is None:
        fd, caminho_saida = tempfile.mkstemp(prefix="grafico_", suffix=".html")
        os.close(fd)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(html)
    return caminho_saida


def _campo(paciente: dict, chave: str, padrao: str = "—") -> str:
    """Valor de ``chave`` no paciente, com um traço quando ausente/None."""
    valor = paciente.get(chave, padrao)
    return padrao if valor is None else str(valor)


def gerar_mapa_html(train_2d, train_y, batch_2d, pacientes: list, caminho_saida: str = None) -> str:
    """
    Monta o mapa populacional interativo e o grava como HTML autocontido.

    Parameters
    ----------
    train_2d : array-like (n_treino, 2)
        Projeção 2D dos pacientes de treino (fundo do mapa).
    train_y : array-like (n_treino,)
        Rótulos reais do treino (0 = Benigno, 1 = Maligno).
    batch_2d : array-like (n_lote, 2)
        Projeção 2D dos pacientes do lote.
    pacientes : list[dict]
        Um item por paciente do lote. Usa as chaves ``indice`` e ``classe`` e,
        quando presentes, ``certeza``/``perfil``/``decisao`` (enriquecem o hover).
    caminho_saida : str, optional
        Caminho do HTML de saída. Se None, cria um arquivo temporário.

    Returns
    -------
    str
        Caminho do arquivo HTML gerado.
    """
    train_2d = np.asarray(train_2d, dtype=float)
    train_y = np.asarray(train_y)
    batch_2d = np.asarray(batch_2d, dtype=float)

    fig = figure(
        title="Mapa Populacional Interativo: pacientes do lote sobre a população de treino",
        width=980, height=680,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        background_fill_color=COR_FUNDO, border_fill_color=COR_FUNDO,
        outline_line_color="gray",
    )
    fig.xaxis.axis_label, fig.yaxis.axis_label = "Dim-1", "Dim-2"
    fig.title.text_color = "white"
    fig.xaxis.axis_label_text_color = fig.yaxis.axis_label_text_color = "gray"
    fig.xaxis.major_label_text_color = fig.yaxis.major_label_text_color = "gray"
    fig.xaxis.axis_line_color = fig.yaxis.axis_line_color = "gray"
    fig.grid.grid_line_color = "#3a3a3a"

    # --- treino (população ao fundo), separado por diagnóstico para a legenda ---
    ben, mal = train_y == 0, train_y == 1
    src_ben = ColumnDataSource(dict(x=train_2d[ben, 0], y=train_2d[ben, 1]))
    src_mal = ColumnDataSource(dict(x=train_2d[mal, 0], y=train_2d[mal, 1]))
    r_ben = fig.scatter("x", "y", source=src_ben, size=6, color=COR_BENIGNO,
                        alpha=0.22, legend_label="Treino: Benigno")
    r_mal = fig.scatter("x", "y", source=src_mal, size=6, color=COR_MALIGNO,
                        alpha=0.22, legend_label="Treino: Maligno")

    # --- lote (pacientes novos) como losangos destacados ---
    src_batch = ColumnDataSource(dict(
        x=batch_2d[:, 0], y=batch_2d[:, 1],
        indice=[_campo(p, "indice", str(i)) for i, p in enumerate(pacientes)],
        classe=[_campo(p, "classe") for p in pacientes],
        certeza=[_campo(p, "certeza") for p in pacientes],
        perfil=[_campo(p, "perfil") for p in pacientes],
        decisao=[_campo(p, "decisao") for p in pacientes],
        cor=[_cor(p.get("classe")) for p in pacientes],
    ))
    r_batch = fig.scatter("x", "y", source=src_batch, size=13, marker="diamond",
                          color="cor", line_color="white", line_width=1.0,
                          legend_label="Lote (paciente novo)")

    # --- hovers distintos para treino e para o lote ---
    fig.add_tools(HoverTool(renderers=[r_ben], tooltips=[("Origem", "Treino"), ("Diagnóstico", "Benigno")]))
    fig.add_tools(HoverTool(renderers=[r_mal], tooltips=[("Origem", "Treino"), ("Diagnóstico", "Maligno")]))
    fig.add_tools(HoverTool(renderers=[r_batch], tooltips=[
        ("Paciente", "@indice"),
        ("Diagnóstico", "@classe"),
        ("Certeza", "@certeza"),
        ("Perfil", "@perfil"),
        ("Zona", "@decisao"),
    ]))

    fig.legend.click_policy = "hide"
    fig.legend.background_fill_color = COR_FUNDO
    fig.legend.label_text_color = "white"
    fig.legend.border_line_color = "gray"

    return _salvar(fig, "Mapa Populacional Interativo", caminho_saida)


def gerar_margem_svm_html(explicacoes: list, caminho_saida: str = None) -> str:
    """
    Gera o gráfico interativo da MARGEM do SVM como HTML autocontido.

    Torna a margem "visível": o eixo X é o escore de decisão z (decision_function
    do SVM — a distância, com sinal, à fronteira). A faixa cinza entre z = −1 e
    z = +1 é a margem (a "rua" que o SVM maximiza); suas bordas passam pelos
    vetores de suporte. Pacientes dentro da faixa (|z| < 1) estão na zona de
    baixa confiança; à esquerda de z = 0, Benigno; à direita, Maligno. Hover
    mostra paciente, diagnóstico, confiança e o próprio z.

    Parameters
    ----------
    explicacoes : list[dict]
        Explicações por paciente do SVMExplainer (usa 'indice', 'classe',
        'confianca', 'distancia').
    caminho_saida : str, optional
        Caminho do HTML de saída. Se None, cria um arquivo temporário.

    Returns
    -------
    str
        Caminho do arquivo HTML gerado.
    """
    n = len(explicacoes)
    ys = np.random.default_rng(42).uniform(0.05, 0.95, n)  # espalhamento vertical
    xs = [float(e['distancia']) for e in explicacoes]

    fig = figure(
        title="Margem do SVM: distância de cada paciente à fronteira de decisão",
        width=980, height=520,
        tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom",
    )
    _tema_escuro(fig)
    fig.xaxis.axis_label = "z = escore de decisão (distância à fronteira)"
    fig.yaxis.visible = False
    fig.y_range.start, fig.y_range.end = 0, 1

    # A margem: faixa entre z = −1 e z = +1 (a "rua" maximizada pelo SVM).
    fig.add_layout(BoxAnnotation(left=-1, right=1, fill_color="#7f8c8d", fill_alpha=0.18))
    fig.add_layout(BoxAnnotation(right=-1, fill_color=COR_BENIGNO, fill_alpha=0.06))
    fig.add_layout(BoxAnnotation(left=1, fill_color=COR_MALIGNO, fill_alpha=0.06))
    fig.add_layout(Span(location=0, dimension="height", line_color="white",
                        line_dash="dashed", line_width=2))          # fronteira
    for m in (-1, 1):  # bordas da margem (onde ficam os vetores de suporte)
        fig.add_layout(Span(location=m, dimension="height", line_color="gray",
                            line_dash="dotted", line_width=1))

    src = ColumnDataSource(dict(
        x=xs, y=list(ys),
        indice=[str(e.get('indice', i)) for i, e in enumerate(explicacoes)],
        classe=[str(e.get('classe', '—')) for e in explicacoes],
        confianca=[f"{e.get('confianca', 0):.0f}%" for e in explicacoes],
        z=[f"{x:+.2f}" for x in xs],
        cor=[_cor(e.get('classe')) for e in explicacoes],
    ))
    r = fig.scatter("x", "y", source=src, size=12, color="cor",
                    line_color="white", line_width=0.8)
    fig.add_tools(HoverTool(renderers=[r], tooltips=[
        ("Paciente", "@indice"),
        ("Diagnóstico", "@classe"),
        ("Confiança", "@confianca"),
        ("z (dist. à margem)", "@z"),
    ]))
    return _salvar(fig, "Margem do SVM", caminho_saida)
