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
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from bokeh.resources import INLINE

COR_MALIGNO = "#e74c3c"
COR_BENIGNO = "#2ecc71"
COR_FUNDO = "#2b2b2b"


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
        title="Mapa Populacional Interativo — pacientes do lote sobre a população de treino",
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
                        alpha=0.22, legend_label="Treino — Benigno")
    r_mal = fig.scatter("x", "y", source=src_mal, size=6, color=COR_MALIGNO,
                        alpha=0.22, legend_label="Treino — Maligno")

    # --- lote (pacientes novos) como losangos destacados ---
    src_batch = ColumnDataSource(dict(
        x=batch_2d[:, 0], y=batch_2d[:, 1],
        indice=[_campo(p, "indice", str(i)) for i, p in enumerate(pacientes)],
        classe=[_campo(p, "classe") for p in pacientes],
        certeza=[_campo(p, "certeza") for p in pacientes],
        perfil=[_campo(p, "perfil") for p in pacientes],
        decisao=[_campo(p, "decisao") for p in pacientes],
        cor=[COR_MALIGNO if p.get("classe") == "Maligno" else COR_BENIGNO for p in pacientes],
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
        ("Decisão", "@decisao"),
    ]))

    fig.legend.click_policy = "hide"
    fig.legend.background_fill_color = COR_FUNDO
    fig.legend.label_text_color = "white"
    fig.legend.border_line_color = "gray"

    html = file_html(fig, INLINE, "Mapa Populacional Interativo")
    if caminho_saida is None:
        fd, caminho_saida = tempfile.mkstemp(prefix="mapa_interativo_", suffix=".html")
        os.close(fd)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(html)
    return caminho_saida
