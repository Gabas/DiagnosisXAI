"""
Geração de relatórios em PDF a partir das sessões de diagnóstico.

Dois documentos são suportados:

- ``export_batch_report``  : resumo de uma sessão de lote — metadados,
  acurácia (se auditado), importância global dos biomarcadores e a tabela
  de diagnóstico por paciente.
- ``export_patient_report`` : o relatório de explicabilidade de um único
  paciente, tal como exibido numa janela de relatório (texto + gráfico,
  quando houver) — usado pelo botão "Exportar PDF" de cada janela.

As fontes base do PDF (Helvetica/Courier) não têm glifos para setas ou
símbolos como "⚠" — ``_sanitizar`` os substitui por equivalentes em ASCII
antes de qualquer texto ser desenhado.
"""

import io
import re
from datetime import datetime

from core.caminhos import pasta_relatorios

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
)

_ESTILOS = getSampleStyleSheet()
_TITULO = ParagraphStyle('DXAI_Titulo', parent=_ESTILOS['Title'], fontSize=18, spaceAfter=2)
_SUBTITULO = ParagraphStyle('DXAI_Subtitulo', parent=_ESTILOS['Heading3'], spaceAfter=2)
_TIMESTAMP = ParagraphStyle('DXAI_Timestamp', parent=_ESTILOS['Normal'], textColor=colors.grey, fontSize=9)
_H2 = ParagraphStyle('DXAI_H2', parent=_ESTILOS['Heading2'], spaceBefore=14, spaceAfter=6)
_MONO = ParagraphStyle('DXAI_Mono', parent=_ESTILOS['Code'], fontName='Courier', fontSize=9, leading=12)
_CORPO = ParagraphStyle('DXAI_Corpo', parent=_ESTILOS['Normal'], fontSize=9, leading=12,
                        textColor=colors.grey, spaceBefore=6)

_MARGENS = dict(leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)

# Glifos usados nas janelas de relatório que os fontes base do PDF não têm.
_SUBSTITUICOES = {
    '⚠': '[!]', '←': '<-', '→': '->', '↑': '^', '↓': 'v',
    # A régua de decisão (PoliticaDecisao.regra) usa os dois — e aparece tanto
    # no PDF do lote quanto no bloco de fatores do PDF do paciente.
    '≥': '>=', '≤': '<=',
}


def resolve_reports_dir() -> str:
    """Pasta padrão dos PDFs: ``reports/`` pelo código-fonte, ``~/DiagnosisXAI/relatorios`` no executável."""
    return pasta_relatorios()


def _sanitizar(texto: str) -> str:
    """Substitui glifos sem suporte nas fontes base do PDF por equivalentes ASCII."""
    for original, ascii_eq in _SUBSTITUICOES.items():
        texto = texto.replace(original, ascii_eq)
    return texto


def _escapar_html(texto: str) -> str:
    """Escapa marcação especial do ReportLab (subconjunto de HTML) em texto livre."""
    return (texto.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _preservar_colunas(texto: str) -> str:
    """
    Mantém o recuo e os espaços repetidos de uma linha em fonte monoespaçada.

    O ``Paragraph`` do ReportLab colapsa espaços como o HTML, o que desfazia
    as colunas alinhadas do texto de detalhe (a cascata de fatores, o balanço
    de forças do SVM, a lista de membros do comitê). Recuo e sequências de dois
    ou mais espaços viram espaços não separáveis; o espaço simples entre
    palavras continua normal, para a linha longa ainda poder quebrar.
    """
    recuo = len(texto) - len(texto.lstrip(' '))
    corpo = re.sub(r' {2,}', lambda m: '&nbsp;' * len(m.group()), texto[recuo:])
    return '&nbsp;' * recuo + corpo


def _estilo_tabela(zebra: bool = False) -> TableStyle:
    """Estilo padrão (cabeçalho azul, grade cinza) para as tabelas do relatório."""
    regras = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f538d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    if zebra:
        regras.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]))
    return TableStyle(regras)


def _cabecalho(titulo_doc: str) -> list:
    """Bloco inicial comum aos dois relatórios: título, subtítulo e data de geração."""
    agora = datetime.now().strftime('%d/%m/%Y %H:%M')
    return [
        Paragraph("DiagnosisXAI", _TITULO),
        Paragraph(_escapar_html(titulo_doc), _SUBTITULO),
        Paragraph(f"Gerado em {agora}", _TIMESTAMP),
        Spacer(1, 0.5 * cm),
    ]


def export_batch_report(path: str, meta: dict, df, importancias_por_modelo: dict,
                         auditoria: dict = None):
    """
    Gera o PDF-resumo de uma sessão de diagnóstico em lote.

    Parameters
    ----------
    path : str
        Caminho de destino do arquivo PDF.
    meta : dict
        {'arquivo', 'modelo', 'total', 'malignos', 'benignos'} e, opcionalmente,
        'adiados', 'regua' (saída de ``PoliticaDecisao.regua``) e
        'justificativa' (o motivo dos cortes). As duas últimas fazem o PDF
        carregar o ponto de operação junto com os resultados: sem elas, uma
        linha "Maligno, certeza 25%" fica sem como ser conferida.
    df : pandas.DataFrame
        ``df_resultado`` do pipeline — usa-se apenas o índice e as colunas de
        diagnóstico (as 30 colunas de biomarcadores não entram na tabela).
    importancias_por_modelo : dict
        {nome_do_modelo: [(biomarcador, valor_formatado), ...]} — uma seção
        por modelo com relatório de importância disponível. ``valor_formatado``
        já vem como texto (ex.: "3.01%" para importância por permutação/Gini,
        "+1.40 (Maligno)" para coeficiente de regressão) — cada tipo de
        explicador tem sua própria unidade e não deve ser normalizado aqui.
    auditoria : dict ou None
        {nome_do_modelo: acurácia_percentual}, quando a etapa de auditoria
        (Passo 4) foi executada.
    """
    doc = SimpleDocTemplate(path, pagesize=A4, **_MARGENS)
    story = _cabecalho("Relatório de Diagnóstico em Lote")

    resumo = [
        ["Arquivo", str(meta.get('arquivo', '—'))],
        ["Modelo", str(meta.get('modelo', '—'))],
        ["Total de pacientes", str(meta.get('total', len(df)))],
        ["Maligno", str(meta.get('malignos', '—'))],
        ["Benigno", str(meta.get('benignos', '—'))],
    ]
    if meta.get('adiados'):
        resumo.append(["Devolvidos para revisão", str(meta['adiados'])])
    t = Table(resumo, colWidths=[5 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    if meta.get('regua'):
        story.append(Paragraph("Régua de decisão (ponto de operação)", _H2))
        linhas = [["Resultado", "Faixa de certeza", "O que o sistema faz"]] + [
            [str(f['rotulo']), str(f['faixa']), str(f['efeito'])] for f in meta['regua']
        ]
        tabela = Table(linhas, colWidths=[3 * cm, 4 * cm, 9 * cm])
        tabela.setStyle(_estilo_tabela(zebra=True))
        story.append(tabela)
        if meta.get('justificativa'):
            story.append(Paragraph(_escapar_html(str(meta['justificativa'])), _CORPO))

    if auditoria:
        story.append(Paragraph("Acurácia (auditoria)", _H2))
        linhas = [["Modelo", "Acurácia"]] + [[k, f"{v:.2f}%"] for k, v in auditoria.items()]
        tabela = Table(linhas, colWidths=[11 * cm, 5 * cm])
        tabela.setStyle(_estilo_tabela())
        story.append(tabela)

    for nome_modelo, importancias in importancias_por_modelo.items():
        if not importancias:
            continue
        story.append(Paragraph(f"Biomarcadores mais relevantes: {nome_modelo}", _H2))
        linhas = [["Biomarcador", "Importância"]] + [
            [str(nome), str(valor)] for nome, valor in importancias[:10]
        ]
        tabela = Table(linhas, colWidths=[11 * cm, 5 * cm])
        tabela.setStyle(_estilo_tabela())
        story.append(tabela)

    story.append(Paragraph("Diagnóstico por paciente", _H2))
    colunas = [c for c in df.columns if c == 'Diagnóstico_IA' or c == 'Diagnóstico_Real'
               or c == 'Certeza_Maligno(%)' or str(c).startswith('IA_')]
    cabecalho = ["Paciente"] + colunas
    linhas = [cabecalho] + [
        [str(idx)] + [str(row[c]) for c in colunas] for idx, row in df.iterrows()
    ]
    tabela_pacientes = Table(linhas, repeatRows=1)
    tabela_pacientes.setStyle(_estilo_tabela(zebra=True))
    story.append(tabela_pacientes)

    doc.build(story)


def export_patient_report(path: str, titulo_janela: str, paciente_id, texto_detalhe: str,
                           figura=None):
    """
    Exporta o relatório de explicabilidade de um único paciente.

    Parameters
    ----------
    path : str
        Caminho de destino do arquivo PDF.
    titulo_janela : str
        Título da janela de origem (ex.: "Relatório de Explicabilidade — SVM").
    paciente_id : str ou int
        Identificador do paciente selecionado.
    texto_detalhe : str
        Conteúdo do painel de detalhe (mesmo texto exibido na janela).
    figura : matplotlib.figure.Figure, sequência de figuras, ou None
        Gráfico(s) da janela (mapa populacional, vizinhos, margem, etc.).
        Uma sequência é embutida na ordem dada — o relatório do SVM usa isso
        para juntar o mapa e o painel da margem no mesmo PDF.
    """
    doc = SimpleDocTemplate(path, pagesize=A4, **_MARGENS)
    story = _cabecalho(titulo_janela)
    story.append(Paragraph(f"Paciente selecionado: {_escapar_html(str(paciente_id))}", _ESTILOS['Normal']))
    story.append(Spacer(1, 0.4 * cm))

    if figura is None:
        figuras = []
    elif isinstance(figura, (list, tuple)):
        figuras = [f for f in figura if f is not None]
    else:
        figuras = [figura]

    for fig in figuras:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, facecolor=fig.get_facecolor())
        buffer.seek(0)
        largura = 15 * cm
        altura = largura * fig.get_figheight() / fig.get_figwidth()
        story.append(Image(buffer, width=largura, height=altura))
        story.append(Spacer(1, 0.4 * cm))

    for linha in _sanitizar(texto_detalhe).strip('\n').split('\n'):
        story.append(Paragraph(_preservar_colunas(_escapar_html(linha)) or '&nbsp;', _MONO))

    doc.build(story)
