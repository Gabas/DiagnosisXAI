"""
Testes do gerador de PDF (utils/pdf_report.py).

São smoke tests: confirmam que os dois relatórios (lote e por paciente) são
gerados como PDF válido e não vazio, inclusive com texto contendo os glifos
sem suporte nas fontes base ('⚠', setas) que a sanitização precisa tratar
sem lançar exceção.
"""

import pandas as pd

from utils.pdf_report import export_batch_report, export_patient_report


def test_export_batch_report_gera_pdf_valido(tmp_path):
    df = pd.DataFrame({
        'Diagnóstico_IA': ['Maligno', 'Benigno', 'Benigno'],
        'Certeza_Maligno(%)': [91.2, 3.4, 12.0],
    })
    meta = {'arquivo': 'teste.csv', 'modelo': 'SVM', 'total': 3, 'malignos': 1, 'benignos': 2}
    importancias = {'SVM': [('radius_worst', 0.12), ('texture_worst', 0.08)]}
    destino = tmp_path / 'lote.pdf'

    export_batch_report(str(destino), meta, df, importancias, auditoria=None)

    assert destino.exists()
    conteudo = destino.read_bytes()
    assert conteudo.startswith(b'%PDF')
    assert len(conteudo) > 1000


def test_export_batch_report_com_auditoria(tmp_path):
    df = pd.DataFrame({'Diagnóstico_IA': ['Maligno'], 'Diagnóstico_Real': ['Maligno']})
    meta = {'arquivo': 'teste.csv', 'modelo': 'SVM', 'total': 1, 'malignos': 1, 'benignos': 0}
    destino = tmp_path / 'lote_auditado.pdf'

    export_batch_report(str(destino), meta, df, {}, auditoria={'SVM': 97.2})

    assert destino.read_bytes().startswith(b'%PDF')


def test_export_patient_report_gera_pdf_valido(tmp_path):
    destino = tmp_path / 'paciente.pdf'
    texto = (
        "PACIENTE 0\n"
        "Diagnóstico da IA: Maligno\n"
        "⚠ Caso limítrofe — vizinho mais próximo pesou mais ↑\n"
    )

    export_patient_report(str(destino), "Relatório de Teste", 0, texto, figura=None)

    assert destino.exists()
    assert destino.read_bytes().startswith(b'%PDF')


def test_export_patient_report_com_grafico_matplotlib(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2], [0, 1, 0])

    destino = tmp_path / 'paciente_grafico.pdf'
    export_patient_report(str(destino), "Relatório de Teste", 1, "PACIENTE 1\nBenigno", figura=fig)

    assert destino.read_bytes().startswith(b'%PDF')
