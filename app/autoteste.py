"""
Autoteste do app: ``DiagnosisXAI --autoteste``.

Roda, sem abrir nenhuma janela, o caminho completo que um usuário percorre:
carrega o ``wisconsin.pkl``, processa o CSV de exemplo, diagnostica com todos
os modelos e com o comitê, gera as explicações, os fatores, o SHAP, o mapa
Bokeh, os dois PDFs e grava o histórico.

Existe para validar o **executável**, não o código. O PyInstaller empacota o
que consegue ver nos imports, e o que fica de fora só aparece quando o
programa roda. Exemplos: um tema do customtkinter, os arquivos do Bokeh, o
numba que o shap exige ou o ``core.explainers``, que o ``.pkl`` referencia
por nome. O build roda este autoteste no programa pronto, inclusive no
Windows do GitHub Actions, onde ninguém pode clicar em nada.

Nunca toca nos dados reais: grava numa pasta temporária, a menos que
``DIAGNOSISXAI_DADOS`` já aponte para outra. O relatório vai para o arquivo
indicado em ``DIAGNOSISXAI_AUTOTESTE_LOG``, ou para ``autoteste.txt`` nessa
pasta. O código de saída é 0 se tudo passou e 1 caso contrário.
"""

import os
import sys
import tempfile
import time
import traceback
import warnings

VARIAVEL_LOG = 'DIAGNOSISXAI_AUTOTESTE_LOG'


def executar() -> int:
    """
    Roda todas as verificações e devolve o código de saída do processo.

    Returns
    -------
    int
        0 se todas passaram; 1 se alguma falhou.
    """
    from core import caminhos

    if not os.environ.get(caminhos.VARIAVEL_DADOS):
        os.environ[caminhos.VARIAVEL_DADOS] = tempfile.mkdtemp(prefix='diagnosisxai-autoteste-')
    warnings.filterwarnings('ignore')

    resultados, contexto = [], {}
    for nome, etapa in _ETAPAS:
        inicio = time.time()
        try:
            detalhe = etapa(contexto) or ''
            resultados.append((True, nome, time.time() - inicio, detalhe))
        except Exception:
            resultados.append((False, nome, time.time() - inicio, traceback.format_exc()))

    linhas = [f"DiagnosisXAI — autoteste ({'executável' if caminhos.empacotado() else 'código-fonte'})",
              f"Python {sys.version.split()[0]} · {sys.platform}", ""]
    for ok, nome, duracao, detalhe in resultados:
        linhas.append(f"[{'OK ' if ok else 'ERRO'}] {nome}  ({duracao:.1f}s)  {detalhe if ok else ''}".rstrip())
        if not ok:
            linhas.append("       " + detalhe.rstrip().replace("\n", "\n       "))
    falhas = sum(not ok for ok, *_ in resultados)
    linhas += ["", "RESULTADO: " + ("tudo certo" if not falhas else f"{falhas} etapa(s) com erro")]
    relatorio = "\n".join(linhas) + "\n"

    destino = os.environ.get(VARIAVEL_LOG) or os.path.join(
        os.environ[caminhos.VARIAVEL_DADOS], 'autoteste.txt')
    with open(destino, 'w', encoding='utf-8') as arquivo:
        arquivo.write(relatorio)
    # No executável sem console (Windows), sys.stdout é None: o arquivo é o registro.
    if sys.stdout is not None:
        print(relatorio + f"Relatório gravado em: {destino}")
    return 1 if falhas else 0


# --- etapas: cada uma recebe e enriquece o mesmo contexto -----------------------

def _recursos(ctx):
    from core import caminhos
    faltando = [n for n in ('wisconsin.pkl', 'limiares.json', 'umap_train_2d.npy')
                if not os.path.exists(caminhos.recurso(n))]
    if faltando:
        raise FileNotFoundError(f"recursos ausentes em {caminhos.pasta_recursos()}: {faltando}")
    return str(caminhos.pasta_recursos())


def _carregar_modelos(ctx):
    from core.inference import ModelLoader
    loader = ModelLoader()
    explicadores = [k for k, v in loader.explainers.items() if v is not None]
    if len(loader.models) != 5 or len(loader.calibrated_models) != 4 or len(explicadores) != 5:
        raise RuntimeError(f"pkl incompleto: {len(loader.models)} modelos, "
                           f"{len(loader.calibrated_models)} calibrados, explicadores {explicadores}")
    if loader.umap_train_2d is None or loader.X_train_scaled is None:
        raise RuntimeError("faltam o treino padronizado ou o embedding UMAP")
    ctx['loader'] = loader
    return f"{len(loader.models)} modelos, {len(explicadores)} explicadores"


def _politica(ctx):
    from core.decision import PoliticaDecisao
    politica = PoliticaDecisao.carregar()
    if not politica.calibrada:
        raise RuntimeError("limiares.json não foi lido: o app cairia no corte de 50%")
    return f"comitê corta em {politica.limiar('Comitê (voto suave)') * 100:.1f}%"


def _processar_csv(ctx):
    import pandas as pd
    from core import caminhos
    from core.batch_processor import BatchProcessor

    pasta = caminhos.pasta_exemplos()
    if not os.path.exists(os.path.join(pasta, 'dataTeste_sem_diagnostico.csv')):
        pasta = str(caminhos.raiz_projeto() / 'data')     # código-fonte
    ctx['pasta_exemplos'] = pasta
    bruto = pd.read_csv(os.path.join(pasta, 'dataTeste_sem_diagnostico.csv'))
    processador = BatchProcessor()
    ctx['df_pad'], ctx['df_limpo'] = processador.process(bruto)
    return f"{len(bruto)} pacientes de {pasta}"


def _validar_entrada(ctx):
    import pandas as pd
    from core.validacao import LoteInvalido, validar_colunas
    try:
        validar_colunas(pd.DataFrame({'radius_mean': [1.0]}), ctx['loader'].feature_names)
    except LoteInvalido:
        return "planilha incompleta recusada"
    raise RuntimeError("a validação deixou passar uma planilha sem 29 colunas")


def _diagnosticar(ctx):
    from core.predictor import PredictorEngine
    motor = PredictorEngine(ctx['loader'])
    ctx['motor'] = motor
    for modelo in motor.modelos_disponiveis():
        resultado = motor.predict(ctx['df_pad'], ctx['df_limpo'], modelo)
        if len(resultado) != len(ctx['df_pad']):
            raise RuntimeError(f"{modelo}: {len(resultado)} linhas")
    return ", ".join(motor.modelos_disponiveis())


def _auditar(ctx):
    import pandas as pd
    from core.metrics import avaliar_modelos
    gabarito = pd.read_csv(os.path.join(ctx['pasta_exemplos'], 'dataTeste_com_diagnostico.csv'))
    resultado = ctx['motor'].predict(ctx['df_pad'], ctx['df_limpo'], 'Comitê (voto suave)')
    resultado['Diagnóstico_Real'] = ['Maligno' if v == 1 else 'Benigno' for v in gabarito['diagnosis']]
    metricas = next(iter(avaliar_modelos(resultado, nome_modelo='Comitê (voto suave)').values()))
    return f"comitê: cobertura {metricas['cobertura']:.1f}%, {metricas['fn']} FN, {metricas['fp']} FP"


def _explicar(ctx):
    exp, pad, limpo = ctx['loader'].explainers, ctx['df_pad'], ctx['df_limpo']
    exp['arvore'].explain(limpo)
    exp['logistica'].explain(pad, limpo)
    for chave in ('knn', 'randomforest', 'svm'):
        exp[chave].explain(pad)
    return "árvore, logística, KNN, random forest, SVM"


def _fatores(ctx):
    from core.decision import NOME_COMITE
    from core.fatores import FatoresDoLote
    loader, fn = ctx['loader'], ctx['loader'].feature_names
    fatores = FatoresDoLote(loader, ctx['df_pad'][fn].values, ctx['df_limpo'][fn].values,
                            list(ctx['df_pad'].index))
    r = fatores.explicar(NOME_COMITE, ctx['df_pad'].index[0])
    certeza = ctx['motor'].probabilidades_calibradas(
        ctx['df_pad'].iloc[:1], ctx['df_limpo'].iloc[:1], [NOME_COMITE])[NOME_COMITE][0]
    if abs(r['prob'] - certeza) > 1e-9:
        raise RuntimeError(f"bloco fecha em {r['prob']:.4f}, tabela mostra {certeza:.4f}")
    return f"comitê, paciente 0: {r['prob'] * 100:.1f}% (igual à tabela)"


def _shap(ctx):
    from core.shap_explainer import ShapExplainer   # importa shap -> numba -> llvmlite
    loader, fn = ctx['loader'], ctx['loader'].feature_names
    for chave, nome in (('rf', 'Random Forest'), ('lr', 'Regressão Logística')):
        ShapExplainer(loader.models[nome], chave, fn, loader.shap_background).explain_one(
            ctx['df_pad'][fn].values[0])
    return "TreeExplainer e KernelExplainer"


def _bokeh(ctx):
    import numpy as np
    from utils.bokeh_map import gerar_margem_svm_html
    exp = ctx['loader'].explainers['svm'].explain(ctx['df_pad'].iloc[:5])
    caminho = gerar_margem_svm_html(exp, os.path.join(tempfile.mkdtemp(), 'margem.html'))
    tamanho = os.path.getsize(caminho)
    if tamanho < 100_000:     # BokehJS embutido no HTML pesa centenas de KB
        raise RuntimeError(f"HTML de {tamanho} bytes: o BokehJS não foi embutido")
    return f"HTML autocontido de {tamanho // 1024} KB"


def _pdfs(ctx):
    from matplotlib.figure import Figure
    from core import caminhos
    from utils.pdf_report import export_batch_report, export_patient_report

    pasta = caminhos.pasta_relatorios()
    resultado = ctx['motor'].predict(ctx['df_pad'], ctx['df_limpo'], 'SVM')
    meta = {'arquivo': 'exemplo.csv', 'modelo': 'SVM', 'total': len(resultado),
            'malignos': 0, 'benignos': 0}
    export_batch_report(os.path.join(pasta, 'lote.pdf'), meta, resultado, {}, auditoria=None)
    figura = Figure(figsize=(4, 3))
    figura.add_subplot(111).plot([0, 1], [1, 0])
    export_patient_report(os.path.join(pasta, 'paciente.pdf'), 'Autoteste', 0,
                          "PACIENTE 0\n  Régua: Maligno: certeza ≥ 68.2%  →", figura=figura)
    return pasta


def _historico(ctx):
    from core.history_manager import HistoryManager
    gerenciador = HistoryManager()
    gerenciador.save_session('exemplo.csv', 'SVM', 143, 49, 57, {'politica': {}}, 37)
    if gerenciador.load()[0]['total'] != 143:
        raise RuntimeError("a sessão gravada não foi lida de volta")
    return gerenciador._path


def _interface(ctx):
    # Só importa: abrir janela exigiria tela, que o build não tem.
    import customtkinter
    tema = os.path.join(os.path.dirname(customtkinter.__file__), 'assets', 'themes', 'green.json')
    if not os.path.exists(tema):
        raise FileNotFoundError(f"tema do customtkinter ausente: {tema}")
    import views.main_window  # noqa: F401  (puxa todas as telas e janelas de relatório)
    return "customtkinter e todas as telas importam"


_ETAPAS = [
    ("recursos embutidos", _recursos),
    ("carregar wisconsin.pkl", _carregar_modelos),
    ("régua (limiares.json)", _politica),
    ("processar CSV de exemplo", _processar_csv),
    ("validação de entrada", _validar_entrada),
    ("diagnóstico (todos os modelos)", _diagnosticar),
    ("auditoria contra o gabarito", _auditar),
    ("explicadores por modelo", _explicar),
    ("fatores que mais pesaram", _fatores),
    ("SHAP", _shap),
    ("mapa Bokeh", _bokeh),
    ("PDFs", _pdfs),
    ("histórico", _historico),
    ("interface", _interface),
]
