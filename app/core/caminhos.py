"""
Onde o app lê e grava arquivos, rodando pelo código-fonte ou como executável.

Existem dois tipos de arquivo, e separá-los é o motivo deste módulo existir.

**Recursos, somente leitura:** ``wisconsin.pkl``, ``limiares.json`` e
``umap_train_2d.npy``. Pelo código-fonte ficam em ``data/`` do repositório. No
executável (PyInstaller) vêm embutidos no pacote, em ``sys._MEIPASS/data``.

**Dados do usuário, gravação:** o ``history.json`` e os PDFs. Pelo código-fonte
continuam onde sempre estiveram (``data/`` e ``reports/``). No executável vão
para uma pasta permanente na pasta pessoal, ``~/DiagnosisXAI``. A pasta do
pacote não serve: pode ser somente leitura (ex.: ``C:\\Program Files``), e é
apagada ao fechar se o app for empacotado como arquivo único.

Antes deste módulo, cinco arquivos achavam a raiz subindo três níveis a partir
do próprio ``__file__``. Dentro do executável esse cálculo aponta para fora do
pacote, e o app nem abria.

A variável de ambiente ``DIAGNOSISXAI_DADOS`` substitui a pasta de dados do
usuário. Os testes e o autoteste do executável a usam para não tocar no
histórico real.
"""

import os
import sys
from pathlib import Path

NOME_APP = 'DiagnosisXAI'
VARIAVEL_DADOS = 'DIAGNOSISXAI_DADOS'


def empacotado() -> bool:
    """True quando o app roda como executável gerado pelo PyInstaller."""
    return bool(getattr(sys, 'frozen', False))


def raiz_projeto() -> Path:
    """Raiz do repositório (só faz sentido rodando pelo código-fonte)."""
    return Path(__file__).resolve().parents[2]


def pasta_recursos() -> Path:
    """Pasta dos recursos somente leitura (o ``data/`` do pacote ou do repositório)."""
    if empacotado():
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'data'
    return raiz_projeto() / 'data'


def recurso(nome: str) -> str:
    """Caminho de um recurso somente leitura, ex.: ``recurso('wisconsin.pkl')``."""
    return str(pasta_recursos() / nome)


def _pasta_gravacao_externa():
    """
    Pasta de gravação fora do repositório, ou None pelo código-fonte.

    Returns
    -------
    pathlib.Path ou None
        A pasta da variável ``DIAGNOSISXAI_DADOS``, se definida; a pasta
        pessoal do usuário, no executável; None pelo código-fonte (que segue
        gravando no próprio repositório).
    """
    sobreposta = os.environ.get(VARIAVEL_DADOS)
    if sobreposta:
        return Path(sobreposta)
    if empacotado():
        return Path.home() / NOME_APP
    return None


def arquivo_historico() -> str:
    """Caminho do ``history.json``. A pasta é criada se ainda não existir."""
    externa = _pasta_gravacao_externa()
    pasta = externa if externa is not None else raiz_projeto() / 'data'
    pasta.mkdir(parents=True, exist_ok=True)
    return str(pasta / 'history.json')


def pasta_relatorios() -> str:
    """Pasta padrão dos PDFs exportados. É criada se ainda não existir."""
    externa = _pasta_gravacao_externa()
    pasta = externa / 'relatorios' if externa is not None else raiz_projeto() / 'reports'
    pasta.mkdir(parents=True, exist_ok=True)
    return str(pasta)


def pasta_exemplos() -> str:
    """
    Onde o diálogo de abrir planilha começa: nos CSVs de exemplo do app.

    No executável eles vão numa pasta ``exemplos/`` ao lado do programa, à
    vista de quem abre o .zip. Pelo código-fonte estão em ``data/``.
    """
    if empacotado():
        junto = Path(sys.executable).resolve().parent / 'exemplos'
        return str(junto if junto.is_dir() else Path.home())
    dados = raiz_projeto() / 'data'
    return str(dados if dados.is_dir() else raiz_projeto())
