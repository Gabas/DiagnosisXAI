"""
Testes de core/caminhos.py: onde o app lê e grava, pelo código-fonte e no executável.

O modo executável é simulado como o PyInstaller o apresenta ao programa:
``sys.frozen = True``, ``sys._MEIPASS`` apontando para a pasta interna do
pacote e ``sys.executable`` para o programa. Assim dá para testar o
executável sem gerá-lo.
"""

import sys
from pathlib import Path

import pytest

from core import caminhos


@pytest.fixture
def sem_sobreposicao(monkeypatch):
    monkeypatch.delenv(caminhos.VARIAVEL_DADOS, raising=False)


@pytest.fixture
def executavel(tmp_path, monkeypatch, sem_sobreposicao):
    """Simula o app empacotado em ``tmp/DiagnosisXAI/`` com a pasta pessoal em ``tmp/casa``."""
    pacote = tmp_path / 'DiagnosisXAI'
    (pacote / '_internal' / 'data').mkdir(parents=True)
    casa = tmp_path / 'casa'
    casa.mkdir()
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(pacote / '_internal'), raising=False)
    monkeypatch.setattr(sys, 'executable', str(pacote / 'DiagnosisXAI'))
    monkeypatch.setattr(Path, 'home', lambda: casa)
    return pacote, casa


# --- pelo código-fonte: nada muda em relação ao que o app sempre fez ---------

def test_codigo_fonte_le_os_recursos_do_data_do_repositorio(sem_sobreposicao):
    assert not caminhos.empacotado()
    pkl = Path(caminhos.recurso('wisconsin.pkl'))
    assert pkl.parent == caminhos.raiz_projeto() / 'data'
    assert pkl.exists()


def test_codigo_fonte_grava_onde_sempre_gravou(sem_sobreposicao):
    raiz = caminhos.raiz_projeto()
    assert Path(caminhos.arquivo_historico()) == raiz / 'data' / 'history.json'
    assert Path(caminhos.pasta_relatorios()) == raiz / 'reports'
    assert Path(caminhos.pasta_exemplos()) == raiz / 'data'


# --- no executável ------------------------------------------------------------

def test_executavel_le_recursos_de_dentro_do_pacote(executavel):
    pacote, _ = executavel
    assert caminhos.empacotado()
    assert Path(caminhos.recurso('limiares.json')) == pacote / '_internal' / 'data' / 'limiares.json'


def test_executavel_grava_na_pasta_pessoal_e_nao_no_pacote(executavel):
    pacote, casa = executavel
    historico = Path(caminhos.arquivo_historico())
    relatorios = Path(caminhos.pasta_relatorios())
    assert historico == casa / 'DiagnosisXAI' / 'history.json'
    assert relatorios == casa / 'DiagnosisXAI' / 'relatorios'
    assert relatorios.is_dir() and historico.parent.is_dir()   # criadas sob demanda
    assert pacote not in historico.parents


def test_executavel_abre_o_dialogo_nos_exemplos_ao_lado_do_programa(executavel):
    pacote, casa = executavel
    assert Path(caminhos.pasta_exemplos()) == casa          # sem a pasta: recua para a pessoal
    (pacote / 'exemplos').mkdir()
    assert Path(caminhos.pasta_exemplos()) == pacote / 'exemplos'


# --- sobreposição (testes e autoteste do executável) -----------------------------

def test_variavel_de_ambiente_desvia_toda_a_gravacao(tmp_path, monkeypatch):
    monkeypatch.setenv(caminhos.VARIAVEL_DADOS, str(tmp_path / 'isolado'))
    assert Path(caminhos.arquivo_historico()) == tmp_path / 'isolado' / 'history.json'
    assert Path(caminhos.pasta_relatorios()) == tmp_path / 'isolado' / 'relatorios'


def test_historico_e_pdfs_do_app_seguem_o_modulo(tmp_path, monkeypatch):
    from core.history_manager import HistoryManager
    from utils.pdf_report import resolve_reports_dir

    monkeypatch.setenv(caminhos.VARIAVEL_DADOS, str(tmp_path))
    gerenciador = HistoryManager()
    gerenciador.save_session('lote.csv', 'SVM', 3, 1, 2)
    assert (tmp_path / 'history.json').exists()
    assert len(gerenciador.load()) == 1
    assert Path(resolve_reports_dir()) == tmp_path / 'relatorios'
