"""
Testes do HistoryManager (core/history_manager.py).

Usa sempre um arquivo isolado em ``tmp_path`` — nunca o ``data/history.json``
real, que guarda o histórico de uso de verdade. ``HistoryManager`` não recebe
o caminho pelo construtor, então o teste aponta ``_path`` manualmente logo
após instanciar.
"""

from core.history_manager import HistoryManager


def _isolado(tmp_path):
    hm = HistoryManager()
    hm._path = str(tmp_path / 'history.json')
    return hm


def test_load_sem_arquivo_retorna_lista_vazia(tmp_path):
    hm = _isolado(tmp_path)
    assert hm.load() == []


def test_save_session_insere_no_topo_mais_recente_primeiro(tmp_path):
    hm = _isolado(tmp_path)
    hm.save_session('a.csv', 'SVM', 10, 4, 6)
    hm.save_session('b.csv', 'KNN', 20, 8, 12)

    entradas = hm.load()
    assert len(entradas) == 2
    assert entradas[0]['arquivo'] == 'b.csv'
    assert entradas[1]['arquivo'] == 'a.csv'
    assert entradas[0]['modelo'] == 'KNN'
    assert entradas[0]['total'] == 20
    assert entradas[0]['acuracia'] is None


def test_save_session_com_relatorio_embutido(tmp_path):
    hm = _isolado(tmp_path)
    # Listas (não tuplas): o relatório passa por JSON, que não tem tipo tupla.
    relatorio = {'svm': {'importancias': [['radius_worst', 0.3]]}}
    hm.save_session('a.csv', 'SVM', 10, 4, 6, relatorio=relatorio)
    entradas = hm.load()
    assert entradas[0]['relatorio'] == relatorio


def test_update_last_accuracy_atualiza_apenas_a_mais_recente(tmp_path):
    hm = _isolado(tmp_path)
    hm.save_session('a.csv', 'SVM', 10, 4, 6)
    hm.save_session('b.csv', 'KNN', 20, 8, 12)
    hm.update_last_accuracy({'KNN': 97.9})

    entradas = hm.load()
    assert entradas[0]['acuracia'] == {'KNN': 97.9}
    assert entradas[1]['acuracia'] is None


def test_update_last_accuracy_sem_sessoes_nao_gera_erro(tmp_path):
    hm = _isolado(tmp_path)
    hm.update_last_accuracy({'KNN': 97.9})  # não deve lançar exceção
    assert hm.load() == []


def test_delete_remove_pelo_indice_na_ordem_do_load(tmp_path):
    hm = _isolado(tmp_path)
    hm.save_session('a.csv', 'SVM', 10, 4, 6)
    hm.save_session('b.csv', 'KNN', 20, 8, 12)
    hm.delete(0)  # remove a mais recente (b.csv)

    entradas = hm.load()
    assert len(entradas) == 1
    assert entradas[0]['arquivo'] == 'a.csv'


def test_delete_indice_fora_do_intervalo_nao_altera_nada(tmp_path):
    hm = _isolado(tmp_path)
    hm.save_session('a.csv', 'SVM', 10, 4, 6)
    hm.delete(5)
    assert len(hm.load()) == 1


def test_clear_remove_o_arquivo(tmp_path):
    hm = _isolado(tmp_path)
    hm.save_session('a.csv', 'SVM', 10, 4, 6)
    hm.clear()
    assert hm.load() == []
