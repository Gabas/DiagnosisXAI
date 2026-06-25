"""
Módulo responsável pela persistência do histórico de sessões de diagnóstico.
"""

import json
import os
from datetime import datetime


class HistoryManager:
    """
    Gerencia a leitura e escrita do histórico de sessões em um arquivo JSON.

    Attributes
    ----------
    _path : str
        Caminho absoluto para o arquivo history.json em data/.
    """

    def __init__(self):
        """Inicializa o gerenciador resolvendo o caminho do arquivo de histórico."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._path = os.path.join(base_dir, 'data', 'history.json')

    def load(self) -> list[dict]:
        """
        Carrega e retorna todas as entradas do histórico.

        Returns
        -------
        list[dict]
            Lista de sessões ordenadas da mais recente para a mais antiga.
            Retorna lista vazia se o arquivo ainda não existir.
        """
        if not os.path.exists(self._path):
            return []
        with open(self._path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_session(self, arquivo: str, modelo: str, total: int,
                     malignos: int | None, benignos: int | None):
        """
        Salva uma nova sessão de diagnóstico no topo do histórico.

        Parameters
        ----------
        arquivo : str
            Nome do arquivo CSV utilizado no diagnóstico.
        modelo : str
            Nome do modelo de IA aplicado.
        total : int
            Número total de pacientes no lote.
        malignos : int ou None
            Contagem de diagnósticos Maligno. None para modo comparação.
        benignos : int ou None
            Contagem de diagnósticos Benigno. None para modo comparação.
        """
        entries = self.load()
        entries.insert(0, {
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'arquivo': arquivo,
            'modelo': modelo,
            'total': total,
            'malignos': malignos,
            'benignos': benignos,
            'acuracia': None,
        })
        self._write(entries)

    def update_last_accuracy(self, acuracia: dict):
        """
        Atualiza a acurácia da sessão mais recente após a etapa de auditoria.

        Parameters
        ----------
        acuracia : dict
            Dicionário mapeando nome do modelo para acurácia percentual.
            Ex.: {'Modelo Selecionado': 97.31} ou {'RAN': 97.31, 'SVM': 96.50}.
        """
        entries = self.load()
        if entries:
            entries[0]['acuracia'] = acuracia
            self._write(entries)

    def clear(self):
        """Remove o arquivo de histórico do disco, apagando todas as sessões."""
        if os.path.exists(self._path):
            os.remove(self._path)

    def _write(self, entries: list[dict]):
        """
        Serializa e grava a lista de sessões no arquivo JSON.

        Parameters
        ----------
        entries : list[dict]
            Lista completa de sessões a ser persistida.
        """
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
