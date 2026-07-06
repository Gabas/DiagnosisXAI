"""
Construtores compartilhados dos relatórios SHAP e UMAP.

Tanto o Passo 5 (predict_view, lote recém-processado) quanto o histórico
(history_view, sessão salva) abrem os relatórios SHAP e UMAP a partir dos mesmos
dados: os arrays do lote (padronizado + bruto) e os artefatos do modelo já
carregados no ModelLoader. Centralizar a construção aqui evita duplicar a lógica
e garante que a reabertura pelo histórico seja idêntica à do processamento.

Estratégia de persistência (relatório interativo reabre igual):
- SHAP: guarda-se o lote (X padronizado, X bruto, índices) + a lista de modelos
  com SHAP. O explicador é reconstruído do modelo (no .pkl) ao abrir; as
  contribuições por paciente continuam sendo calculadas sob demanda (lazy).
- UMAP: guarda-se a projeção 2D do lote (batch_2d) + os pacientes. O fundo
  (embedding de treino) vem do ModelLoader.
"""

import numpy as np

# Chave curta do SHAP <-> nome do modelo no ModelLoader.
MODELO_POR_KEY = {
    'dt': 'Árvore de Decisão',
    'rf': 'Random Forest',
    'lr': 'Regressão Logística',
    'svm': 'SVM',
    'knn': 'KNN',
}


def projetar_umap(loader, X_scaled_batch):
    """
    Projeta um lote padronizado no embedding UMAP do treino.

    Cada paciente é posicionado pela média ponderada (por distância) das posições
    dos seus vizinhos de treino no embedding — rápido, determinístico e sem
    depender do umap em tempo de execução.

    Parameters
    ----------
    loader : ModelLoader
        Fonte de ``X_train_scaled`` e ``umap_train_2d``.
    X_scaled_batch : array-like
        Lote padronizado (n × n_features).

    Returns
    -------
    numpy.ndarray ou None
        Posições 2D do lote (n × 2), ou None se os artefatos UMAP faltarem.
    """
    from sklearn.neighbors import NearestNeighbors

    if loader.umap_train_2d is None or loader.X_train_scaled is None:
        return None

    Xtr = np.asarray(loader.X_train_scaled)
    emb = np.asarray(loader.umap_train_2d)
    Xb = np.asarray(X_scaled_batch, dtype=float)

    k = min(10, len(Xtr))
    distancias, vizinhos = NearestNeighbors(n_neighbors=k).fit(Xtr).kneighbors(Xb)
    pesos = 1.0 / (distancias + 1e-9)
    pesos /= pesos.sum(axis=1, keepdims=True)
    return np.einsum('nk,nkd->nd', pesos, emb[vizinhos])


def abrir_umap(master, loader, batch_2d, pacientes):
    """
    Abre o Mapa Populacional (UMAP) a partir da projeção do lote.

    Parameters
    ----------
    master : ctk widget
        Janela que origina o relatório.
    loader : ModelLoader
        Fonte do embedding de treino (fundo do mapa) e dos rótulos.
    batch_2d : array-like
        Projeção 2D dos pacientes do lote (n × 2).
    pacientes : list[dict]
        Itens {'indice', 'classe'} de cada paciente do lote.

    Returns
    -------
    UmapMapWindow ou None
        A janela criada, ou None se o embedding de treino não estiver disponível.
    """
    from views.report_window_umap import UmapMapWindow

    if loader.umap_train_2d is None or batch_2d is None:
        return None
    return UmapMapWindow(
        master, loader.umap_train_2d, loader.y_train, batch_2d, pacientes)


def construir_shap(loader, key, cache=None):
    """
    Constrói (ou reaproveita do cache) o explicador SHAP de um modelo.

    Parameters
    ----------
    loader : ModelLoader
        Fonte do modelo e do background do SHAP.
    key : str
        Chave do modelo ('dt', 'rf', 'lr', 'svm' ou 'knn').
    cache : dict ou None
        Cache opcional {key: ShapExplainer} para evitar reconstruções.

    Returns
    -------
    ShapExplainer
        Explicador SHAP pronto para calcular contribuições por paciente.
    """
    from core.shap_explainer import ShapExplainer

    if cache is not None and key in cache:
        return cache[key]

    modelo = loader.models.get(MODELO_POR_KEY[key])
    explainer = ShapExplainer(modelo, key, loader.feature_names, loader.shap_background)
    if cache is not None:
        cache[key] = explainer
    return explainer


def abrir_shap(master, loader, key, X_scaled, X_raw, indices, cache=None):
    """
    Abre o relatório SHAP de um modelo a partir dos dados do lote.

    Reconstrói o explicador do modelo carregado e monta a lista de pacientes com
    a predição daquele modelo (como no processamento). As contribuições por
    paciente seguem sendo calculadas sob demanda dentro da janela.

    Parameters
    ----------
    master : ctk widget
        Janela que origina o relatório.
    loader : ModelLoader
        Fonte do modelo, do background e das importâncias globais do SHAP.
    key : str
        Chave do modelo ('dt', 'rf', 'lr', 'svm' ou 'knn').
    X_scaled : array-like
        Lote padronizado (n × n_features).
    X_raw : array-like
        Lote bruto/limpo (n × n_features) — entrada da árvore e valores exibidos.
    indices : list[int]
        Rótulos originais das linhas do lote.
    cache : dict ou None
        Cache opcional de explicadores SHAP.

    Returns
    -------
    ShapReportWindow
        A janela de relatório SHAP recém-criada.
    """
    from core.shap_explainer import ShapExplainer
    from views.report_window_shap import ShapReportWindow

    explainer = construir_shap(loader, key, cache)
    modelo = loader.models.get(MODELO_POR_KEY[key])

    X_scaled = np.asarray(X_scaled, dtype=float)
    X_raw = np.asarray(X_raw, dtype=float)
    # A árvore foi treinada em dados brutos; os demais em dados padronizados.
    entrada = X_raw if key == 'dt' else X_scaled

    preds = modelo.predict(entrada)
    pacientes = [
        {'indice': int(indices[i]),
         'classe': 'Maligno' if preds[i] == ShapExplainer.CLASSE_MALIGNO else 'Benigno'}
        for i in range(len(indices))
    ]
    importancias = loader.shap_importances.get(key, [])
    return ShapReportWindow(
        master, explainer, entrada, X_raw, pacientes, importancias, MODELO_POR_KEY[key])
