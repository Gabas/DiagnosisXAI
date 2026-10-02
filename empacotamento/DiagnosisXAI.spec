# -*- mode: python ; coding: utf-8 -*-
"""
Receita do PyInstaller para o executável do DiagnosisXAI.

Não rode este arquivo direto: use ``python empacotamento/gerar.py``, que roda
o PyInstaller com esta receita, junta os CSVs de exemplo e o LEIA-ME, testa o
programa pronto com ``--autoteste`` e gera o .zip.

Formato: pasta (``onedir``), e não arquivo único. Um arquivo único teria de
descompactar ~500 MB numa pasta temporária a cada vez que o app abre.
"""

import os

from PyInstaller.utils.hooks import collect_data_files

RAIZ = os.path.abspath(os.path.join(SPECPATH, '..'))
APP = os.path.join(RAIZ, 'app')

# Recursos somente leitura, no mesmo "data/" que o app espera (core/caminhos.py).
datas = [(os.path.join(RAIZ, 'data', nome), 'data')
         for nome in ('wisconsin.pkl', 'limiares.json', 'umap_train_2d.npy')]
datas += collect_data_files('customtkinter')   # temas (.json) e fontes
datas += collect_data_files('bokeh')           # BokehJS, embutido nos HTMLs interativos

# O .pkl referencia estas classes pelo nome, e nenhuma é importada pelo código
# do app. A análise de imports do PyInstaller não as veria. A lista sai de um
# Unpickler que registra cada find_class ao carregar o wisconsin.pkl.
hiddenimports = [
    'core.explainers',
    'sklearn.calibration',
    'sklearn.decomposition._pca',
    'sklearn.ensemble._forest',
    'sklearn.linear_model._logistic',
    'sklearn.neighbors._classification',
    'sklearn.preprocessing._data',
    'sklearn.svm._classes',
    'sklearn.tree._classes',
    'sklearn.tree._tree',
    'autoteste',
]

# Bibliotecas do notebook, ou que alguma dependência importa só de forma
# opcional. Nada disso roda no app; ficando de fora, o pacote encolhe.
excludes = [
    'umap', 'pynndescent', 'mlxtend', 'scikit_posthocs', 'seaborn', 'plotly',
    'yellowbrick', 'statsmodels', 'IPython', 'ipykernel', 'ipywidgets', 'jupyter',
    'jupyter_client', 'jupyter_core', 'jupyterlab', 'notebook', 'nbformat',
    'pytest', '_pytest', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'torch',
    'tensorflow', 'sphinx', 'docutils',
]

a = Analysis(
    [os.path.join(APP, 'main.py')],
    pathex=[APP],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DiagnosisXAI',
    console=False,          # sem janela preta de terminal atrás do app
    upx=False,              # UPX costuma disparar falso positivo de antivírus
    debug=False,
    strip=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name='DiagnosisXAI',
    upx=False,
    strip=False,
)
