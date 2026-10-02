"""
Gera o executável do DiagnosisXAI para o sistema em que roda.

    python empacotamento/gerar.py

Passos:

1. roda o PyInstaller com ``DiagnosisXAI.spec``;
2. põe ao lado do programa a pasta ``exemplos/`` (os CSVs de teste) e o
   ``LEIA-ME.txt``;
3. **testa o programa pronto** com ``--autoteste``, isolado numa pasta
   temporária, e aborta se algo falhar;
4. compacta tudo em ``dist/DiagnosisXAI-<sistema>.zip``.

O passo 3 é o que importa: o PyInstaller só empacota o que enxerga nos
imports, e o que fica de fora só aparece quando o programa roda.

O PyInstaller não gera executável para outro sistema. Para o Windows, o
workflow ``.github/workflows/executavel.yml`` roda este mesmo script numa
máquina Windows do GitHub Actions.

Use o ambiente de ``requirements-executavel.txt`` (versões exatas): o
``wisconsin.pkl`` precisa das mesmas versões de scikit-learn, numpy e pandas
que o geraram.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(RAIZ, 'dist')
PASTA_APP = os.path.join(DIST, 'DiagnosisXAI')
EXEMPLOS = ('dataTeste_sem_diagnostico.csv', 'dataTeste_com_diagnostico.csv')


def _sistema() -> str:
    """Rótulo do sistema para o nome do .zip, ex.: 'windows-x64', 'linux-x64'."""
    arq = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}
    return f"{platform.system().lower()}-{arq.get(platform.machine().lower(), platform.machine())}"


def _tamanho_mb(caminho: str) -> float:
    total = 0
    for raiz, _, arquivos in os.walk(caminho):
        total += sum(os.path.getsize(os.path.join(raiz, a)) for a in arquivos)
    return total / 1e6


def empacotar():
    print(f"[1/4] PyInstaller ({sys.version.split()[0]}, {_sistema()})", flush=True)
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                    '--distpath', DIST, '--workpath', os.path.join(RAIZ, 'build'),
                    os.path.join(RAIZ, 'empacotamento', 'DiagnosisXAI.spec')],
                   check=True, cwd=RAIZ)


def completar_pasta():
    print("[2/4] Exemplos e LEIA-ME", flush=True)
    destino = os.path.join(PASTA_APP, 'exemplos')
    os.makedirs(destino, exist_ok=True)
    for nome in EXEMPLOS:
        shutil.copy2(os.path.join(RAIZ, 'data', nome), destino)
    shutil.copy2(os.path.join(RAIZ, 'empacotamento', 'LEIA-ME.txt'), PASTA_APP)


def autotestar():
    print("[3/4] Autoteste do programa pronto", flush=True)
    programa = os.path.join(PASTA_APP, 'DiagnosisXAI' + ('.exe' if os.name == 'nt' else ''))
    isolado = tempfile.mkdtemp(prefix='diagnosisxai-build-')
    log = os.path.join(isolado, 'autoteste.txt')
    ambiente = dict(os.environ, DIAGNOSISXAI_DADOS=isolado, DIAGNOSISXAI_AUTOTESTE_LOG=log)
    inicio = time.time()
    # subprocess espera o processo terminar mesmo sendo um executável sem
    # console no Windows — diferente de chamá-lo direto do prompt.
    processo = subprocess.run([programa, '--autoteste'], env=ambiente, timeout=900,
                              capture_output=True, text=True, errors='replace')
    duracao = time.time() - inicio
    if os.path.exists(log):
        with open(log, encoding='utf-8') as arquivo:
            print(arquivo.read())
    else:
        # Morreu antes de gravar o relatório (ex.: módulo ausente no pacote):
        # a saída do próprio programa é a única pista.
        print(processo.stdout, processo.stderr, sep="\n")
    if processo.returncode != 0:
        sys.exit(f"Autoteste falhou (código {processo.returncode}) — o executável NÃO está pronto.")
    print(f"Autoteste passou em {duracao:.0f} s.", flush=True)


def compactar() -> str:
    print("[4/4] Compactando", flush=True)
    base = os.path.join(DIST, f"DiagnosisXAI-{_sistema()}")
    zip_final = shutil.make_archive(base, 'zip', root_dir=DIST, base_dir='DiagnosisXAI')
    print(f"Pasta: {_tamanho_mb(PASTA_APP):.0f} MB  ·  zip: "
          f"{os.path.getsize(zip_final) / 1e6:.0f} MB  ->  {zip_final}")
    return zip_final


if __name__ == '__main__':
    empacotar()
    completar_pasta()
    autotestar()
    compactar()
