"""
Escolhe o limiar de operação de cada modelo e grava ``data/limiares.json``.

Por que não usar 0,5
--------------------
``predict()`` do scikit-learn decide por 0,5, o limiar que minimiza o erro
*total* — uma escolha que trata falso positivo e falso negativo como igualmente
caros. Em rastreio de câncer eles não são: o falso positivo custa um exame
adicional; o falso negativo custa um tumor não tratado. Baixar o limiar troca um
pelo outro de forma deliberada e auditável.

Critério de escolha
-------------------
Maximizar a sensibilidade **sujeita a um piso de especificidade** (padrão 92%).
A formulação importa: pedir apenas "sensibilidade ≥ 98%" não tem freio — o KNN,
cujas probabilidades calibradas são grosseiras (k=4 gera poucos valores
distintos), atende esse pedido com limiar 0,027 e especificidade de 36% no
treino, isto é, acusando quase todo mundo. Com o piso, a busca é limitada por
construção: entre os limiares que mantêm a especificidade aceitável, toma-se o
mais sensível.

Como o limiar é escolhido (e por que assim)
-------------------------------------------
Escolher o limiar olhando o conjunto de teste seria vazamento: o número
publicado deixaria de ser uma estimativa de desempenho futuro e viraria o melhor
caso possível daquele lote. Aqui a escolha usa **apenas o treino**, por
probabilidades *out-of-fold*: em cada uma das 5 dobras, o modelo (com a mesma
calibração de Platt do notebook — Seção 14) é treinado nas outras quatro e prevê
a que ficou de fora. Assim cada paciente do treino recebe uma probabilidade
vinda de um modelo que não o viu, e o limiar é escolhido sobre essas
probabilidades honestas.

O conjunto de teste aparece no relatório apenas como conferência do que a
escolha produziu; ele não participa da decisão.

Uso
---
    .venv/bin/python scripts/calibrar_limiares.py                  # piso 92%
    .venv/bin/python scripts/calibrar_limiares.py --piso 95        # mais conservador
    .venv/bin/python scripts/calibrar_limiares.py --so-relatorio   # não grava nada
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from core.decision import NOME_COMITE
from core.inference import ModelLoader

RANDOM_STATE = 42
PISO_PADRAO = 92.0
BANDA_REVISAO = 0.10

# Cobertura mínima para um modelo receber faixa de recusa. Um modelo que só
# consegue não errar adiando dois terços do lote não está sendo cauteloso: está
# admitindo que suas probabilidades não separam as classes. É o caso do KNN,
# cujas probabilidades calibradas são grosseiras (k=4 gera poucos valores
# distintos) — ele decide 34% do treino e ainda assim erra o único benigno que
# decide. Modelos assim ficam sem faixa, e a recusa não é oferecida para eles.
COBERTURA_MINIMA = 45.0

# Mesmos hiperparâmetros do notebook (Seções 6–9). Reconstruídos aqui, e não
# clonados do .pkl, para que a validação cruzada possa retreiná-los dobra a
# dobra — o .pkl guarda modelos já ajustados em todo o treino.
MODELOS_BASE = {
    'Regressão Logística': LogisticRegression(
        C=0.1, penalty='l2', class_weight='balanced',
        solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=4, min_samples_leaf=2,
        max_features='sqrt', criterion='entropy',
        class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1),
    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=4, weights='distance', metric='manhattan'),
}


def calibrador(base):
    """Envolve um modelo na mesma calibração de Platt usada pelo notebook."""
    return CalibratedClassifierCV(clone(base), method='sigmoid', cv=5, ensemble=False)


def metricas(y, prob, limiar):
    """Sensibilidade, especificidade, F1 e acurácia (%) a um dado limiar."""
    y = np.asarray(y)
    pred = (np.asarray(prob) >= limiar).astype(int)
    vp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    vn = int(((y == 0) & (pred == 0)).sum())
    return {
        'sens': vp / (vp + fn) * 100 if vp + fn else float('nan'),
        'espec': vn / (vn + fp) * 100 if vn + fp else float('nan'),
        'f1': 2 * vp / (2 * vp + fp + fn) * 100 if 2 * vp + fp + fn else float('nan'),
        'acur': (vp + vn) / len(y) * 100,
        'fn': fn, 'fp': fp,
    }


def limiar_mais_sensivel(y, prob, piso_especificidade):
    """
    Menor limiar cuja especificidade ainda respeita o piso.

    Percorre em ordem crescente os valores de probabilidade observados — os
    únicos pontos em que a decisão muda — e devolve o primeiro aceitável. Como
    a especificidade cresce com o limiar, o primeiro aceitável é também o mais
    sensível. Recai em 0,5 se nem o limiar padrão respeitar o piso (modelo
    ruim demais para operar deslocado).
    """
    for candidato in np.unique(np.round(prob, 4)):
        if metricas(y, prob, candidato)['espec'] >= piso_especificidade:
            return float(candidato)
    return 0.5


def limiar_para_sensibilidade(y, prob, alvo):
    """Maior limiar que ainda alcança o alvo de sensibilidade (para comparação)."""
    for candidato in np.unique(np.round(prob, 4))[::-1]:
        if metricas(y, prob, candidato)['sens'] >= alvo:
            return float(candidato)
    return 0.0


def faixa_de_recusa(y, prob, tolerancia: float = 0.0):
    """
    Faixa de probabilidade em que o modelo deve se abster.

    Fora da faixa, a decisão é automática; dentro, o caso volta para o médico.
    Os dois limites saem de uma pergunta cada:

    - inferior: até onde dá para dizer "Benigno" sem passar por cima de nenhum
      maligno? É a menor probabilidade que um paciente maligno recebeu.
    - superior: a partir de onde dá para dizer "Maligno" sem acusar nenhum
      benigno? É a maior probabilidade que um paciente benigno recebeu.

    Com ``tolerancia = 0`` a faixa é a que zera os erros entre os casos
    decididos — e fica larga, porque os dois pacientes mais atípicos do treino
    a determinam sozinhos. Uma tolerância pequena descarta essas caudas e
    encolhe bastante a faixa, ao custo de admitir alguns erros.

    Parameters
    ----------
    y : array-like
        Rótulos reais (0/1).
    prob : array-like
        Probabilidades calibradas out-of-fold.
    tolerancia : float, optional
        Fração de cada classe que se aceita errar (0.02 = 2%).

    Returns
    -------
    tuple[float, float] ou None
        ``(inferior, superior)``, ou None quando as classes se separam
        perfeitamente e não há nada a adiar.
    """
    prob = np.asarray(prob)
    y = np.asarray(y)
    malignos, benignos = prob[y == 1], prob[y == 0]
    if not len(malignos) or not len(benignos):
        return None

    inferior = float(np.percentile(malignos, tolerancia * 100))
    # O limite superior é EXCLUSIVO na decisão ("Maligno a partir de superior"),
    # então precisa ficar um passo acima do pior benigno: no valor exato, esse
    # paciente seria decidido — e decidido errado. O passo é maior que o
    # arredondamento com que a faixa é gravada, para não ser desfeito por ele.
    superior = float(np.percentile(benignos, 100 - tolerancia * 100)) + 1e-4
    return (inferior, superior) if inferior < superior else None


def metricas_com_recusa(y, prob, faixa):
    """Desempenho entre os casos decididos, mais a cobertura, para uma faixa."""
    y, prob = np.asarray(y), np.asarray(prob)
    if faixa is None:
        m = metricas(y, prob, 0.5)
        m['cobertura'] = 100.0
        m['adiados'] = 0
        return m

    inferior, superior = faixa
    decidido = (prob < inferior) | (prob >= superior)
    m = metricas(y[decidido], prob[decidido], superior) if decidido.any() else {
        'sens': float('nan'), 'espec': float('nan'), 'f1': float('nan'),
        'acur': float('nan'), 'fn': 0, 'fp': 0}
    m['cobertura'] = decidido.mean() * 100
    m['adiados'] = int((~decidido).sum())
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--piso', type=float, default=PISO_PADRAO,
                        help=f"especificidade mínima aceitável no treino, em %% (padrão {PISO_PADRAO})")
    parser.add_argument('--banda', type=float, default=BANDA_REVISAO,
                        help="meia-largura da faixa de revisão em torno do limiar (padrão 0.10)")
    parser.add_argument('--so-relatorio', action='store_true',
                        help="apenas imprime a análise, sem gravar data/limiares.json")
    parser.add_argument('--tolerancia-recusa', type=float, default=0.0,
                        help="fração de cada classe que se aceita errar ao definir a faixa de "
                             "recusa (0 = faixa que zera os erros no treino, padrão)")
    args = parser.parse_args()

    loader = ModelLoader()
    if loader.X_train_scaled is None or loader.y_train is None:
        raise SystemExit("O wisconsin.pkl não traz X_train_scaled/y_train — regenere pelo notebook.")

    X_train = np.asarray(loader.X_train_scaled, dtype=float)
    y_train = np.asarray(loader.y_train)

    print(f"Treino: {len(y_train)} pacientes ({int(y_train.sum())} malignos)")
    print("Calculando probabilidades out-of-fold (5 dobras, calibração de Platt por dobra)…\n")

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = {nome: cross_val_predict(calibrador(base), X_train, y_train,
                                   cv=cv, method='predict_proba')[:, 1]
           for nome, base in MODELOS_BASE.items()}
    oof[NOME_COMITE] = np.mean([oof[n] for n in MODELOS_BASE], axis=0)

    limiares = {nome: limiar_mais_sensivel(y_train, p, args.piso) for nome, p in oof.items()}
    desempenho = {nome: metricas(y_train, oof[nome], limiares[nome]) for nome in oof}

    faixas = {nome: faixa_de_recusa(y_train, p, args.tolerancia_recusa)
              for nome, p in oof.items()}
    # Descarta faixas que só "acertam" porque quase não decidem (ver COBERTURA_MINIMA).
    for nome, faixa in list(faixas.items()):
        cobertura = metricas_com_recusa(y_train, oof[nome], faixa)['cobertura']
        if faixa is not None and cobertura < COBERTURA_MINIMA:
            print(f"  {nome}: faixa descartada — decidiria só {cobertura:.1f}% do treino "
                  f"(mínimo {COBERTURA_MINIMA:.0f}%). Este modelo não oferecerá recusa.")
            faixas[nome] = None
    recusa = {nome: metricas_com_recusa(y_train, oof[nome], faixas[nome]) for nome in oof}

    # --- Conferência no teste: exatamente o que o app fará, com os modelos
    # calibrados que estão no .pkl. Não participa da escolha do limiar. ---
    X_test, y_test = _conjunto_de_teste()
    teste = {}
    if X_test is not None:
        for nome in MODELOS_BASE:
            modelo_cal = loader.calibrated_models.get(nome)
            if modelo_cal is not None:
                teste[nome] = modelo_cal.predict_proba(X_test)[:, 1]
        if len(teste) == len(MODELOS_BASE):
            teste[NOME_COMITE] = np.mean([teste[n] for n in MODELOS_BASE], axis=0)

    _relatorio(oof, y_train, teste, y_test, limiares, args.piso)
    _relatorio_recusa(oof, y_train, teste, y_test, faixas, args.tolerancia_recusa)

    if args.so_relatorio:
        print("\n(--so-relatorio: nada foi gravado)")
        return

    destino = os.path.join(REPO_ROOT, 'data', 'limiares.json')
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump({
            'gerado_em': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'piso_especificidade': args.piso,
            'banda_revisao': args.banda,
            'metodo': ('menor limiar com especificidade out-of-fold >= piso '
                       '(5 dobras estratificadas sobre o treino, calibração de Platt por dobra)'),
            'limiares': {k: round(v, 4) for k, v in limiares.items()},
            'desempenho_treino': {
                k: {'sensibilidade': round(v['sens'], 1), 'especificidade': round(v['espec'], 1)}
                for k, v in desempenho.items()
            },
            'tolerancia_recusa': args.tolerancia_recusa,
            'faixas_recusa': {k: [round(v[0], 4), round(v[1], 4)]
                              for k, v in faixas.items() if v is not None},
            # Só faz sentido para quem tem faixa: sem ela, "cobertura" seria
            # sempre 100% e daria a impressão de um modelo excepcional.
            'cobertura_treino': {k: round(v['cobertura'], 1)
                                 for k, v in recusa.items() if faixas.get(k)},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nGravado em {destino}")


def _conjunto_de_teste():
    """Lê X_test_scaled/y_test do .pkl, quando o notebook os empacotou."""
    import cloudpickle

    caminho = os.path.join(REPO_ROOT, 'data', 'wisconsin.pkl')
    with open(caminho, 'rb') as f:
        dados = cloudpickle.load(f)
    X, y = dados.get('X_test_scaled'), dados.get('y_test')
    if X is None or y is None:
        return None, None
    return np.asarray(X, dtype=float), np.asarray(y)


_CABECALHO = (f"{'modelo':22s} {'limiar':>7s} {'sens':>7s} {'espec':>7s} "
              f"{'F1':>7s} {'acur':>7s} {'FN':>4s} {'FP':>4s}")


def _linhas(probabilidades, y, limiares, usar_limiar=True):
    """Imprime uma linha de métricas por modelo."""
    print(_CABECALHO)
    for nome, p in probabilidades.items():
        t = limiares[nome] if usar_limiar else 0.5
        m = metricas(y, p, t)
        print(f"{nome:22s} {t:7.3f} {m['sens']:6.1f}% {m['espec']:6.1f}% "
              f"{m['f1']:6.1f}% {m['acur']:6.1f}% {m['fn']:4d} {m['fp']:4d}")


def _relatorio(oof, y_train, teste, y_test, limiares, piso):
    """Imprime a comparação entre o limiar padrão e o calibrado."""
    print("=" * len(_CABECALHO))
    print("TREINO (out-of-fold) — foi sobre estes números que o limiar foi escolhido")
    print("=" * len(_CABECALHO))
    print("\n--- limiar padrão 0,50")
    _linhas(oof, y_train, limiares, usar_limiar=False)
    print(f"\n--- limiar calibrado (mais sensível com especificidade ≥ {piso:.0f}%)")
    _linhas(oof, y_train, limiares)

    if teste and y_test is not None:
        print("\n" + "=" * len(_CABECALHO))
        print("TESTE (conferência — não participou da escolha)")
        print("=" * len(_CABECALHO))
        print("\n--- limiar padrão 0,50")
        _linhas(teste, y_test, limiares, usar_limiar=False)
        print("\n--- limiar calibrado")
        _linhas(teste, y_test, limiares)

    print("\n--- para comparação: o que custaria exigir 100% de sensibilidade no treino")
    limiares_totais = {n: limiar_para_sensibilidade(y_train, p, 100.0) for n, p in oof.items()}
    _linhas(oof, y_train, limiares_totais)


_CABECALHO_RECUSA = (f"{'modelo':22s} {'faixa de recusa':>18s} {'decide':>8s} {'sens':>7s} "
                     f"{'espec':>7s} {'FN':>4s} {'FP':>4s} {'adia':>5s}")


def _linhas_recusa(probabilidades, y, faixas):
    """Imprime uma linha por modelo com o desempenho entre os casos decididos."""
    print(_CABECALHO_RECUSA)
    for nome, p in probabilidades.items():
        faixa = faixas.get(nome)
        m = metricas_com_recusa(y, p, faixa)
        desenho = f"[{faixa[0]:.3f}; {faixa[1]:.3f})" if faixa else "— (nada a adiar)"
        print(f"{nome:22s} {desenho:>18s} {m['cobertura']:7.1f}% {m['sens']:6.1f}% "
              f"{m['espec']:6.1f}% {m['fn']:4d} {m['fp']:4d} {m['adiados']:5d}")


def _relatorio_recusa(oof, y_train, teste, y_test, faixas, tolerancia):
    """
    Imprime o efeito da opção de recusa e o custo de cada nível de tolerância.

    A tabela de tolerâncias é o que permite escolher o ponto: ela mostra quanto
    de cobertura se ganha ao aceitar errar um pouco entre os casos decididos.
    """
    print("\n" + "=" * len(_CABECALHO_RECUSA))
    print(f"OPÇÃO DE RECUSA (tolerância {tolerancia * 100:.0f}%) — métricas só entre os DECIDIDOS")
    print("=" * len(_CABECALHO_RECUSA))
    print("\n--- treino (out-of-fold), onde a faixa foi escolhida")
    _linhas_recusa(oof, y_train, faixas)

    if teste and y_test is not None:
        print("\n--- teste (conferência — não participou da escolha)")
        _linhas_recusa(teste, y_test, faixas)

    print("\n--- quanto se ganha de cobertura aceitando errar um pouco (no treino)")
    print(f"{'tolerância':>11s} " + " ".join(f"{n[:12]:>13s}" for n in oof))
    for tol in (0.0, 0.01, 0.02, 0.05):
        celulas = []
        for nome, p in oof.items():
            f = faixa_de_recusa(y_train, p, tol)
            m = metricas_com_recusa(y_train, p, f)
            celulas.append(f"{m['cobertura']:5.1f}% {m['fn']}FN/{m['fp']}FP".rjust(13))
        print(f"{tol * 100:10.0f}% " + " ".join(celulas))


if __name__ == '__main__':
    main()
