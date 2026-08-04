"""
Métricas de desempenho e leitura crítica dos modelos na auditoria (Passo 4).

A auditoria compara o diagnóstico previsto com o gabarito (diagnóstico real) do
lote. Acurácia sozinha é uma medida pobre neste domínio: numa base desbalanceada
ela esconde exatamente o erro mais caro — o falso negativo, um tumor maligno
classificado como benigno. Por isso este módulo decompõe o desempenho na matriz
de confusão e nas métricas derivadas usadas no notebook (Seções 5–9):

- Sensibilidade (revocação da classe Maligno): dos tumores realmente malignos,
  quantos o modelo pegou. É a métrica crítica em rastreio oncológico.
- Especificidade (revocação da classe Benigno): dos benignos, quantos foram
  poupados de um alarme falso.
- Precisão (valor preditivo positivo) e VPN (valor preditivo negativo): dado o
  que o modelo disse, qual a chance de estar certo — dependem da prevalência do
  lote, e por isso não são comparáveis entre lotes diferentes.
- F1: média harmônica entre precisão e sensibilidade, resumo único para quando
  as duas importam.

Todas as proporções vêm acompanhadas de um intervalo de confiança de Wilson de
95%. Num lote de dezenas de pacientes, a diferença entre 97% e 95% costuma estar
inteiramente dentro do ruído amostral — o intervalo torna isso explícito e evita
a leitura ingênua de um ranking por décimos de ponto percentual.

O módulo é puro (só depende de numpy/sklearn) e não conhece a interface: a view
formata o que ele devolve.
"""

import math

import numpy as np

CLASSE_POSITIVA = 'Maligno'
CLASSE_NEGATIVA = 'Benigno'

# Nome completo de cada modelo a partir da sigla usada nas colunas 'IA_XXX' do
# modo "Todos (Comparação)" (ver PredictorEngine.predict, que corta os 3
# primeiros caracteres do nome). O mapa evita exibir "RAN"/"REG" na tabela.
SIGLA_PARA_MODELO = {
    'ÁRV': 'Árvore de Decisão',
    'ARV': 'Árvore de Decisão',
    'KNN': 'KNN',
    'RAN': 'Random Forest',
    'REG': 'Regressão Logística',
    'SVM': 'SVM',
}

# Características intrínsecas de cada modelo — o que o algoritmo oferece e o que
# ele cobra, independentemente do resultado deste lote. A leitura crítica combina
# estes traços fixos com o que os números do lote mostram.
PERFIL_MODELO = {
    'Árvore de Decisão': {
        'forte': "explicação auditável de ponta a ponta — o caminho de regras da raiz à "
                 "folha é a própria justificativa da decisão, o que a torna o modelo mais "
                 "defensável diante de um clínico",
        'limite': "árvore única decora o treino (overfitting) e é instável: pequenas mudanças "
                  "nos dados mudam a estrutura. Além disso, suas folhas são puras, então a "
                  "'certeza' que ela reporta é sempre 0% ou 100% e não serve para triagem por "
                  "risco",
    },
    'KNN': {
        'forte': "explicação por casos concretos — mostra os pacientes de treino mais parecidos "
                 "que decidiram o diagnóstico, um argumento intuitivo em contexto clínico",
        'limite': "com k=4 a decisão depende de pouquíssimos vizinhos, o que a torna sensível a "
                  "ruído local; e o custo de predição cresce com o tamanho da base, pois compara "
                  "com todo o treino a cada paciente",
    },
    'Regressão Logística': {
        'forte': "modelo linear com coeficientes assinados: dá para ler o peso e a direção de "
                 "cada biomarcador, e as probabilidades tendem a ser bem calibradas — útil "
                 "quando se quer decidir por faixa de risco, não só pelo rótulo",
        'limite': "assume fronteira essencialmente linear entre benigno e maligno; interações "
                  "complexas entre biomarcadores lhe escapam por construção",
    },
    'Random Forest': {
        'forte': "ensemble de 500 árvores com bagging: erros individuais se cancelam, o que "
                 "costuma dar o desempenho mais estável do conjunto e o menos sensível a "
                 "ajuste fino",
        'limite': "não é interpretável diretamente — a explicação depende de SHAP ou do consenso "
                  "das árvores, camadas que aproximam o raciocínio em vez de exibi-lo; e o "
                  "custo computacional é o maior dos cinco",
    },
    'SVM': {
        'forte': "margem máxima com kernel RBF: lida bem com fronteiras não-lineares em espaço "
                 "de alta dimensão (30 biomarcadores) sem precisar de muitos dados",
        'limite': "opaco por natureza (a decisão vive num espaço projetado), sensível à escolha "
                  "de hiperparâmetros e, com probability=True, as probabilidades saem de um "
                  "ajuste de Platt à parte — não da própria fronteira",
    },
}

# Diferença mínima (em pontos percentuais) para tratar dois modelos como
# distintos numa comparação. Abaixo disso, a diferença é ruído amostral em lotes
# desta ordem de grandeza e o ranking não deve ser levado a sério.
MARGEM_EMPATE = 0.5

# Abaixo deste número de pacientes de uma classe, o intervalo de confiança da
# métrica correspondente fica largo demais para sustentar comparação entre modelos.
MINIMO_POR_CLASSE = 30


def ic_wilson(sucessos: int, total: int, z: float = 1.96):
    """
    Intervalo de confiança de Wilson para uma proporção, em porcentagem.

    Prefere-se Wilson ao intervalo normal (Wald) porque ele continua válido com
    poucas amostras e em proporções perto de 0% ou 100% — exatamente o regime
    desta auditoria, em que um modelo pode acertar 100% de um punhado de casos
    malignos sem que isso signifique sensibilidade perfeita.

    Parameters
    ----------
    sucessos : int
        Número de acertos.
    total : int
        Número de tentativas.
    z : float, optional
        Escore normal do nível de confiança (1.96 = 95%).

    Returns
    -------
    tuple[float, float] ou None
        (limite_inferior, limite_superior) em porcentagem, ou None se ``total``
        for zero.
    """
    if total <= 0:
        return None

    p = sucessos / total
    denominador = 1 + z ** 2 / total
    centro = (p + z ** 2 / (2 * total)) / denominador
    margem = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominador
    return (max(0.0, (centro - margem) * 100), min(100.0, (centro + margem) * 100))


def _proporcao(numerador: int, denominador: int):
    """Devolve a proporção em porcentagem, ou None quando não é definida."""
    if denominador <= 0:
        return None
    return numerador / denominador * 100


def calcular_metricas(reais, previstos, certezas=None) -> dict:
    """
    Calcula a matriz de confusão e as métricas derivadas de um modelo no lote.

    Parameters
    ----------
    reais : sequence of str
        Diagnóstico real de cada paciente ('Maligno' ou 'Benigno').
    previstos : sequence of str
        Diagnóstico previsto pelo modelo, na mesma ordem.
    certezas : sequence of float, optional
        Probabilidade de malignidade (0–100) por paciente, quando o modelo a
        fornece. Habilita o cálculo do ROC-AUC, que mede a capacidade de
        ordenar pacientes por risco independentemente do limiar de 50%.

    Returns
    -------
    dict
        Chaves: ``n``, ``vp``, ``fn``, ``fp``, ``vn``, ``prevalencia``,
        ``acuracia``, ``sensibilidade``, ``especificidade``, ``precisao``,
        ``vpn``, ``f1``, ``auc`` e os intervalos ``ic_acuracia``,
        ``ic_sensibilidade``, ``ic_especificidade``. Proporções em porcentagem;
        ``None`` quando indefinidas (ex.: especificidade sem nenhum benigno real).
    """
    reais = list(reais)
    previstos = list(previstos)

    vp = sum(1 for r, p in zip(reais, previstos) if r == CLASSE_POSITIVA and p == CLASSE_POSITIVA)
    fn = sum(1 for r, p in zip(reais, previstos) if r == CLASSE_POSITIVA and p == CLASSE_NEGATIVA)
    fp = sum(1 for r, p in zip(reais, previstos) if r == CLASSE_NEGATIVA and p == CLASSE_POSITIVA)
    vn = sum(1 for r, p in zip(reais, previstos) if r == CLASSE_NEGATIVA and p == CLASSE_NEGATIVA)

    n = vp + fn + fp + vn
    positivos = vp + fn
    negativos = fp + vn

    f1 = _proporcao(2 * vp, 2 * vp + fp + fn)

    return {
        'n': n,
        'vp': vp, 'fn': fn, 'fp': fp, 'vn': vn,
        'prevalencia': _proporcao(positivos, n),
        'acuracia': _proporcao(vp + vn, n),
        'sensibilidade': _proporcao(vp, positivos),
        'especificidade': _proporcao(vn, negativos),
        'precisao': _proporcao(vp, vp + fp),
        'vpn': _proporcao(vn, vn + fn),
        'f1': f1,
        'auc': _roc_auc(reais, certezas),
        'ic_acuracia': ic_wilson(vp + vn, n),
        'ic_sensibilidade': ic_wilson(vp, positivos),
        'ic_especificidade': ic_wilson(vn, negativos),
    }


def _roc_auc(reais, certezas):
    """
    Calcula o ROC-AUC a partir das certezas, quando disponíveis e aplicáveis.

    Retorna None se o modelo não fornece certeza (ex.: Árvore de Decisão) ou se
    o lote tem uma única classe real — caso em que a área sob a curva ROC não é
    definida.
    """
    if certezas is None:
        return None

    y = np.array([1 if r == CLASSE_POSITIVA else 0 for r in reais])
    if y.min() == y.max():
        return None

    scores = np.asarray(certezas, dtype=float)
    if len(scores) != len(y) or np.isnan(scores).any():
        return None

    # AUC pela estatística de Mann-Whitney (média dos postos), equivalente à área
    # sob a curva ROC e com tratamento correto de empates via postos médios.
    ordem = scores.argsort()
    postos = np.empty(len(scores), dtype=float)
    postos[ordem] = np.arange(1, len(scores) + 1)
    for valor in np.unique(scores):
        iguais = scores == valor
        if iguais.sum() > 1:
            postos[iguais] = postos[iguais].mean()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    soma_postos = postos[y == 1].sum()
    return float((soma_postos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def avaliar_modelos(df, coluna_real: str = 'Diagnóstico_Real', nome_modelo: str = None) -> dict:
    """
    Avalia todos os modelos presentes no resultado do lote contra o gabarito.

    Reconhece os dois formatos produzidos pelo ``PredictorEngine``: a coluna
    única ``Diagnóstico_IA`` (um modelo) e as colunas ``IA_XXX`` (modo
    comparação), traduzindo as siglas para o nome completo do modelo.

    Parameters
    ----------
    df : pandas.DataFrame
        ``df_resultado`` já acrescido da coluna de diagnóstico real.
    coluna_real : str, optional
        Nome da coluna com o gabarito.
    nome_modelo : str, optional
        Nome do modelo que gerou a coluna ``Diagnóstico_IA``. Informá-lo é o
        que permite à leitura crítica reconhecer o algoritmo e comentar suas
        características; sem ele, o modelo aparece como "Modelo Selecionado".

    Returns
    -------
    dict
        {nome_do_modelo: métricas}, ordenado da maior para a menor
        sensibilidade (desempate por F1) — o critério que importa em rastreio
        oncológico. Vazio se não houver coluna de diagnóstico no DataFrame.
    """
    reais = df[coluna_real].tolist()
    certezas = df['Certeza_Maligno(%)'].tolist() if 'Certeza_Maligno(%)' in df.columns else None

    colunas_ia = [c for c in df.columns if str(c).startswith('IA_')]
    if colunas_ia:
        # Modo comparação: nenhuma coluna de certeza por modelo, logo sem AUC.
        pares = [(SIGLA_PARA_MODELO.get(c[3:], c[3:]), df[c].tolist(), None) for c in colunas_ia]
    elif 'Diagnóstico_IA' in df.columns:
        rotulo = nome_modelo if nome_modelo in PERFIL_MODELO else 'Modelo Selecionado'
        pares = [(rotulo, df['Diagnóstico_IA'].tolist(), certezas)]
    else:
        return {}

    metricas = {nome: calcular_metricas(reais, previstos, certeza)
                for nome, previstos, certeza in pares}

    return dict(sorted(
        metricas.items(),
        key=lambda item: (item[1]['sensibilidade'] or 0, item[1]['f1'] or 0),
        reverse=True,
    ))


def _frase(texto: str) -> str:
    """
    Converte um traço de ``PERFIL_MODELO`` em frase: maiúscula inicial e ponto.

    Não se usa ``str.capitalize()`` porque ele rebaixaria o restante do texto,
    estragando termos técnicos ("kernel RBF", "SHAP", "probability=True").
    """
    return texto[0].upper() + texto[1:] + "."


def _melhores(metricas_por_modelo: dict, chave: str) -> set:
    """
    Nomes dos modelos no topo de uma métrica, tolerando empates técnicos.

    Quando todos os modelos empatam (o caso comum na especificidade, em que
    vários zeram os falsos positivos), devolve conjunto vazio: dizer que cada um
    deles "está entre os melhores" seria elogio vazio, já que a métrica não
    distingue ninguém.
    """
    valores = {nome: m[chave] for nome, m in metricas_por_modelo.items() if m[chave] is not None}
    if not valores or max(valores.values()) - min(valores.values()) <= MARGEM_EMPATE:
        return set()
    topo = max(valores.values())
    return {nome for nome, v in valores.items() if topo - v <= MARGEM_EMPATE}


def analise_critica(nome: str, metricas: dict, metricas_por_modelo: dict) -> dict:
    """
    Monta a leitura crítica de um modelo: pontos fortes, ressalvas e veredito.

    Combina o que os números deste lote mostram (falsos negativos, desequilíbrio
    entre sensibilidade e especificidade, largura do intervalo de confiança,
    comparação com os demais modelos e com o palpite trivial da classe
    majoritária) com as características intrínsecas do algoritmo em
    ``PERFIL_MODELO``.

    Parameters
    ----------
    nome : str
        Nome do modelo, como devolvido por :func:`avaliar_modelos`.
    metricas : dict
        Métricas desse modelo.
    metricas_por_modelo : dict
        Métricas de todos os modelos avaliados, para as comparações relativas.

    Returns
    -------
    dict
        {'fortes': list[str], 'ressalvas': list[str], 'veredito': str}.
    """
    fortes, ressalvas = [], []

    sens = metricas['sensibilidade']
    espec = metricas['especificidade']
    acuracia = metricas['acuracia']
    fn, fp = metricas['fn'], metricas['fp']
    n = metricas['n']
    comparando = len(metricas_por_modelo) > 1

    perfil = PERFIL_MODELO.get(nome)
    if perfil:
        fortes.append(_frase(perfil['forte']))

    # --- O erro que importa: falso negativo (câncer classificado como benigno) ---
    if fn == 0 and metricas['vp'] + fn > 0:
        fortes.append(f"Não deixou passar nenhum tumor maligno neste lote "
                      f"({metricas['vp']} de {metricas['vp']} detectados).")
    elif fn > 0:
        ressalvas.append(f"Deixou passar {fn} caso(s) maligno(s) como benigno(s) — é o erro mais "
                         f"caro do domínio, pois adia o tratamento e não gera nenhum sinal de "
                         f"alerta ao clínico.")

    if fp > 0:
        ressalvas.append(f"Gerou {fp} alarme(s) falso(s): pacientes benignos encaminhados a "
                         f"exames adicionais, com custo e ansiedade evitáveis.")

    # --- Posição relativa aos demais modelos do lote ---
    if comparando:
        if nome in _melhores(metricas_por_modelo, 'sensibilidade'):
            fortes.append("Está entre os melhores do lote em sensibilidade — a métrica decisiva "
                          "para rastreio.")
        if nome in _melhores(metricas_por_modelo, 'especificidade'):
            fortes.append("Está entre os melhores do lote em especificidade: poupa benignos de "
                          "investigação desnecessária.")
        if nome in _melhores(metricas_por_modelo, 'f1'):
            fortes.append("Melhor F1 do lote — o equilíbrio mais favorável entre pegar malignos "
                          "e não alarmar benignos.")

    # --- Assimetria entre os dois tipos de erro ---
    if sens is not None and espec is not None and espec - sens > 5:
        ressalvas.append(f"Erra para o lado perigoso: especificidade ({espec:.1f}%) bem acima da "
                         f"sensibilidade ({sens:.1f}%). Está mais protegido contra alarme falso "
                         f"do que contra deixar passar um câncer.")

    # --- Nada foi aprendido? Comparação com o palpite trivial ---
    prevalencia = metricas['prevalencia']
    if prevalencia is not None and acuracia is not None:
        trivial = max(prevalencia, 100 - prevalencia)
        if acuracia <= trivial + MARGEM_EMPATE:
            ressalvas.append(f"A acurácia ({acuracia:.1f}%) não supera de forma clara o palpite "
                             f"trivial de responder sempre a classe majoritária ({trivial:.1f}%) "
                             f"— neste lote o modelo não demonstra ganho real.")

    # --- Incerteza amostral: o lote é pequeno demais para o ranking? ---
    positivos = metricas['vp'] + fn
    ic_sens = metricas['ic_sensibilidade']
    if positivos < MINIMO_POR_CLASSE and ic_sens:
        ressalvas.append(f"Só {positivos} paciente(s) maligno(s) no lote: a sensibilidade tem "
                         f"IC 95% de [{ic_sens[0]:.1f}%, {ic_sens[1]:.1f}%]. É larga demais para "
                         f"declarar um vencedor por diferença de décimos.")

    if metricas['auc'] is not None:
        fortes.append(f"ROC-AUC de {metricas['auc']:.3f}: ordena os pacientes por risco bem "
                      f"além do que o corte fixo de 50% aproveita — dá margem para baixar o "
                      f"limiar e ganhar sensibilidade.")

    if perfil:
        ressalvas.append(_frase(perfil['limite']))

    return {
        'fortes': fortes,
        'ressalvas': ressalvas,
        'veredito': _veredito(metricas),
    }


def _veredito(metricas: dict) -> str:
    """
    Resume em uma frase para que uso o modelo se presta neste lote.

    O julgamento é guiado pela sensibilidade (que erros o modelo comete do lado
    perigoso) e só depois pela especificidade — nenhum patamar aqui equivale a
    aval para uso autônomo: o laudo continua sendo humano.
    """
    sens, espec = metricas['sensibilidade'], metricas['especificidade']
    fn, fp, n = metricas['fn'], metricas['fp'], metricas['n']
    malignos = metricas['vp'] + fn

    if sens is None or espec is None:
        return "Lote sem as duas classes no gabarito — não dá para julgar o modelo por ele."

    if fn == 0 and espec >= 95:
        return ("Perfil adequado para triagem assistida: não perdeu nenhum maligno e manteve os "
                "alarmes falsos sob controle. Ainda assim, o laudo final é humano.")
    if fn == 0:
        return (f"Bom para rastreio — não perdeu malignos —, mas ao custo de {fp} alarme(s) "
                f"falso(s). Aceitável quando a confirmação seguinte é barata.")
    if sens >= 95:
        return (f"Sensibilidade alta, porém não perfeita ({fn} de {malignos} malignos perdidos): "
                f"serve como filtro inicial, nunca como palavra final para descartar "
                f"malignidade.")
    if sens >= 85:
        return (f"Sensibilidade intermediária — perdeu {fn} de {malignos} malignos. Utilizável "
                f"como segunda opinião ao lado de um modelo mais sensível, não como triagem "
                f"isolada.")
    if espec >= 95:
        return ("Mais útil para confirmar do que para descartar: quando aponta maligno costuma "
                "acertar, mas perde casos demais para funcionar como rastreio.")
    return (f"Desempenho insuficiente para uso isolado neste lote de {n} paciente(s): erra nos "
            f"dois sentidos e exigiria revisão humana de todos os casos.")


def ressalvas_do_lote(metricas_por_modelo: dict) -> list:
    """
    Ressalvas metodológicas que valem para a auditoria inteira, não por modelo.

    Parameters
    ----------
    metricas_por_modelo : dict
        Saída de :func:`avaliar_modelos`.

    Returns
    -------
    list[str]
        Avisos sobre tamanho do lote, desbalanceamento e limites do que uma
        única auditoria pode sustentar.
    """
    if not metricas_por_modelo:
        return []

    qualquer = next(iter(metricas_por_modelo.values()))
    n = qualquer['n']
    positivos = qualquer['vp'] + qualquer['fn']
    negativos = qualquer['vn'] + qualquer['fp']

    avisos = []
    if positivos < MINIMO_POR_CLASSE or negativos < MINIMO_POR_CLASSE:
        avisos.append(f"Lote pequeno ({n} pacientes: {positivos} malignos, {negativos} benignos). "
                      f"Um único paciente a mais ou a menos move as métricas em vários pontos "
                      f"percentuais — compare os modelos pelos intervalos de confiança, não "
                      f"pelos valores pontuais.")

    avisos.append("Se este gabarito contém pacientes já vistos no treino, os números estão "
                  "otimistas: eles medem memorização, não generalização. A estimativa honesta "
                  "de desempenho é a validação cruzada repetida do notebook (Seção 10).")
    avisos.append("Todas as métricas assumem o limiar fixo de 50%. Em rastreio, baixá-lo troca "
                  "especificidade por sensibilidade — uma decisão clínica, não estatística.")
    return avisos
