"""
Glossário dos 30 biomarcadores morfológicos da base Wisconsin (WDBC).

Fonte única de verdade para a explicação de cada atributo — usada tanto pela
janela de detalhes da aba "Sobre" (``views.info_window``) quanto pelos tooltips
dos cabeçalhos da tabela de resultados (``views.predict_view``).

Os 30 atributos derivam de uma imagem digitalizada de Punção Aspirativa por
Agulha Fina (PAAF) de uma massa mamária e descrevem os núcleos das células. São
10 medições-base, cada uma resumida por três estatísticas (média, erro padrão e
"pior"), totalizando 30 colunas (ex.: ``radius_mean``, ``radius_se``,
``radius_worst``).

Nota sobre unidades: a maioria dos atributos NÃO tem unidade física — são
índices adimensionais ou medidas na escala de pixel da imagem, não em
milímetros. Além disso, o app padroniza tudo por Z-score antes de prever.
"""

# Chave-base (como aparece no início do nome da coluna) -> (rótulo, o que mede, unidade).
# A chave 'concave points' tem espaço e 'fractal_dimension' tem underscore, exatamente
# como no cabeçalho do CSV — por isso a separação da coluna é feita pelo sufixo (ver
# separar_coluna), e não por um split ingênuo em '_'.
BIOMARCADORES_BASE = {
    "radius":            ("Raio",             "tamanho do núcleo — distância média do centro à borda",      "comprimento, em pixels da imagem"),
    "texture":           ("Textura",          "variação da tonalidade de cinza (heterogeneidade interna)",  "desvio-padrão de intensidade — adimensional"),
    "perimeter":         ("Perímetro",        "comprimento do contorno do núcleo",                          "comprimento, em pixels"),
    "area":              ("Área",             "área do núcleo",                                             "pixels²"),
    "smoothness":        ("Suavidade",        "variação local do raio (quão liso é o contorno)",            "índice — adimensional"),
    "compactness":       ("Compacidade",      "perímetro² / área − 1 (o quanto a forma é compacta)",        "adimensional"),
    "concavity":         ("Concavidade",      "severidade das reentrâncias (partes côncavas) do contorno",  "adimensional"),
    "concave points":    ("Pontos côncavos",  "número de reentrâncias (partes côncavas) do contorno",       "contagem"),
    "symmetry":          ("Simetria",         "simetria do núcleo",                                        "adimensional"),
    "fractal_dimension": ("Dimensão fractal", "complexidade do contorno (\"aproximação de costa\" − 1)",    "adimensional"),
}

# Sufixo do nome da coluna -> (rótulo, explicação).
ESTATISTICAS = {
    "mean":  ("média",       "média do biomarcador entre os núcleos da imagem"),
    "se":    ("erro padrão", "erro padrão da medida — variabilidade entre os núcleos"),
    "worst": ("pior",        "média dos três maiores valores na imagem (o núcleo \"pior caso\") — "
                             "costuma ser o mais preditivo de malignidade"),
}

# Colunas geradas pelo app (não são biomarcadores de entrada) — descrições para o tooltip.
_COLUNAS_RESULTADO = {
    "Diagnóstico_IA":     "Classe prevista pelo modelo selecionado: Maligno ou Benigno.",
    "Certeza_Maligno(%)": "Probabilidade calibrada de o tumor ser Maligno (0–100%). Valores próximos "
                          "de 50% indicam um caso limítrofe, que merece revisão.",
    "Perfil":             "Indica se o paciente está dentro da distribuição de treino. \"Atípico\" = "
                          "perfil muito distante dos casos vistos no aprendizado (distância de "
                          "Mahalanobis além do limiar) — a previsão é menos confiável e merece cautela.",
    "Decisão":            "\"Limítrofe\" quando a certeza calibrada está perto de 50% (a até 10 pontos) — "
                          "a decisão é incerta e um pequeno deslocamento inverteria o diagnóstico, "
                          "então recomenda-se revisão. \"Definida\" caso contrário.",
}


def separar_coluna(nome):
    """
    Separa o nome de uma coluna em (base, sufixo).

    Ex.: ``'radius_worst'`` -> ``('radius', 'worst')``; ``'concave points_mean'``
    -> ``('concave points', 'mean')``. Retorna ``(nome, None)`` quando não termina
    num sufixo estatístico conhecido (ou quando ``nome`` não é uma string).
    """
    if not isinstance(nome, str):
        return nome, None
    for sufixo in ESTATISTICAS:
        if nome.endswith("_" + sufixo):
            return nome[: -(len(sufixo) + 1)], sufixo
    return nome, None


def rotulo_amigavel(nome):
    """
    Rótulo legível de uma coluna de biomarcador, ex.: ``'Raio (pior)'``.

    Retorna ``None`` se ``nome`` não for um dos 30 biomarcadores.
    """
    base, sufixo = separar_coluna(nome)
    if base in BIOMARCADORES_BASE and sufixo is not None:
        return f"{BIOMARCADORES_BASE[base][0]} ({ESTATISTICAS[sufixo][0]})"
    return None


def descricao_coluna(nome):
    """
    Texto explicativo curto de uma coluna da tabela de resultados (para tooltip).

    Cobre os 30 biomarcadores de entrada, as colunas de diagnóstico geradas pelo
    app (``Diagnóstico_IA``, ``Certeza_Maligno(%)``) e as colunas de comparação
    ``IA_XXX``. Retorna ``None`` quando não há descrição conhecida — nesse caso o
    tooltip simplesmente não é exibido.
    """
    if nome in _COLUNAS_RESULTADO:
        return _COLUNAS_RESULTADO[nome]
    if isinstance(nome, str) and nome.startswith("IA_"):
        return f"Diagnóstico do modelo {nome[3:]} (modo de comparação entre todas as IAs)."

    base, sufixo = separar_coluna(nome)
    if base in BIOMARCADORES_BASE and sufixo is not None:
        _, o_que_mede, unidade = BIOMARCADORES_BASE[base]
        _, stat_expl = ESTATISTICAS[sufixo]
        return (f"{rotulo_amigavel(nome)}\n"
                f"{o_que_mede}.\n"
                f"Estatística — {stat_expl}.\n"
                f"Unidade: {unidade}.")
    return None


# Textos descritivos do glossário, exibidos no card da aba "Sobre"
# (views.about_view). A tabela dos 10 biomarcadores e das 3 estatísticas é
# montada lá diretamente a partir de BIOMARCADORES_BASE e ESTATISTICAS.
GLOSSARIO_SUBTITULO = "Os 30 atributos morfológicos da base Wisconsin (WDBC)"

GLOSSARIO_INTRO = (
    "Os 30 atributos derivam de uma imagem digitalizada de Punção Aspirativa por Agulha Fina "
    "(PAAF) de uma massa mamária e descrevem os núcleos das células. São 10 medições-base, cada "
    "uma resumida por 3 estatísticas (média, erro padrão e \"pior\"), totalizando 30 colunas — "
    "ex.: radius_mean, radius_se, radius_worst."
)

GLOSSARIO_UNIDADES = (
    "A maioria dos atributos NÃO tem unidade física: são índices adimensionais ou medidas na "
    "escala de pixel da imagem — não em milímetros. Apenas raio e perímetro (comprimento), área "
    "(pixels²) e pontos côncavos (contagem) têm noção de unidade, ainda assim relativa à imagem. "
    "Além disso, o app padroniza tudo por Z-score (μ=0, σ=1) antes de prever, então os modelos "
    "enxergam valores padronizados, não os brutos."
)
