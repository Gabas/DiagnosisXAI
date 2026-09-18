"""
Validação da planilha de entrada, antes de qualquer conta.

Sem isto, uma planilha malformada atravessa o Passo 2 inteira e só estoura lá
dentro do scikit-learn, devolvendo ao usuário uma mensagem em inglês que não
diz onde está o problema — ``Input X contains NaN``, ``could not convert string
to float: 'abc'``, ``The feature names should match those that were passed
during fit``. Nenhuma delas aponta a linha nem a coluna.

Aqui cada problema vira uma mensagem em português que nomeia a coluna e a linha
da planilha (contando o cabeçalho, como o usuário vê no Excel), levantada como
``LoteInvalido`` — que ``BatchProcessor.process`` deixa subir e a interface do
Passo 2 já exibe.

O módulo é puro: recebe DataFrames e a lista de colunas esperadas, sem tocar no
``wisconsin.pkl`` nem no ``ModelLoader``.
"""

import pandas as pd

# Quantas colunas citar antes de resumir o resto em "e mais N".
_MAX_CITADAS = 3


class LoteInvalido(ValueError):
    """
    Planilha recusada na validação, com mensagem pronta para a interface.

    Herda de ``ValueError`` para que qualquer ``except ValueError`` já
    existente continue funcionando.
    """


def _linha_planilha(df: pd.DataFrame, posicao: int) -> int:
    """
    Converte a posição de uma linha na numeração que o usuário vê na planilha.

    A posição 0 é a primeira linha de dados, que no arquivo é a linha 2 — a 1
    é o cabeçalho.

    Parameters
    ----------
    df : pandas.DataFrame
        Lote sendo validado (usado apenas para manter a assinatura explícita).
    posicao : int
        Índice posicional da linha, começando em zero.

    Returns
    -------
    int
        Número da linha no arquivo, começando em 2.
    """
    return int(posicao) + 2


def _listar(nomes) -> str:
    """
    Formata uma lista de nomes de coluna, resumindo quando são muitos.

    Parameters
    ----------
    nomes : sequence of str
        Nomes a citar.

    Returns
    -------
    str
        Os nomes separados por vírgula; acima de ``_MAX_CITADAS``, os
        primeiros seguidos de "e mais N".
    """
    nomes = list(nomes)
    if len(nomes) <= _MAX_CITADAS:
        return ", ".join(nomes)
    restantes = len(nomes) - _MAX_CITADAS
    return f"{', '.join(nomes[:_MAX_CITADAS])} e mais {restantes}"


def validar_colunas(df: pd.DataFrame, esperadas) -> None:
    """
    Confere que a planilha tem linhas e todas as colunas que o modelo exige.

    Colunas a mais não são erro: o ``BatchProcessor`` as descarta, e é comum a
    planilha do laboratório trazer campos administrativos.

    Parameters
    ----------
    df : pandas.DataFrame
        Lote já sem as colunas não preditivas (``id``, ``diagnosis``).
    esperadas : sequence of str
        Nomes das colunas que o modelo espera, na ordem do treino.

    Raises
    ------
    LoteInvalido
        Se a planilha estiver vazia ou faltar alguma coluna.
    """
    if df.shape[0] == 0:
        raise LoteInvalido(
            "A planilha não tem nenhuma linha de dados — só o cabeçalho.")

    faltando = [c for c in esperadas if c not in df.columns]
    if faltando:
        verbo = "Falta" if len(faltando) == 1 else "Faltam"
        raise LoteInvalido(
            f"{verbo} {len(faltando)} das {len(esperadas)} colunas de "
            f"biomarcadores: {_listar(faltando)}. "
            "Confira se o arquivo é o da base Wisconsin (WDBC) e se os nomes "
            "das colunas não foram traduzidos ou renomeados."
        )


def validar_valores(df: pd.DataFrame) -> None:
    """
    Confere que toda célula é um número presente — sem texto e sem vazio.

    Texto é verificado antes de vazio porque uma coluna com texto no meio nem
    chega a ter média, e é da média de ``area_mean`` que o ``BatchProcessor``
    deduz se o lote está em escala bruta ou já padronizado.

    Não julga o *valor* do número: isso é de ``validar_nao_negativos``, que só
    se aplica a lotes em escala bruta.

    Parameters
    ----------
    df : pandas.DataFrame
        Lote com exatamente as colunas de biomarcadores.

    Raises
    ------
    LoteInvalido
        No primeiro tipo de problema encontrado, citando coluna e linha.
    """
    nao_numericas = {}
    for coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            continue
        convertida = pd.to_numeric(df[coluna], errors='coerce')
        # Só é "texto" o que não era vazio antes e virou NaN na conversão.
        ruins = convertida.isna() & df[coluna].notna()
        if ruins.any():
            pos = int(ruins.to_numpy().argmax())
            nao_numericas[coluna] = (_linha_planilha(df, pos), df[coluna].iloc[pos])

    if nao_numericas:
        coluna, (linha, valor) = next(iter(nao_numericas.items()))
        extra = (f" Também há texto em: {_listar(list(nao_numericas)[1:])}."
                 if len(nao_numericas) > 1 else "")
        raise LoteInvalido(
            f"A coluna '{coluna}' tem texto onde era esperado número: "
            f"{valor!r} na linha {linha} da planilha.{extra} "
            "Verifique o separador decimal (o arquivo deve usar ponto) e "
            "células com observações escritas no meio dos números."
        )

    numerico = df.apply(pd.to_numeric, errors='coerce')

    vazias = {c: int(numerico[c].isna().sum()) for c in numerico.columns
              if numerico[c].isna().any()}
    if vazias:
        coluna = next(iter(vazias))
        pos = int(numerico[coluna].isna().to_numpy().argmax())
        total = sum(vazias.values())
        onde = (f"na coluna '{coluna}'" if len(vazias) == 1
                else f"em {len(vazias)} colunas, a primeira '{coluna}'")
        quantas = "1 célula vazia" if total == 1 else f"{total} células vazias"
        raise LoteInvalido(
            f"A planilha tem {quantas} {onde} — a primeira na linha "
            f"{_linha_planilha(df, pos)}. O modelo não decide com medidas "
            "faltando: preencha as células ou remova essas linhas."
        )


def validar_nao_negativos(df: pd.DataFrame) -> None:
    """
    Recusa valores negativos num lote em escala bruta.

    Os 30 biomarcadores do WDBC são medidas de tamanho e forma de núcleos
    celulares: o mínimo na base completa é 0,0 e nenhum deles pode ser
    negativo. Num lote já padronizado (Z-score), porém, metade dos valores é
    negativa por construção — por isso esta checagem é separada de
    ``validar_valores`` e só o ``BatchProcessor`` decide quando aplicá-la,
    depois de detectar a escala.

    Parameters
    ----------
    df : pandas.DataFrame
        Lote em escala bruta, com as colunas de biomarcadores.

    Raises
    ------
    LoteInvalido
        Se alguma coluna tiver valor negativo.
    """
    numerico = df.apply(pd.to_numeric, errors='coerce')
    negativas = {c: int((numerico[c] < 0).sum()) for c in numerico.columns
                 if (numerico[c] < 0).any()}
    if negativas:
        coluna = next(iter(negativas))
        pos = int((numerico[coluna] < 0).to_numpy().argmax())
        raise LoteInvalido(
            f"A coluna '{coluna}' tem valor negativo "
            f"({numerico[coluna].iloc[pos]:g} na linha "
            f"{_linha_planilha(df, pos)} da planilha). Os 30 biomarcadores do "
            "WDBC são medidas de tamanho e forma de núcleos celulares, todas "
            "não negativas — um negativo indica erro de digitação ou de "
            "exportação."
        )
