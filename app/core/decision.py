"""
Política de decisão do DiagnosisXAI: limiar de operação e zona de revisão.

Este módulo é a **fonte única** de três respostas que antes estavam espalhadas
(e divergiam) pelo aplicativo:

1. *Qual classe o paciente recebe?* — Antes vinha de ``modelo.predict()``, que
   usa implicitamente o limiar de 0,5 sobre a probabilidade NÃO calibrada,
   enquanto a coluna de certeza exibida vinha do modelo CALIBRADO. As duas
   podiam se contradizer na mesma linha da tabela (um paciente rotulado
   "Benigno" com 61% de certeza de malignidade). Aqui a classe passa a sair da
   mesma probabilidade calibrada que o usuário vê.

2. *Onde fica o corte?* — Não em 0,5. Em rastreio oncológico o falso negativo
   (câncer não detectado) custa muito mais que o falso positivo (exame extra),
   e o corte que equilibra os dois erros não é o que minimiza o erro total. O
   limiar de cada modelo é escolhido fora deste módulo, por validação cruzada
   no conjunto de treino (ver ``scripts/calibrar_limiares.py``), e lido de
   ``data/limiares.json``. Sem esse arquivo, tudo recai em 0,5 e o app se
   comporta como antes.

3. *Quando a decisão é limítrofe?* — Antes havia quatro definições simultâneas:
   ±10 pontos em torno de 50% no ``PredictorEngine``, e as faixas próprias de
   cada explicador ((0,30–0,70) na Regressão Logística, (0,35–0,65) no Random
   Forest e no SVM, confiança < 0,65 no KNN), todas sobre a probabilidade não
   calibrada. O mesmo paciente aparecia como "Definida" na tabela e "limítrofe,
   recomenda-se revisão" no relatório. Agora existe uma faixa só, medida em
   torno do limiar de operação — que é onde a decisão de fato pode virar.

Sobre calibrar e explicar ao mesmo tempo: o escalonamento de Platt é monotônico,
então decidir por ``p_calibrada ≥ τ`` equivale a decidir por ``score_bruto ≥ s``
para algum ``s``. As explicações (contribuições da regressão, vetores de
suporte, votos das árvores) decompõem o score bruto e continuam válidas — a
calibração só troca a régua em que o número é lido. É por isso que
:func:`aplicar_a_explicacoes` pode sobrescrever a classe e a probabilidade
exibidas sem invalidar o raciocínio que as acompanha.
"""

import json
import os

ROTULO_MALIGNO = 'Maligno'
ROTULO_BENIGNO = 'Benigno'

# Terceira saída possível: o sistema se recusa a decidir e devolve o caso ao
# médico. Não é um diagnóstico intermediário ("talvez maligno") — é a ausência
# deliberada de diagnóstico, para os casos em que decidir seria adivinhar.
ROTULO_REVISAR = 'Revisar'

# Limiar usado quando não há um limiar calibrado para o modelo — o comportamento
# padrão de ``predict()`` do scikit-learn.
LIMIAR_NEUTRO = 0.5

# Meia-largura (em probabilidade) da faixa de revisão em torno do limiar de
# operação. Uma decisão dentro dela inverteria com um deslocamento pequeno da
# probabilidade, então é sinalizada para revisão humana em vez de ser entregue
# como definitiva.
BANDA_PADRAO = 0.10

ZONA_LIMITROFE = 'Limítrofe'
ZONA_DEFINIDA = 'Definida'
ZONA_REVISAO = ROTULO_REVISAR

# Nome do comitê no seletor de modelos. Vive aqui (e não no PredictorEngine)
# porque tanto a política quanto o motor de inferência precisam dele.
NOME_COMITE = 'Comitê (voto suave)'

_ARQUIVO_PADRAO = 'limiares.json'


def limitrofe_padrao(prob: float) -> bool:
    """
    Marcação de limítrofe sem política carregada: faixa em torno de 0,5.

    É o que os explicadores usam quando rodam fora do aplicativo (no notebook),
    onde não há limiar de operação. Existe para que a *definição* de limítrofe
    seja uma só no projeto inteiro — antes cada explicador tinha a sua
    ((0,30–0,70) na Regressão Logística, (0,35–0,65) no Random Forest e no SVM,
    confiança < 0,65 no KNN). Dentro do app, esta marcação é substituída pela
    faixa em torno do limiar de operação (ver :func:`aplicar_a_explicacoes`).

    Parameters
    ----------
    prob : float
        P(Maligno), entre 0 e 1.
    """
    return abs(float(prob) - LIMIAR_NEUTRO) <= BANDA_PADRAO


class PoliticaDecisao:
    """
    Regra que transforma uma probabilidade calibrada em decisão e zona.

    Parameters
    ----------
    limiares : dict[str, float] ou None
        Limiar de operação por modelo, em probabilidade (0–1). Modelos ausentes
        usam ``LIMIAR_NEUTRO``. ``None`` equivale a dicionário vazio.
    banda : float, optional
        Meia-largura da faixa de revisão em torno do limiar (padrão 0,10).
    metadados : dict ou None, optional
        Procedência dos limiares (critério, data de geração, desempenho medido)
        — exibida na interface para que a decisão seja auditável.
    faixas_revisao : dict ou None, optional
        ``{modelo: (inferior, superior)}`` — faixa de probabilidade em que o
        modelo se abstém, quando ``adiar_incertos`` está ligado.

    Attributes
    ----------
    calibrada : bool
        True quando há pelo menos um limiar diferente do neutro, isto é,
        quando ``data/limiares.json`` foi carregado.
    adiar_incertos : bool
        Liga a recusa: em vez de um rótulo, os casos dentro da faixa recebem
        ``'Revisar'``. Começa desligado — trata-se de uma decisão de operação
        (aceitar devolver parte dos casos ao médico), não de um detalhe técnico,
        e portanto cabe a quem opera, não ao padrão do programa.
    """

    def __init__(self, limiares: dict = None, banda: float = BANDA_PADRAO,
                 metadados: dict = None, faixas_revisao: dict = None):
        """Guarda os limiares por modelo, a faixa de revisão e a de recusa."""
        self._limiares = {k: float(v) for k, v in (limiares or {}).items()}
        self._faixas = {k: (float(v[0]), float(v[1]))
                        for k, v in (faixas_revisao or {}).items() if v}
        self.banda = float(banda)
        self.metadados = dict(metadados or {})
        self.adiar_incertos = False

    @property
    def calibrada(self) -> bool:
        """True se algum modelo opera com limiar diferente de 0,5."""
        return any(abs(v - LIMIAR_NEUTRO) > 1e-9 for v in self._limiares.values())

    @classmethod
    def carregar(cls, caminho: str = None):
        """
        Carrega a política de ``data/limiares.json``, com recuo seguro.

        Parameters
        ----------
        caminho : str ou None, optional
            Caminho do arquivo. Quando None, procura ``data/limiares.json`` na
            raiz do repositório.

        Returns
        -------
        PoliticaDecisao
            A política do arquivo; ou a política neutra (limiar 0,5 para todos)
            se o arquivo não existir ou estiver ilegível — o app continua
            funcionando exatamente como antes de haver limiares calibrados.
        """
        if caminho is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            caminho = os.path.join(base_dir, 'data', _ARQUIVO_PADRAO)

        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
        except (OSError, ValueError):
            return cls()

        return cls(
            limiares=dados.get('limiares'),
            banda=dados.get('banda_revisao', BANDA_PADRAO),
            faixas_revisao=dados.get('faixas_recusa'),
            metadados={k: v for k, v in dados.items()
                       if k not in ('limiares', 'banda_revisao', 'faixas_recusa')},
        )

    def limiar(self, modelo: str) -> float:
        """Limiar de operação do modelo, em probabilidade (0–1)."""
        return self._limiares.get(modelo, LIMIAR_NEUTRO)

    def faixa_recusa(self, modelo: str):
        """
        Faixa de probabilidade em que o modelo se abstém, ou None.

        Returns
        -------
        tuple[float, float] ou None
            ``(inferior, superior)``: abaixo do inferior o caso é decidido como
            Benigno; do superior para cima, como Maligno; entre os dois, o
            sistema não decide. ``None`` quando o modelo não tem faixa
            calibrada — nesse caso a recusa não se aplica a ele.
        """
        return self._faixas.get(modelo)

    def pode_adiar(self, modelo: str) -> bool:
        """True se o modelo tem faixa de recusa calibrada."""
        return self.faixa_recusa(modelo) is not None

    def _adiando(self, modelo: str):
        """Faixa em vigor agora (None quando a recusa está desligada)."""
        return self.faixa_recusa(modelo) if self.adiar_incertos else None

    def rotular(self, prob: float, modelo: str) -> str:
        """
        Classe atribuída a uma probabilidade calibrada de malignidade.

        Com a recusa ligada, a decisão deixa de ser um corte único: abaixo da
        faixa é Benigno, acima é Maligno, e dentro dela o sistema devolve
        ``'Revisar'`` em vez de arriscar um palpite.

        Parameters
        ----------
        prob : float
            P(Maligno) calibrada, entre 0 e 1.
        modelo : str
            Nome do modelo, para resolver o limiar.
        """
        faixa = self._adiando(modelo)
        if faixa is not None:
            inferior, superior = faixa
            if prob < inferior:
                return ROTULO_BENIGNO
            if prob >= superior:
                return ROTULO_MALIGNO
            return ROTULO_REVISAR

        return ROTULO_MALIGNO if prob >= self.limiar(modelo) else ROTULO_BENIGNO

    def zona(self, prob: float, modelo: str) -> str:
        """
        Classifica a decisão como ``'Revisar'``, ``'Limítrofe'`` ou ``'Definida'``.

        Limítrofe significa que a probabilidade está a menos de ``banda`` do
        limiar de operação — perto o bastante para que a decisão vire com uma
        variação pequena, e portanto merecedora de revisão humana. Revisar é o
        caso mais forte: nem sequer houve decisão a rever.
        """
        if self.rotular(prob, modelo) == ROTULO_REVISAR:
            return ZONA_REVISAO
        return ZONA_LIMITROFE if abs(prob - self.limiar(modelo)) <= self.banda else ZONA_DEFINIDA

    def resumo(self, modelo: str) -> str:
        """Frase curta descrevendo o ponto de operação, para exibir na interface."""
        faixa = self._adiando(modelo)
        if faixa is not None:
            cobertura = (self.metadados.get('cobertura_treino') or {}).get(modelo)
            medido = (f" No treino, decidiu {cobertura:.0f}% dos casos sem errar nenhum."
                      if isinstance(cobertura, (int, float)) else "")
            return (f"Recusa ligada: decide Benigno abaixo de {faixa[0] * 100:.1f}% e Maligno "
                    f"a partir de {faixa[1] * 100:.1f}%; entre os dois, devolve o caso para "
                    f"revisão em vez de arriscar.{medido}")

        limiar = self.limiar(modelo)
        if abs(limiar - LIMIAR_NEUTRO) <= 1e-9:
            return "Limiar de decisão: 50% (padrão, sem calibração de operação)."

        piso = self.metadados.get('piso_especificidade')
        criterio = (f" — o mais sensível que mantém especificidade ≥ {piso:.0f}% no treino"
                    if isinstance(piso, (int, float)) else "")

        treino = (self.metadados.get('desempenho_treino') or {}).get(modelo) or {}
        medido = ""
        if 'sensibilidade' in treino and 'especificidade' in treino:
            medido = (f" No treino: sensibilidade {treino['sensibilidade']:.1f}%, "
                      f"especificidade {treino['especificidade']:.1f}%.")

        return (f"Limiar de decisão: {limiar * 100:.1f}%{criterio}.{medido} "
                f"Casos entre {max(0.0, limiar - self.banda) * 100:.1f}% e "
                f"{min(1.0, limiar + self.banda) * 100:.1f}% são marcados para revisão.")


def aplicar_a_explicacoes(explicacoes: list, probabilidades, politica: PoliticaDecisao,
                          modelo: str) -> list:
    """
    Alinha as explicações de um modelo à decisão efetivamente exibida ao usuário.

    Os explicadores calculam a classe e a probabilidade a partir do modelo
    original (limiar de 0,5, probabilidade não calibrada). A tabela do Passo 3
    usa a probabilidade calibrada e o limiar de operação. Sem esta função as
    duas telas discordam — e, com limiar ajustado, discordariam sistematicamente
    em toda a faixa entre o limiar e 0,5.

    Sobrescreve apenas o que é *decisão* (classe, probabilidade exibida,
    confiança e marcação de limítrofe); o conteúdo explicativo de cada relatório
    (contribuições, vizinhos, votos das árvores, vetores de suporte) permanece
    intocado, pois descreve o score do modelo — que a calibração apenas reescala
    de forma monotônica.

    Parameters
    ----------
    explicacoes : list[dict]
        Saída de ``explain()`` de um explicador, na ordem do lote.
    probabilidades : sequence of float
        P(Maligno) calibrada de cada paciente, entre 0 e 1, na mesma ordem.
    politica : PoliticaDecisao
        Política em vigor.
    modelo : str
        Nome do modelo, para resolver o limiar.

    Returns
    -------
    list[dict]
        A mesma lista recebida, com os campos de decisão atualizados. Se o
        número de probabilidades não bater com o de explicações, a lista volta
        inalterada (situação impossível no fluxo normal, mas que não deve
        derrubar o relatório).
    """
    if probabilidades is None or len(probabilidades) != len(explicacoes):
        return explicacoes

    for explicacao, prob in zip(explicacoes, probabilidades):
        p = float(prob)
        classe = politica.rotular(p, modelo)
        if classe == ROTULO_REVISAR:
            # Sem classe decidida não há "confiança na decisão"; exibe-se a
            # confiança no lado para o qual o modelo pendeu — insuficiente,
            # que é justamente o motivo de o caso ter sido devolvido.
            confianca = max(p, 1.0 - p)
        else:
            confianca = p if classe == ROTULO_MALIGNO else 1.0 - p

        explicacao['classe'] = classe
        explicacao['limitrofe'] = politica.zona(p, modelo) == ZONA_LIMITROFE
        # Cada explicador nomeia seus campos de forma diferente ('probabilidade'
        # é P(Maligno); 'certeza'/'confianca' são a probabilidade da classe
        # predita). Atualiza-se o que existir, sem inventar campos novos.
        if 'probabilidade' in explicacao:
            explicacao['probabilidade'] = round(p * 100, 1)
        if 'confianca' in explicacao:
            explicacao['confianca'] = round(confianca * 100, 1)
        if 'certeza' in explicacao:
            explicacao['certeza'] = round(confianca * 100, 1)

    return explicacoes
