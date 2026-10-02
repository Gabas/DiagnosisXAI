"""
Os fatores que mais pesaram na decisão de um paciente — SHAP exato por fator.

Responde, para qualquer modelo do app (inclusive o comitê), à pergunta que o
médico faz primeiro: *que características do exame levaram o modelo a esta
certeza?* Quatro escolhas de método, todas medidas no lote de teste antes de
virarem código:

1. **Por fator, e não por coluna.** Raio, perímetro e área têm correlação de
   0,98 a 0,998 no treino. Um "top 6" das 30 colunas gastava em média 3 das 6
   posições repetindo "o núcleo é grande" e cobria só 62–69% da explicação.
   Além disso, com colunas quase idênticas a divisão do crédito entre elas é
   arbitrária. Aqui as 30 colunas viram 8 fatores — as 10 medições-base do
   WDBC, com raio, perímetro e área fundidos em "tamanho" — e cada fator leva
   junto as suas três estatísticas (média, erro padrão e pior).

2. **Shapley exato, não amostrado.** Com 8 fatores como "jogadores" há só
   2⁸ = 256 coalizões, então o valor de Shapley de cada fator é calculado por
   enumeração completa, sem a amostragem do KernelExplainer. Dentro de um fator
   as colunas andam sempre juntas (nunca se combina o raio de um paciente com a
   área de outro), o que elimina as combinações impossíveis entre colunas quase
   colineares.

3. **Sobre a certeza calibrada.** A função explicada é a mesma probabilidade
   calibrada que a tabela do Passo 3 exibe. Vale, até o arredondamento de
   máquina: ``base + Σ fatores = certeza``. (O SHAP do modelo bruto chegava a
   20 pontos de diferença da tabela.)

4. **Fundo = o treino inteiro.** A "base" é a certeza média de um paciente do
   treino — 37,5%, a prevalência de malignos, como se espera de uma
   probabilidade calibrada. Os 25 centroides usados pelo SHAP da janela
   dedicada davam 52%, e subamostras aleatórias de 60–150 pacientes erravam os
   fatores em até 6 pontos.

O comitê não precisa de cálculo próprio: o Shapley é linear na função
explicada, e a certeza do comitê é a média das certezas dos membros — logo os
fatores do comitê são, exatamente, a média dos fatores dos membros.

Custo medido por paciente: Regressão Logística ~0,1 s; Random Forest, KNN e
SVM entre 1,5 e 2,2 s; comitê ~5 s (os quatro membros). Por isso o cálculo é
sob demanda e com cache (``FatoresDoLote``), e as janelas o fazem em segundo
plano.

O que isto NÃO é: uma relação de causa. Um fator que "puxa para Maligno"
descreve o que o modelo usou para chegar à certeza, não a biologia do tumor.
"""

import itertools
import math
import queue
import textwrap
import threading
from concurrent.futures import Future

import numpy as np

from core.biomarkers import BIOMARCADORES_BASE, separar_coluna
from core.decision import NOME_COMITE, ROTULO_BENIGNO, ROTULO_MALIGNO

# Raio, perímetro e área: correlação de 0,98 a 0,998 no treino — um fator só.
_FUNDIDOS = {'radius': 'tamanho', 'perimeter': 'tamanho', 'area': 'tamanho'}

ROTULO_FATOR = {'tamanho': 'Tamanho do núcleo'}
ROTULO_FATOR.update({base: rotulo for base, (rotulo, _, _) in BIOMARCADORES_BASE.items()
                     if base not in _FUNDIDOS})

# Quantos fatores listar por paciente (pedido do orientador); os demais vão
# somados numa linha, para que a conta continue fechando.
TOP_FATORES = 6

# Mesma ordem e mesma regra do PredictorEngine — a certeza do comitê é a média
# destes membros, então os fatores dele também têm de ser.
MEMBROS_COMITE = ('Regressão Logística', 'Random Forest', 'SVM', 'KNN')
MODELO_SEM_CALIBRACAO = 'Árvore de Decisão'


def agrupar_colunas(feature_names) -> list:
    """
    Agrupa as 30 colunas do WDBC nos 8 fatores, na ordem em que aparecem.

    Parameters
    ----------
    feature_names : sequence of str
        Nomes das colunas na ordem que o modelo recebe.

    Returns
    -------
    list[tuple[str, numpy.ndarray, numpy.ndarray]]
        ``(fator, índices de todas as colunas do fator, índices das colunas
        "_mean" do fator)`` — as últimas só servem para dizer ao médico onde
        o paciente está em relação à média.

    Raises
    ------
    ValueError
        Se alguma coluna não for um dos 30 biomarcadores.
    """
    fator_de, media = [], []
    for nome in feature_names:
        base, sufixo = separar_coluna(nome)
        if base not in BIOMARCADORES_BASE or sufixo is None:
            raise ValueError(f"'{nome}' não é um dos 30 biomarcadores do WDBC.")
        fator_de.append(_FUNDIDOS.get(base, base))
        media.append(sufixo == 'mean')

    grupos = []
    for fator in dict.fromkeys(fator_de):
        idx = np.array([j for j, f in enumerate(fator_de) if f == fator])
        grupos.append((fator, idx, np.array([j for j in idx if media[j]])))
    return grupos


def _matriz_shapley(n_jogadores: int) -> tuple:
    """
    Coalizões e a matriz que transforma os valores delas nos valores de Shapley.

    Com ``v`` o valor de cada coalizão, ``φ = M @ v``, porque
    ``φ_g = Σ_{S ∌ g} w(|S|) · [v(S ∪ {g}) − v(S)]`` com
    ``w(s) = s! (n − s − 1)! / n!``: cada coalizão T entra com ``+w(|T|−1)``
    para os jogadores que contém e com ``−w(|T|)`` para os que não contém.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(coalizoes, M)`` — ``coalizoes`` é booleana (2ⁿ × n) e começa pela
        coalizão vazia e termina pela completa; ``M`` é (n × 2ⁿ).
    """
    n = n_jogadores
    coalizoes = np.array(list(itertools.product([False, True], repeat=n)))
    w = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    tamanho = coalizoes.sum(axis=1)
    M = np.where(coalizoes.T,
                 np.array([w[t - 1] if t > 0 else 0.0 for t in tamanho])[None, :],
                 -np.array([w[t] if t < n else 0.0 for t in tamanho])[None, :])
    return coalizoes, M


class ShapleyPorFator:
    """
    Valores de Shapley exatos, por fator, de uma função de probabilidade.

    A função de valor é a intervencional: ``v(S) = média, sobre o fundo, de
    f(x com os fatores de S vindos do paciente e o resto vindo do fundo)``.
    Assim ``v(∅)`` é a certeza média do fundo e ``v(todos) = f(x)``.

    Parameters
    ----------
    funcao : callable
        Recebe uma matriz (n × 30) na escala que o modelo espera e devolve
        P(Maligno) para cada linha.
    fundo : array-like
        Pacientes de referência (n_fundo × 30), na mesma escala.
    feature_names : sequence of str
        Nomes das colunas, na ordem de ``funcao``.
    """

    def __init__(self, funcao, fundo, feature_names):
        self._f = funcao
        self._fundo = np.asarray(fundo, dtype=float)
        self._grupos = agrupar_colunas(feature_names)
        coalizoes, self._M = _matriz_shapley(len(self._grupos))

        n_col = len(feature_names)
        self._mascara = np.zeros((len(coalizoes), n_col), dtype=bool)
        for g, (_, idx, _) in enumerate(self._grupos):
            self._mascara[:, idx] = coalizoes[:, [g]]

    @property
    def fatores(self) -> list:
        """Nomes dos fatores, na ordem dos grupos."""
        return [fator for fator, _, _ in self._grupos]

    def explicar(self, x, z=None) -> dict:
        """
        Decompõe a certeza de um paciente entre os 8 fatores.

        Parameters
        ----------
        x : array-like
            O paciente (30 valores), na escala de ``funcao``.
        z : array-like ou None
            O mesmo paciente em Z-score, só para exibição (posição de cada
            fator em relação à média do treino). Se None, usa ``x``.

        Returns
        -------
        dict
            ``{'base', 'prob', 'fatores'}``; ``fatores`` vem ordenado pelo
            impacto, cada item ``{'fator', 'rotulo', 'contribuicao', 'z'}``.
            Vale ``base + Σ contribuicao = prob``.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        z = x if z is None else np.asarray(z, dtype=float).reshape(-1)

        linhas = np.where(self._mascara[:, None, :], x[None, None, :], self._fundo[None, :, :])
        valores = np.asarray(self._f(linhas.reshape(-1, x.size)), dtype=float)
        v = valores.reshape(len(self._mascara), len(self._fundo)).mean(axis=1)
        phi = self._M @ v

        fatores = [
            {
                'fator': fator,
                'rotulo': ROTULO_FATOR[fator],
                'contribuicao': float(phi[g]),
                'z': float(np.mean(z[media])) if len(media) else None,
            }
            for g, (fator, _, media) in enumerate(self._grupos)
        ]
        fatores.sort(key=lambda item: abs(item['contribuicao']), reverse=True)
        return {'base': float(v[0]), 'prob': float(v[-1]), 'fatores': fatores}


def combinar(resultados: list) -> dict:
    """
    Fatores de uma média de modelos, a partir dos fatores de cada um.

    Exato, não aproximado: o valor de Shapley é linear na função explicada, e
    o comitê explica a média das certezas dos membros. Os fatores do comitê
    são, portanto, a média dos fatores dos membros — e a base e a certeza dele
    também.

    Parameters
    ----------
    resultados : list[dict]
        Saídas de ``ShapleyPorFator.explicar`` do mesmo paciente, uma por
        membro.

    Returns
    -------
    dict
        No mesmo formato de ``explicar``.
    """
    if not resultados:
        raise ValueError("combinar() precisa de ao menos um resultado.")
    soma = {}
    for r in resultados:
        for item in r['fatores']:
            soma[item['fator']] = soma.get(item['fator'], 0.0) + item['contribuicao']
    z = {item['fator']: item['z'] for item in resultados[0]['fatores']}
    n = len(resultados)
    fatores = [{'fator': f, 'rotulo': ROTULO_FATOR[f], 'contribuicao': s / n, 'z': z.get(f)}
               for f, s in soma.items()]
    fatores.sort(key=lambda item: abs(item['contribuicao']), reverse=True)
    return {
        'base': float(np.mean([r['base'] for r in resultados])),
        'prob': float(np.mean([r['prob'] for r in resultados])),
        'fatores': fatores,
    }


# Largura máxima de uma linha do bloco. As caixas de detalhe das janelas
# comportam de 47 a 53 caracteres em Courier (medido em telas de 768p e 1080p);
# acima disso a linha quebra e as colunas da cascata se desalinham.
LARGURA_BLOCO = 47
_ROTULO = 17          # "Tamanho do núcleo", o rótulo mais longo


def _pp(valor: float) -> float:
    """Arredonda para exibição sem o "-0.0" que o arredondamento produz."""
    return round(valor, 1) + 0.0


def _paragrafo(texto: str, recuo: str = " ") -> list:
    """Quebra um texto corrido na largura do bloco, com recuo."""
    return textwrap.wrap(texto, width=LARGURA_BLOCO, initial_indent=recuo,
                         subsequent_indent=recuo)


def linhas_fatores(resultado: dict, modelo: str, rotulo: str = None,
                   regra: str = None, corte: float = None,
                   top: int = TOP_FATORES) -> list:
    """
    Texto do bloco "Fatores que mais pesaram", para o detalhe e o PDF.

    O bloco é uma cascata: parte da certeza média do treino, soma o efeito de
    cada fator e chega à certeza do paciente. Cada linha de fator diz o efeito
    (pontos percentuais de certeza), para que lado empurrou e onde o paciente
    está naquele fator (desvios-padrão em relação à média do treino). Cabe em
    ``LARGURA_BLOCO`` colunas e só usa caracteres que o PDF (Courier básica)
    imprime — as setas viram ``->`` lá.

    Parameters
    ----------
    resultado : dict
        Saída de ``ShapleyPorFator.explicar`` ou de ``combinar``.
    modelo : str
        Nome do modelo — a Árvore muda o vocabulário (ela não tem certeza
        calibrada, só a classe da folha).
    rotulo : str ou None
        Diagnóstico exibido para o paciente. Com ele, o bloco avisa quando o
        sentido dos fatores e o diagnóstico parecem se contradizer.
    regra : str ou None
        A régua do modelo em uma linha (``PoliticaDecisao.regra``), quando se
        sabe qual política valia no lote.
    corte : float ou None
        O corte (0–1) que de fato decidiu este diagnóstico — ver
        ``FatoresDoLote.corte``. Se informado, o aviso de contradição o cita:
        a régua sozinha pode confundir, porque a faixa "Limítrofe" decide pelo
        limiar e não pelos extremos que ela exibe.
    top : int, optional
        Quantos fatores listar individualmente.

    Returns
    -------
    list[str]
        Linhas de texto, sem a quebra final.
    """
    arvore = modelo == MODELO_SEM_CALIBRACAO
    base, prob = resultado['base'], resultado['prob']
    # Só se lista o fator cujo efeito aparece no arredondamento: na árvore,
    # por exemplo, os fatores fora do caminho têm efeito zero, e listá-los com
    # "+0.0 pp → Benigno" afirmaria um sentido que não existe.
    visiveis = [item for item in resultado['fatores'] if abs(item['contribuicao']) >= 0.0005]
    principais = visiveis[:top]
    demais = [item for item in resultado['fatores'] if item not in principais]

    # Rótulo, efeito e sentido em colunas fixas: os números em pp e em % caem
    # na mesma coluna, e a cascata se lê de cima para baixo.
    def linha_valor(rotulo_linha, valor_pct):
        return f" {rotulo_linha:<{_ROTULO}} {valor_pct:5.1f}%"

    alvo = "na árvore" if arvore else "na certeza"
    linhas = [f"Fatores que mais pesaram {alvo} (SHAP):",
              linha_valor("Média do treino", base * 100)]
    for item in principais:
        c = _pp(item['contribuicao'] * 100)
        sentido = "→ Maligno" if c > 0 else "→ Benigno"
        z = "" if item['z'] is None else f"  {_pp(item['z']):+.1f} dp"
        linhas.append(f" {item['rotulo']:<{_ROTULO}} {c:+5.1f} pp  {sentido}{z}")
    if demais:
        resto = _pp(sum(item['contribuicao'] for item in demais) * 100)
        linhas.append(f" {f'Demais {len(demais)} fatores':<{_ROTULO}} {resto:+5.1f} pp")
    linhas.append(" " + "-" * (_ROTULO + 9))
    linhas.append(linha_valor("Saída (folha)" if arvore else "Este paciente", prob * 100))

    linhas += _paragrafo(
        f"pp: efeito na {'saída' if arvore else 'certeza'}. dp: o paciente naquele fator, "
        "em desvios-padrão da média do treino. Tamanho reúne raio, perímetro e área; "
        "cada fator soma média, erro padrão e pior. Mostra o que o modelo usou, não a "
        "causa do tumor.")

    # Armadilha do valor-base: a soma dos fatores puxa para um lado, mas o que
    # decide é o corte da régua, não a média do treino. Sem este aviso, um
    # "Maligno" com quase todos os fatores apontando para Benigno parece erro.
    if rotulo == ROTULO_MALIGNO and prob < base:
        valor = f" ({corte * 100:.1f}%)" if corte is not None and prob >= corte else ""
        linhas += _paragrafo(
            f"Atenção: somados, os fatores puxaram para baixo da média do treino "
            f"({base * 100:.1f}% → {prob * 100:.1f}%). O diagnóstico é Maligno porque "
            f"{prob * 100:.1f}% ainda está acima do corte deste modelo{valor}, que é baixo "
            f"de propósito: quem decide é o corte, não a média.")
    elif rotulo == ROTULO_BENIGNO and prob > base:
        valor = f" ({corte * 100:.1f}%)" if corte is not None and prob < corte else ""
        linhas += _paragrafo(
            f"Atenção: somados, os fatores puxaram para cima da média do treino "
            f"({base * 100:.1f}% → {prob * 100:.1f}%), mas a certeza ainda ficou abaixo "
            f"do corte deste modelo{valor}: quem decide é o corte, não a média.")

    if regra and not arvore:
        # Uma faixa por linha: a régua inteira numa linha só passa de 80 colunas.
        linhas.append(" Régua do modelo:")
        linhas += [f"   {faixa.strip()}" for faixa in regra.split("·")]
    return linhas


class FatoresDoLote:
    """
    Fatores de cada paciente de um lote, por modelo, sob demanda e com cache.

    É o que as janelas de relatório consultam. Constrói o explicador de cada
    modelo só quando é pedido pela primeira vez, guarda cada resultado e é
    seguro para ser chamado de uma thread de fundo (o cálculo leva segundos).

    Parameters
    ----------
    loader : ModelLoader
        Fonte dos modelos (calibrados), do treino padronizado e do scaler.
    X_scaled, X_raw : array-like
        O lote padronizado e bruto (n × 30), na ordem de ``indices``.
    indices : sequence of int
        Rótulo de cada linha do lote — o mesmo ``indice`` das explicações.
    politica : PoliticaDecisao ou None
        A política que valia quando o lote foi processado (cópia, para que
        ligar ou desligar a recusa depois não mude a régua exibida). Sem ela,
        o bloco sai sem a linha da régua.
    """

    def __init__(self, loader, X_scaled, X_raw, indices, politica=None):
        self._loader = loader
        self._X = np.asarray(X_scaled, dtype=float)
        self._X_raw = np.asarray(X_raw, dtype=float)
        self._pos = {int(i): p for p, i in enumerate(indices)}
        self.politica = politica
        self._explicadores = {}
        self._cache = {}
        self._trava = threading.Lock()
        self._futuros = {}
        self._fila = None

    def _funcao(self, modelo: str):
        """P(Maligno) do modelo como o app a usa: calibrada quando existe."""
        m = (self._loader.calibrated_models.get(modelo)
             or self._loader.models.get(modelo))
        if m is None or not hasattr(m, 'predict_proba'):
            return None
        classes = list(getattr(m, 'classes_', [0, 1]))
        idx = classes.index(1) if 1 in classes else -1
        return lambda X: m.predict_proba(X)[:, idx]

    def membros(self) -> list:
        """Membros do comitê disponíveis no .pkl, na ordem do PredictorEngine."""
        return [n for n in MEMBROS_COMITE if self._loader.calibrated_models.get(n) is not None]

    def disponivel(self, modelo: str) -> bool:
        """True se dá para calcular os fatores deste modelo."""
        if self._loader.X_train_scaled is None:
            return False
        if modelo == NOME_COMITE:
            return len(self.membros()) >= 2
        return self._funcao(modelo) is not None

    def _explicador(self, modelo: str) -> ShapleyPorFator:
        """Explicador do modelo, construído na primeira vez em que é pedido."""
        with self._trava:
            if modelo not in self._explicadores:
                fundo = np.asarray(self._loader.X_train_scaled, dtype=float)
                if modelo == MODELO_SEM_CALIBRACAO:
                    # A árvore foi treinada em dados brutos: o fundo também.
                    fundo = self._loader.scaler.inverse_transform(fundo)
                self._explicadores[modelo] = ShapleyPorFator(
                    self._funcao(modelo), fundo, self._loader.feature_names)
            return self._explicadores[modelo]

    def em_cache(self, modelo: str, indice) -> dict:
        """Resultado já calculado, ou None — nunca dispara cálculo."""
        return self._cache.get((modelo, int(indice)))

    def explicar(self, modelo: str, indice) -> dict:
        """
        Fatores de um paciente do lote para um modelo (calcula se preciso).

        Returns
        -------
        dict ou None
            Saída de ``ShapleyPorFator.explicar``; None se o paciente não está
            no lote ou o modelo não tem fatores disponíveis.
        """
        chave = (modelo, int(indice))
        if chave in self._cache:
            return self._cache[chave]
        pos = self._pos.get(int(indice))
        if pos is None or not self.disponivel(modelo):
            return None

        if modelo == NOME_COMITE:
            resultado = combinar([self.explicar(m, indice) for m in self.membros()])
        else:
            x = self._X_raw[pos] if modelo == MODELO_SEM_CALIBRACAO else self._X[pos]
            resultado = self._explicador(modelo).explicar(x, z=self._X[pos])

        with self._trava:
            self._cache[chave] = resultado
        return resultado

    def agendar(self, modelo: str, indice) -> Future:
        """
        Pede o cálculo em segundo plano e devolve um ``Future`` do resultado.

        Um único trabalhador atende todas as janelas, em ordem: cliques rápidos
        não disparam quatro cálculos pesados em paralelo, e quando o comitê
        termina os membros já estão no cache para as janelas deles. A thread é
        daemon, para não segurar o fechamento do app com cálculos na fila.
        """
        chave = (modelo, int(indice))
        with self._trava:
            if chave in self._cache:
                pronto = Future()
                pronto.set_result(self._cache[chave])
                return pronto
            if chave in self._futuros:
                return self._futuros[chave]
            futuro = Future()
            self._futuros[chave] = futuro
            if self._fila is None:
                self._fila = queue.Queue()
                threading.Thread(target=self._trabalhar, name="fatores-shap",
                                 daemon=True).start()
            self._fila.put((chave, futuro))
        return futuro

    def _trabalhar(self):
        """Laço do trabalhador: calcula, na ordem, o que foi agendado."""
        while True:
            (modelo, indice), futuro = self._fila.get()
            if futuro.set_running_or_notify_cancel():
                try:
                    futuro.set_result(self.explicar(modelo, indice))
                except Exception as erro:
                    futuro.set_exception(erro)
            with self._trava:
                self._futuros.pop((modelo, indice), None)

    def regra(self, modelo: str):
        """A régua do modelo no lote, ou None sem política conhecida."""
        if self.politica is None or modelo == MODELO_SEM_CALIBRACAO:
            return None
        return self.politica.regra(modelo)

    def corte(self, modelo: str, rotulo: str):
        """
        O corte que decidiu um diagnóstico neste lote, ou None se não se sabe.

        Com a recusa ligada e faixa calibrada, Maligno é decidido pelo extremo
        superior da faixa e Benigno pelo inferior; sem faixa (ou com a recusa
        desligada), os dois pelo limiar de operação.
        """
        if self.politica is None or modelo == MODELO_SEM_CALIBRACAO:
            return None
        faixa = self.politica.faixa_recusa(modelo) if self.politica.adiar_incertos else None
        if faixa is None:
            return self.politica.limiar(modelo)
        return faixa[1] if rotulo == ROTULO_MALIGNO else faixa[0]
