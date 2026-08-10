"""
Explicabilidade do comitê de voto suave.

O comitê é o único modelo do app sem explicador próprio: ele não tem
coeficientes, vizinhos, regras nem vetores de suporte — a decisão dele é uma
média das probabilidades calibradas de quatro modelos. Explicá-lo, portanto,
não é abrir um modelo: é mostrar **como os quatro chegaram (ou não chegaram) a
um acordo**.

Isso importa sobretudo quando a recusa está ligada. "Revisar" não tem um motivo
só, e a média sozinha não os distingue — mas cada um pede uma conduta diferente:

- **Discordância**: os membros se dividiram, com opiniões distantes entre si. A
  média caiu na faixa por cancelamento, não por ignorância. Vale abrir o
  relatório de quem discordou: o desacordo costuma apontar um perfil que um
  modelo reconhece e os outros não.
- **Fronteira**: a média ficou perto do limiar de operação do comitê. É o caso
  genuinamente indeciso — um deslocamento pequeno inverteria a resposta.
- **Cautela da política**: os membros concordaram, com convicção, e ainda assim
  o caso foi devolvido, porque a faixa de recusa é larga o bastante para
  alcançá-lo. Aqui quem está inseguro não é o modelo, é a regra de operação:
  a faixa foi calibrada para não errar nenhum caso do treino, e é definida
  pelos dois pacientes mais atípicos que ele continha. Distinguir esta situação
  evita ler como dúvida clínica o que é conservadorismo do ponto de operação.

O módulo é puro (numpy apenas): recebe as probabilidades já calculadas pelo
``PredictorEngine`` e devolve dicionários no mesmo formato dos demais
explicadores, para que a janela de relatório e a exportação em PDF funcionem
sem tratamento especial.
"""

import numpy as np

from core.decision import ROTULO_MALIGNO, ROTULO_REVISAR

# Distância (em pontos percentuais) do limiar do próprio membro a partir da qual
# consideramos que ele "tem opinião". Abaixo disso, o membro está em cima do
# próprio corte e seu voto não sustenta nem concordância nem discordância.
MARGEM_OPINIAO = 10.0

# Amplitude (em pontos percentuais) entre o membro mais alto e o mais baixo a
# partir da qual o caso é tratado como desacordo. Abaixo disso os membros estão
# essencialmente no mesmo lugar, e a média representa bem o conjunto.
DISPERSAO_DISCORDANCIA = 20.0

MOTIVO_DISCORDANCIA = 'discordância'
MOTIVO_FRONTEIRA = 'fronteira'
MOTIVO_CAUTELA = 'cautela'
MOTIVO_CONSENSO = 'consenso'
MOTIVO_MAIORIA = 'maioria'


def explicar(probabilidades: dict, indices, politica, nome_comite: str) -> dict:
    """
    Monta o relatório de concordância do comitê para um lote.

    Parameters
    ----------
    probabilidades : dict[str, array-like]
        ``{nome_do_membro: P(Maligno) calibrada}`` — vetores entre 0 e 1, todos
        do mesmo tamanho e na ordem do lote.
    indices : sequence
        Índice de cada paciente, na mesma ordem.
    politica : core.decision.PoliticaDecisao
        Política em vigor: fornece o limiar de cada membro, o do comitê e a
        faixa de recusa (para desenhar as referências e classificar o motivo).
    nome_comite : str
        Nome do comitê, para resolver o limiar e a faixa dele.

    Returns
    -------
    dict
        ``{'membros', 'limiares', 'limiar_comite', 'faixa', 'explicacoes',
        'resumo'}`` — ``explicacoes`` segue o formato dos demais explicadores
        (uma entrada por paciente, com 'indice', 'classe', 'probabilidade',
        'confianca' e 'limitrofe'), acrescido dos campos próprios do comitê.
    """
    membros = list(probabilidades)
    matriz = np.array([np.asarray(probabilidades[m], dtype=float) for m in membros])
    media = matriz.mean(axis=0)

    limiares = {m: politica.limiar(m) for m in membros}
    limiar_comite = politica.limiar(nome_comite)
    faixa = politica.faixa_recusa(nome_comite) if politica.adiar_incertos else None

    explicacoes = []
    for i, indice in enumerate(indices):
        coluna = matriz[:, i]
        p = float(media[i])
        classe = politica.rotular(p, nome_comite)

        detalhes = [_detalhar_membro(nome, float(coluna[j]), limiares[nome], p)
                    for j, nome in enumerate(membros)]
        votos_maligno = sum(1 for d in detalhes if d['classe'] == ROTULO_MALIGNO)
        amplitude = (coluna.max() - coluna.min()) * 100

        # Classe que o comitê daria sem a recusa — é contra ela que se mede a
        # concordância dos membros, para a métrica não ficar vazia justamente
        # nos casos difíceis (os adiados não teriam classe a acompanhar).
        inclinacao = ROTULO_MALIGNO if p >= limiar_comite else 'Benigno'

        explicacoes.append({
            'indice': int(indice),
            'classe': classe,
            'inclinacao': inclinacao,
            'probabilidade': round(p * 100, 1),
            'confianca': round((max(p, 1 - p) if classe == ROTULO_REVISAR
                                else (p if classe == ROTULO_MALIGNO else 1 - p)) * 100, 1),
            'limitrofe': politica.zona(p, nome_comite) != 'Definida',
            'votos_maligno': votos_maligno,
            'votos_benigno': len(membros) - votos_maligno,
            'amplitude': round(amplitude, 1),
            'desvio': round(float(coluna.std()) * 100, 1),
            'motivo': _motivo(classe, votos_maligno, len(membros), amplitude,
                              abs(p - limiar_comite) * 100, politica.banda * 100),
            'discordantes': [d['nome'] for d in detalhes if d['classe'] != inclinacao],
            'membros': detalhes,
        })

    return {
        'membros': membros,
        'limiares': {m: round(v * 100, 1) for m, v in limiares.items()},
        'limiar_comite': round(limiar_comite * 100, 1),
        'faixa': [round(faixa[0] * 100, 1), round(faixa[1] * 100, 1)] if faixa else None,
        'explicacoes': explicacoes,
        'resumo': _resumo(membros, matriz, media, explicacoes),
    }


def _detalhar_membro(nome: str, prob: float, limiar: float, media: float) -> dict:
    """Como um membro se posicionou: classe, distância do próprio corte e da média."""
    return {
        'nome': nome,
        'probabilidade': round(prob * 100, 1),
        'limiar': round(limiar * 100, 1),
        'classe': ROTULO_MALIGNO if prob >= limiar else 'Benigno',
        # Distância ao próprio limiar: é o que mede convicção. Comparar
        # probabilidades cruas entre membros seria injusto, porque cada um opera
        # num corte diferente (o RF decide Maligno a partir de 12%; o KNN, 17%).
        'margem': round((prob - limiar) * 100, 1),
        'convicto': abs(prob - limiar) * 100 >= MARGEM_OPINIAO,
        'puxou': round((prob - media) * 100, 1),
    }


def _motivo(classe: str, votos_maligno: int, n_membros: int,
            amplitude: float, distancia_do_limiar: float, banda: float) -> str:
    """
    Classifica por que o comitê decidiu — ou por que não decidiu.

    Para os casos devolvidos, a pergunta é o que colocou a média dentro da
    faixa; ver a discussão no topo do módulo. A ordem dos testes importa: o
    desacordo entre membros é o achado mais informativo e por isso vem antes,
    mesmo quando a média também está perto do limiar.

    Parameters
    ----------
    classe : str
        Resultado do comitê para o paciente.
    votos_maligno, n_membros : int
        Quantos membros apontariam Maligno pelo próprio limiar, e o total.
    amplitude : float
        Diferença, em pontos percentuais, entre o membro mais alto e o mais baixo.
    distancia_do_limiar : float
        Distância, em pontos percentuais, entre a média e o limiar do comitê.
    banda : float
        Meia-largura da faixa de revisão, em pontos percentuais.
    """
    if classe == ROTULO_REVISAR:
        if amplitude >= DISPERSAO_DISCORDANCIA:
            return MOTIVO_DISCORDANCIA
        if distancia_do_limiar <= banda:
            return MOTIVO_FRONTEIRA
        # Membros juntos e longe do limiar: quem adiou foi a política, não o modelo.
        return MOTIVO_CAUTELA
    return MOTIVO_MAIORIA if 0 < votos_maligno < n_membros else MOTIVO_CONSENSO


def _resumo(membros: list, matriz, media, explicacoes: list) -> dict:
    """
    Estatísticas do lote inteiro: com que frequência cada membro acompanha o comitê.

    Um membro que quase nunca acompanha não está necessariamente errado — pode
    ser o único a perceber um perfil —, mas é o que mais desloca a média, e
    saber disso é o que permite julgar o comitê como conjunto.

    A comparação é contra a *inclinação* do comitê (o lado para o qual a média
    pende), e não contra a classe final: medir só sobre os casos decididos
    devolveria 100% para todos, já que decidir é justamente o que o comitê faz
    quando ninguém discorda.
    """
    total = len(explicacoes) or 1
    por_membro = {}
    for j, nome in enumerate(membros):
        acompanhou = sum(1 for e in explicacoes for d in e['membros']
                         if d['nome'] == nome and d['classe'] == e['inclinacao'])
        por_membro[nome] = {
            'media': round(float(matriz[j].mean()) * 100, 1),
            'acompanhou': acompanhou,
            'total': len(explicacoes),
            'taxa': round(acompanhou / total * 100, 1),
            'desvio_medio': round(float(np.abs(matriz[j] - media).mean()) * 100, 1),
        }

    contagem = {}
    for e in explicacoes:
        contagem[e['motivo']] = contagem.get(e['motivo'], 0) + 1

    return {'por_membro': por_membro, 'motivos': contagem, 'total': len(explicacoes)}