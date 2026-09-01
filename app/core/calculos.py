"""
Memorial de cálculo do DiagnosisXAI: de onde vem cada número que o app exibe.

Fonte única do texto exibido no card "Como os números são calculados" da aba
"Sobre" (``views.about_view``). Existe pelo mesmo motivo que
``core.biomarkers``: a explicação precisa morar num lugar só, para não divergir
entre a tela, o relatório e a monografia.

O card responde à pergunta que a interface provoca com mais frequência: *o que
é essa porcentagem?* A resposta depende de qual porcentagem se está olhando,
porque o app exibe duas, com naturezas opostas:

- a **certeza** de um paciente, medida nele, que muda linha a linha;
- os **cortes** da régua, decididos no treino antes de qualquer paciente
  chegar, fixos para todo o lote.

Confundi-las é o erro de leitura mais caro possível aqui: quem compara uma
certeza de 18% com 50% conclui "Benigno" onde o sistema decidiu "Maligno", e
não entende por quê. Por isso a distinção abre o card, antes das seções.

Sobre a redação: o texto é escrito em frases curtas e sem travessões, porque
quem lê esta aba está tentando entender o programa pela primeira vez. Cada
termo técnico aparece seguido do que ele significa em palavras comuns. As
seções seguem a ordem em que os números aparecem na tela (Passos 1 a 5), e cada
item traz, quando existe, a fórmula literal em vez de uma paráfrase dela.
"""

# --- A distinção que o card existe para resolver ---------------------------

DUAS_PORCENTAGENS_TITULO = "As duas porcentagens do programa"

DUAS_PORCENTAGENS_INTRO = (
    "O programa mostra porcentagens de dois tipos, e trocar um pelo outro é o erro de leitura "
    "mais comum aqui. O primeiro tipo mede o paciente. O segundo é a regra do modelo, "
    "escolhida no treino, muito antes de este paciente chegar. O diagnóstico sai da comparação "
    "entre os dois."
)

# (título, natureza, exemplo, explicação) para as duas colunas lado a lado.
DUAS_PORCENTAGENS = (
    (
        "A evidência",
        "Certeza_Maligno(%)  ·  muda a cada paciente",
        "18,29%",
        "É a chance de o tumor ser maligno, segundo o modelo. O número foi calibrado, ou seja, "
        "ajustado para valer como frequência real: de cada 100 pacientes que recebem essa "
        "leitura, cerca de 18 têm mesmo um tumor maligno. Esta é a única porcentagem que fala "
        "sobre o paciente à sua frente.",
    ),
    (
        "A regra",
        "os cortes da régua (Passo 3)  ·  fixos por modelo",
        "17,0%  ·  7,0% a 27,0%",
        "É onde o modelo corta. Não fala de paciente nenhum. Foi escolhida sobre os dados de "
        "treino e vale igual para o lote inteiro. Ela diz duas coisas: a partir de qual valor "
        "o caso vira maligno, e entre quais valores o modelo não tem convicção.",
    ),
)

DUAS_PORCENTAGENS_FECHO = (
    "Junte as duas e o diagnóstico aparece. Uma certeza de 18,29% vira \"Maligno\" no KNN "
    "porque 18,29 é maior que o corte de 17,0. Se você comparar essa mesma certeza com 50%, "
    "que é o costume herdado do scikit-learn, chega à conclusão oposta. A régua do Passo 3 "
    "existe para essa comparação nunca precisar ser adivinhada."
)


# --- Onde o limiar cai na escala nativa de cada modelo ----------------------

# Por modelo: (limiar calibrado, corte equivalente no escore bruto).
#
# Como a calibração de Platt é monotônica, todo limiar na escala calibrada tem
# um equivalente exato na escala do escore bruto, obtido invertendo a sigmoide:
# de P = 1/(1 + e^(a·s + b)) sai s = ( ln((1−P)/P) − b ) / a. Estes valores são
# essa inversão, e existem porque "o limiar do Random Forest é 12%" convida à
# leitura errada de que 12% das árvores bastariam, quando a conta real pede 27%
# delas.
#
# São dados derivados do wisconsin.pkl e do data/limiares.json, então envelhecem
# se qualquer um dos dois for regerado. ``tests/test_calculos.py`` refaz a
# inversão a partir dos artefatos reais e falha quando estes números saem de
# sincronia. É o que impede a aba "Sobre" de anunciar cortes que não existem
# mais.
CORTES_BRUTOS = {
    'Regressão Logística': (0.1514, -0.97),
    'Random Forest':       (0.1205, 0.269),
    'SVM':                 (0.1548, -0.44),
    'KNN':                 (0.1698, 0.255),
}


def _corte(modelo: str, unidade: str = "") -> str:
    """Linha "limiar X% ⟷ escore bruto Y" de um modelo, para a tabela do card."""
    limiar, bruto = CORTES_BRUTOS[modelo]
    valor = f"{bruto * 100:.1f}%{unidade}" if unidade else f"{bruto:+.2f}"
    return f"limiar {limiar * 100:.2f}%  ⟷  {valor}"


# --- Seções do memorial ----------------------------------------------------

# Cada seção: {'titulo', 'resumo', 'itens': [(nome, fórmula, explicação), ...]}.
# A fórmula é opcional (string vazia quando o cálculo não cabe numa linha).
SECOES = (
    {
        'titulo': "1.  Preparação dos dados  (Passos 1 e 2)",
        'resumo': (
            "Antes de qualquer modelo rodar, o lote precisa chegar na mesma forma em que o "
            "treino chegou. Os dois passos abaixo não são cosméticos. Errar qualquer um deles "
            "muda o diagnóstico de todo mundo."
        ),
        'itens': (
            (
                "Higienização",
                "",
                "O programa joga fora as colunas id, diagnosis e Unnamed: 32. Depois coloca as "
                "30 colunas restantes na mesma ordem do treino. Isso importa porque o modelo "
                "identifica as colunas pela posição, não pelo nome. Fora de ordem, ele leria "
                "textura onde deveria ler raio, e ainda assim responderia com toda a "
                "confiança.",
            ),
            (
                "Padronização (Z-score)",
                "z = (x − μ) / σ",
                "μ é a média de cada biomarcador e σ é o desvio-padrão. Os dois vêm dos dados "
                "de treino, nunca do lote que você acabou de abrir. Se viessem do lote, cada "
                "arquivo seria comparado consigo mesmo, e o mesmo paciente mudaria de "
                "diagnóstico dependendo de quem foi enviado junto com ele.",
            ),
            (
                "Detecção de escala",
                "area_mean > 10  ⇒  dados brutos",
                "Um lote já padronizado tem área com média perto de 0. Um lote bruto tem área "
                "na casa das centenas. O programa usa essa diferença para saber se precisa "
                "padronizar. Assim evita padronizar duas vezes um arquivo que já veio "
                "tratado.",
            ),
        ),
    },
    {
        'titulo': "2.  A evidência: Certeza_Maligno(%)",
        'resumo': (
            "Cada modelo calcula um número à sua maneira. Esse número inicial se chama escore "
            "bruto. A calibração transforma o escore bruto numa porcentagem que pode ser lida "
            "como chance real. É essa porcentagem, e não o escore bruto, que vai para a tabela "
            "e é comparada com a régua."
        ),
        'itens': (
            (
                "Escore bruto",
                "s(x)",
                "É uma soma ponderada na Regressão Logística, uma média de árvores no Random "
                "Forest, uma distância no SVM e um voto de vizinhos no KNN. Sozinho, ele só "
                "serve para colocar os pacientes em ordem de suspeita. 0,9 é mais suspeito que "
                "0,7, mas nenhum dos dois vale como porcentagem de chance.",
            ),
            (
                "O escore bruto NÃO é a coluna",
                "s(x)  ≠  Certeza_Maligno(%)",
                "Esta é a confusão mais frequente. No Random Forest, um paciente cujas árvores "
                "dão em média 25% aparece na tabela com cerca de 12% de certeza. São duas "
                "escalas diferentes, e a coluna mostra sempre a segunda. Quem lê a certeza como "
                "\"fração de árvores\" acaba comparando o limiar com o número errado.",
            ),
            (
                "Calibração de Platt",
                "P = 1 / (1 + e^(A·s + B))",
                "É uma curva em S aplicada sobre o escore bruto. Os valores de A e B são "
                "aprendidos no treino, usando validação cruzada de 5 dobras (Seção 14 do "
                "notebook). É esta conta que transforma uma ordem de suspeita numa porcentagem "
                "que pode ser lida.",
            ),
            (
                "Por que calibrar",
                "escore de Brier  ·  ECE",
                "Sem calibrar, \"90%\" pode não significar 90% de verdade, e aí comparar a "
                "certeza com um corte numérico deixa de fazer sentido. O notebook mede a "
                "qualidade dessa calibração com duas estatísticas: o escore de Brier e o ECE.",
            ),
            (
                "Comitê (voto suave)",
                "P = (P_LR + P_RF + P_SVM + P_KNN) / 4",
                "É a média simples das quatro certezas já calibradas. Usa-se a média, e não a "
                "contagem de votos, porque um membro muito convicto de que o caso é maligno "
                "consegue puxar o resultado mesmo quando os outros três discordam por pouco. É "
                "justamente aí que mora o falso negativo.",
            ),
            (
                "Árvore de Decisão",
                "",
                "Ela não recebe esta coluna. As folhas da árvore são puras, então a "
                "probabilidade sai sempre 0% ou 100%. Não há incerteza para mostrar nem limiar "
                "para deslocar. Ela decide pela sequência de regras que aprendeu, e por isso "
                "também fica de fora do comitê.",
            ),
        ),
    },
    {
        'titulo': "3.  A regra: a régua de decisão  (Passo 3)",
        'resumo': (
            "São três números, e cada um nasce de um jeito. A diferença entre eles importa. "
            "Todos são calculados sobre os dados de treino, usando probabilidades out-of-fold. "
            "Isso quer dizer o seguinte: o treino é dividido em 5 partes, e cada paciente é "
            "avaliado por um modelo que foi treinado sem ele. Assim ninguém é pontuado por um "
            "modelo que já o viu. O conjunto de teste não participa dessa escolha, só da "
            "conferência depois. Escolher o corte olhando o teste seria vazamento, e o número "
            "publicado deixaria de estimar o desempenho futuro."
        ),
        'itens': (
            (
                "Limiar de operação  (τ)",
                "menor τ com especificidade ≥ 92%",
                "MEDIDO. O programa percorre em ordem crescente as probabilidades que apareceram "
                "no treino, que são os únicos pontos onde a decisão pode mudar. Ele para no "
                "primeiro corte que respeita o piso de especificidade. Como a especificidade "
                "sobe junto com o corte, esse primeiro corte aceitável também é o mais sensível "
                "de todos.",
            ),
            (
                "Por que não 50%",
                "",
                "50% é o corte que reduz o número total de erros. Isso só faria sentido se os "
                "dois tipos de erro custassem o mesmo, e em rastreio eles não custam. Um falso "
                "positivo gera um exame a mais. Um falso negativo deixa um tumor sem "
                "tratamento. O piso de especificidade existe para dar freio à busca, porque "
                "pedir apenas \"sensibilidade altíssima\" levaria a acusar quase todo mundo.",
            ),
            (
                "Faixa de recusa  [a, s)",
                "a = menor P de um maligno\ns = maior P de um benigno + 0,0001",
                "MEDIDO. Abaixo de a e acima de s, nenhum caso do treino seria decidido errado. "
                "Entre os dois valores as classes se misturam, e o sistema devolve o caso. O "
                "acréscimo de 0,0001 tem um motivo: o limite de cima decide como maligno "
                "(P ≥ s). No valor exato, o pior benigno do treino seria decidido, e decidido "
                "errado.",
            ),
            (
                "Cobertura mínima  (45%)",
                "",
                "Uma faixa que só consegue zerar erros adiando dois terços do lote não é "
                "cautela. É o modelo admitindo que suas probabilidades não separam as classes. "
                "Faixas assim são descartadas, e o modelo deixa de oferecer recusa. É o caso do "
                "KNN, que decidiria apenas 33,6% do treino.",
            ),
            (
                "Banda limítrofe  (±10 pontos)",
                "| P − τ | ≤ 0,10",
                "CONVENÇÃO, não medição. É um valor escolhido a mão no arquivo de limiares, não "
                "algo que os dados revelaram. Ele marca as decisões que virariam com um "
                "empurrão pequeno na certeza.",
            ),
            (
                "Diagnóstico final",
                "recusa ligada:  P < a ⇒ Benigno,  P ≥ s ⇒ Maligno,  entre ⇒ Revisar\n"
                "recusa desligada:  P ≥ τ ⇒ Maligno,  senão Benigno",
                "Adiar casos incertos é o comportamento padrão. Entre devolver um caso e "
                "arriscar um palpite que pode liberar um tumor maligno, o padrão seguro é "
                "devolver. Desligar a recusa é legítimo, porque o serviço pode não ter fila de "
                "revisão, mas passa a ser uma escolha explícita de quem opera.",
            ),
            (
                "Zona de decisão",
                "",
                "É um selo sobre o diagnóstico, não o diagnóstico. \"Limítrofe\" não substitui "
                "a classe. O paciente continua saindo como Benigno ou Maligno, e o selo apenas "
                "avisa que ele saiu por pouco. \"Revisar\" é o único que substitui, porque ali "
                "não houve classe nenhuma para marcar.",
            ),
        ),
    },
    {
        'titulo': "4.  A evidência é confiável? O perfil atípico",
        'resumo': (
            "Uma certeza alta não vale muito se o paciente não se parece com ninguém que o "
            "modelo viu no treino. Esta é a única verificação do programa que ignora o "
            "diagnóstico. Ela olha só para o quanto o perfil do paciente é familiar."
        ),
        'itens': (
            (
                "Distância de Mahalanobis",
                "d² = (x − μ)ᵀ Σ⁻¹ (x − μ)",
                "Mede o quanto o paciente está longe do centro do treino, levando em conta que "
                "os atributos andam juntos. A distância comum trataria raio, perímetro e área "
                "como três informações separadas, quando na prática são quase a mesma coisa.",
            ),
            (
                "Covariância de Ledoit-Wolf",
                "Σ com encolhimento",
                "É necessária justamente porque esses atributos andam juntos. Sem o "
                "encolhimento, a conta da covariância fica instável e a distância deixa de ser "
                "confiável. O encolhimento regulariza essa estimativa.",
            ),
            (
                "Limiar de atipicidade",
                "percentil 99 das distâncias do treino",
                "O corte é empírico. Não se supõe nenhuma forma de distribuição. Supõe-se "
                "apenas que um paciente típico deve ficar tão perto do centro quanto 99% do "
                "treino fica. Passou disso, o perfil é marcado como Atípico.",
            ),
            (
                "Por que não sobre o UMAP",
                "",
                "Reduzir para 2 dimensões joga fora exatamente a informação necessária para "
                "julgar se um perfil é atípico. Além disso, um paciente novo nem poderia ser "
                "projetado sem o redutor ajustado. O mapa continua útil, mas como ferramenta de "
                "inspeção visual, não de detecção.",
            ),
        ),
    },
    {
        'titulo': "5.  Auditoria contra o gabarito  (Passo 4)",
        'resumo': (
            "Todas as métricas saem da matriz de confusão. E todas contam apenas os casos que o "
            "modelo aceitou decidir. Um caso devolvido para revisão não tem decisão a pontuar. "
            "Contar como erro puniria a cautela, e contar como acerto premiaria a omissão. Ele "
            "entra só na cobertura."
        ),
        'itens': (
            (
                "Matriz de confusão",
                "VP · FN · FP · VN",
                "É a origem de tudo o que vem abaixo. Ela fica visível na tabela justamente "
                "para os percentuais poderem ser conferidos à mão quando o lote é pequeno.",
            ),
            (
                "Cobertura",
                "decididos / total",
                "É o número sem o qual as outras métricas ficam incomparáveis. Decidir menos "
                "torna todas elas mais fáceis. Um modelo que se abstivesse de tudo mostraria "
                "100% de sensibilidade e 100% de especificidade.",
            ),
            (
                "Acurácia",
                "(VP + VN) / n",
                "Sozinha, ela engana. Num lote com poucos malignos, um modelo que nunca acusa "
                "malignidade já exibe acurácia alta.",
            ),
            (
                "Sensibilidade",
                "VP / (VP + FN)",
                "De todos os tumores realmente malignos, quantos o modelo detectou. É a métrica "
                "crítica em rastreio, porque o que ela não pega vira falso negativo.",
            ),
            (
                "Especificidade",
                "VN / (VN + FP)",
                "De todos os tumores realmente benignos, quantos foram poupados de um alarme "
                "falso.",
            ),
            (
                "Precisão  (VPP)",
                "VP / (VP + FP)",
                "Dos pacientes que o modelo apontou como malignos, quantos eram mesmo. Ela "
                "depende de quantos malignos existem no lote, então não dá para comparar entre "
                "lotes diferentes.",
            ),
            (
                "Valor preditivo negativo",
                "VN / (VN + FN)",
                "Dos pacientes que o modelo liberou como benignos, quantos eram mesmo.",
            ),
            (
                "F1",
                "2·VP / (2·VP + FP + FN)",
                "Junta precisão e sensibilidade num número só, por média harmônica. Resume o "
                "quanto o modelo pega malignos sem sair alarmando benignos.",
            ),
            (
                "ROC-AUC",
                "estatística de Mann-Whitney sobre os postos",
                "É a chance de o modelo dar a um maligno sorteado uma certeza maior que a de um "
                "benigno sorteado. Ela mede a ordem, não o corte. Por isso não muda quando o "
                "limiar muda, e é a métrica justa para comparar modelos antes de escolher o "
                "ponto de operação.",
            ),
            (
                "Intervalo de confiança 95%",
                "centro = (p̂ + z²/2n) / (1 + z²/n)\n"
                "margem = z·√( p̂(1−p̂)/n + z²/4n² ) / (1 + z²/n),  z = 1,96",
                "Usa-se o intervalo de Wilson, e não o normal (Wald), porque ele continua "
                "válido com poucas amostras e perto de 0% ou 100%. Esse é exatamente o regime "
                "desta auditoria, onde acertar 100% de um punhado de malignos não significa "
                "sensibilidade perfeita.",
            ),
        ),
    },
    {
        'titulo': "6.  Explicabilidade  (Passo 5)",
        'resumo': (
            "Cada modelo é explicado pela sua própria matemática, não por uma aproximação "
            "genérica. A calibração não invalida nenhuma dessas contas. Ela é monotônica, ou "
            "seja, não troca ninguém de lugar na fila. Só muda a régua em que o escore é lido."
        ),
        'itens': (
            (
                "Árvore: ganho de informação",
                "Σ  w·H (pai)\n   − w·H (esq) − w·H (dir)",
                "Em cada nó da árvore, mede-se quanta desordem aquele atributo eliminou "
                "(w é o número de amostras no nó e H é a entropia). O total por atributo é a "
                "importância dele. Para cada paciente, o relatório mostra o caminho de regras "
                "da raiz até a folha.",
            ),
            (
                "Regressão Logística: contribuição",
                "cⱼ = wⱼ · zⱼ\nz = w·x + b\nP = 1 / (1 + e^−z)",
                "Como a entrada está padronizada, o produto wⱼ·zⱼ pode ser comparado "
                "diretamente entre atributos. Somando todas as contribuições e o intercepto sai "
                "o z, e aplicando a curva em S no z sai a probabilidade. A conta fecha exata, "
                "não é aproximação.",
            ),
            (
                "KNN: vizinhos",
                "k = 4  ·  distância de Manhattan\npeso = 1 / d",
                "Os quatro pacientes de treino mais parecidos votam, e quem está mais perto pesa "
                "mais. Com apenas 4 vizinhos, as probabilidades saem grosseiras: quando os "
                "quatro concordam, o escore vai direto para o extremo, e a calibração preserva "
                "esse empate. É por isso que o KNN não recebe faixa de recusa.",
            ),
            (
                "Random Forest: consenso",
                "voto duro: nº de árvores com P ≥ 0,5\ns(x) = média das 500 árvores",
                "São dois números diferentes, e nenhum dos dois é a coluna de certeza. O voto "
                "duro conta árvores. O escore bruto é a média das probabilidades das folhas, "
                "que podem ser impuras porque a profundidade para em 10. A certeza exibida é a "
                "calibração desse escore. O histograma mostra como as árvores se espalharam: "
                "uma floresta rachada ao meio e uma unânime podem dar a mesma média com "
                "significados opostos.",
            ),
            (
                "SVM: vetores de suporte",
                "f(x) = Σ αᵢ·yᵢ·K(x, svᵢ) + b",
                "Cada vetor de suporte empurra a decisão com o seu próprio peso, e o relatório "
                "lista os que mais empurraram naquele paciente. O valor de f(x) é a distância "
                "com sinal até a fronteira, e é ele que forma o eixo do gráfico da margem.",
            ),
            (
                "SHAP",
                "base + Σ φⱼ = P(Maligno)",
                "Cada atributo recebe a parte da previsão que cabe a ele, calculada por valores "
                "de Shapley. Árvore e Random Forest usam o TreeExplainer, que é exato e "
                "instantâneo. Os outros três usam o KernelExplainer, que é aproximado e mais "
                "lento, por isso só roda quando você abre o relatório.",
            ),
            (
                "UMAP",
                "projeção 2D",
                "Serve apenas para visualizar. Ele posiciona os pacientes de forma a preservar "
                "quem está perto de quem. Não participa de nenhuma decisão nem da detecção de "
                "perfil atípico.",
            ),
            (
                "Comitê: por que decidiu ou adiou",
                "amplitude ≥ 20 pp ⇒ discordância\n"
                "| P − τ | ≤ banda ⇒ fronteira\n"
                "| P_membro − τ_membro | ≥ 10 pp ⇒ membro convicto",
                "Separa dois motivos de adiamento que pedem condutas diferentes: os membros se "
                "contradizeram, ou a faixa é que é larga. A convicção de cada membro é medida "
                "contra o limiar dele mesmo, porque comparar probabilidades cruas entre membros "
                "seria injusto quando os cortes são diferentes. Os limites de 20 e 10 pontos "
                "são convenções, assim como a banda limítrofe.",
            ),
        ),
    },
    {
        'titulo': "7.  O escore bruto de cada modelo, e onde o limiar cai nele",
        'resumo': (
            "A seção 2 avisou que o escore bruto não é a coluna de certeza. Aqui está, modelo a "
            "modelo, o que ele é de fato. A resposta muda bastante entre eles: um logito, uma "
            "média de árvores, uma distância, um voto de vizinhos. Como a calibração é "
            "monotônica, cada limiar tem um equivalente exato na escala nativa do modelo, e "
            "para achá-lo basta inverter a curva em S: s = ( ln((1−P)/P) − b ) / a. É esse "
            "equivalente que permite conferir o limiar contra a aritmética do próprio modelo, e "
            "é ele que desfaz a leitura de que o limiar do Random Forest seria \"12% das "
            "árvores\"."
        ),
        'itens': (
            (
                "Regressão Logística",
                f"z = w·x + b\n{_corte('Regressão Logística')}",
                "O escore é o logito, ou seja, a soma dos 30 biomarcadores padronizados, cada "
                "um multiplicado pelo seu peso. A fronteira natural do modelo fica em z = 0, "
                "onde a curva em S vale 50%. O limiar calibrado empurra essa fronteira para "
                "z = −0,97, um ponto onde a curva bruta ainda marca 27%. Na prática: o paciente "
                "é chamado de maligno mesmo estando do lado benigno da fronteira.",
            ),
            (
                "Random Forest",
                f"s = média das 500 árvores\n{_corte('Random Forest', ' das árvores')}",
                "Cada árvore devolve a proporção de malignos na folha em que o paciente caiu, e "
                "o escore é a média dessas 500 proporções. Não é a contagem de árvores que "
                "\"votaram maligno\". O limiar de 12,05% equivale a mais ou menos 134 das 500 "
                "árvores. A regra real é \"mais de um quarto da floresta\", não \"12% dela\".",
            ),
            (
                "SVM",
                f"f(x) = Σ αᵢ·yᵢ·K(x, svᵢ) + b\n{_corte('SVM')}",
                "O escore é a distância com sinal até a fronteira, medida no espaço do kernel "
                "RBF. Em f(x) = 0 está a fronteira. O limiar a empurra para f(x) = −0,44, pela "
                "mesma lógica da Regressão Logística: a fronteira de operação passa a ficar "
                "dentro do território benigno, de propósito.",
            ),
            (
                "KNN",
                f"s = voto dos 4 vizinhos, peso 1/d\n{_corte('KNN', ' do peso')}",
                "Os quatro vizinhos mais próximos votam com peso inverso à distância. O corte de "
                "25,5% cai bem no meio da faixa de \"um vizinho maligno em quatro\", que no lote "
                "de teste vai de 21% a 27%. Ou seja: tendo um único vizinho maligno, quem decide "
                "o diagnóstico é o quão perto ele está.",
            ),
            (
                "Árvore de Decisão",
                "predict_proba ∈ { 0 , 1 }",
                "Ela não tem escore contínuo. As folhas são puras, então a probabilidade só "
                "assume 0 ou 1. Não há o que calibrar nem o que deslocar. É a única dos cinco "
                "sem limiar. Ela decide pela sequência de regras que aprendeu, e por isso "
                "também não recebe a coluna de certeza nem entra no comitê.",
            ),
        ),
    },
)


ONDE_NO_CODIGO = (
    "Cada número acima tem um endereço no código. A padronização está em "
    "core/batch_processor.py. A probabilidade calibrada, em core/predictor.py. A régua inteira "
    "está em core/decision.py, alimentada pelo data/limiares.json, que é gerado por "
    "scripts/calibrar_limiares.py. O perfil atípico fica em core/ood_detector.py, as métricas "
    "da auditoria em core/metrics.py, e os explicadores em core/explainers.py, "
    "core/shap_explainer.py e core/committee.py."
)
