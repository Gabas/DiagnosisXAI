# DiagnosisXAI

**Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação**
Universidade Tuiuti do Paraná · 2026 · Gabriel Ast dos Santos

Software de apoio ao diagnóstico preditivo de câncer de mama com Inteligência Artificial Explicável (XAI), integrando modelos de Machine Learning com SHAP e UMAP sobre a base pública *Breast Cancer Wisconsin (WDBC)*.

---

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Rodando o Notebook](#rodando-o-notebook)
4. [Rodando o App](#rodando-o-app)
5. [Rodando os Testes](#rodando-os-testes)
6. [Estrutura do Repositório](#estrutura-do-repositório)
7. [Base de Dados](#base-de-dados)
8. [Modelos Implementados](#modelos-implementados)
9. [Confiabilidade da Predição (XAI)](#confiabilidade-da-predição-xai)
10. [Cronograma](#cronograma)
11. [Referências](#referências)

---

## Pré-requisitos

- **Python 3.11 ou 3.12** — versões anteriores não são suportadas
- **Git**

### Verificar se já estão instalados

```bash
python3 --version   # deve retornar 3.11.x ou 3.12.x
git --version
```

### Instalar Python (caso necessário)

**macOS:**
```bash
# Via Homebrew (recomendado)
brew install python@3.12

# Ou baixe o instalador oficial em: https://www.python.org/downloads/
```

**Windows:**
Baixe o instalador em [python.org/downloads](https://www.python.org/downloads/) e marque a opção **"Add Python to PATH"** durante a instalação.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip
```

---

## Instalação

> Execute todos os comandos abaixo no terminal, a partir da raiz do repositório.

### 1. Clonar o repositório

```bash
git clone https://github.com/GabAST/DiagnosisXAI.git
cd DiagnosisXAI
```

### 2. Criar e ativar o ambiente virtual

Um ambiente virtual isola as dependências do projeto sem afetar o restante do sistema.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

O prompt mostrará `(.venv)` no início quando o ambiente estiver ativo.

### 3. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> A instalação pode levar alguns minutos na primeira vez — `shap` e `umap-learn` compilam extensões nativas.

### Nota para macOS com Apple Silicon (M1/M2/M3/M4)

As dependências são compatíveis com Apple Silicon. Caso `umap-learn` ou `shap` falhem na compilação, instale as ferramentas de linha de comando do Xcode antes:

```bash
xcode-select --install
```

---

## Rodando o Notebook

O notebook cobre todo o pipeline científico: análise exploratória (EDA), treinamento dos modelos, avaliação com métricas, SHAP e UMAP. Ao final, salva o arquivo `data/wisconsin.pkl` com os modelos treinados, necessário para o app.

### 1. Iniciar o JupyterLab

Com o ambiente virtual ativo:

```bash
jupyter lab
```

O navegador abrirá em `http://localhost:8888`. Navegue até `notebooks/Wisconsin.ipynb`.

### 2. Executar o notebook

Execute todas as células na ordem com **Run > Run All Cells** ou `Shift+Enter` célula a célula.

| Seção | Conteúdo |
|---|---|
| 1 | Importações e configurações globais |
| 2 | Funções auxiliares (ROC/PR, Matriz de Confusão, métricas) |
| 3 | Carregamento e preparação dos dados |
| 4 | Análise Exploratória (EDA) — estatísticas, histogramas, boxplots |
| 5 | Árvore de Decisão |
| 6 | KNN |
| 7 | Regressão Logística |
| 8 | Random Forest |
| 9 | SVM |
| 10 | Redução dimensional: UMAP e PCA |
| 11 | Interpretabilidade com SHAP |

> Ao concluir a Seção 9, o arquivo `data/wisconsin.pkl` será (re)gerado com os modelos treinados.

---

## Rodando o App

O app é uma interface gráfica desktop que permite importar lotes de pacientes (CSV), aplicar padronização Z-score e obter diagnósticos dos 5 modelos de IA.

> O arquivo `data/wisconsin.pkl` precisa existir antes de rodar o app. Ele já está incluído no repositório — só precisará ser regerado se você retreinar os modelos pelo notebook.

Com o ambiente virtual ativo, a partir da raiz do repositório:

**macOS / Linux:**
```bash
cd app
python3 main.py
```

**Windows:**
```bash
cd app
python main.py
```

A janela do aplicativo abrirá. Navegue pelas abas:

| Aba | Função |
|---|---|
| **Início** | Tela inicial com status do sistema |
| **Novo Diagnóstico** | Pipeline completo: importar CSV → padronizar → executar IA → auditar |
| **Sobre** | Informações sobre o projeto, base de dados, tecnologias e o glossário de biomarcadores |

Além do diagnóstico, a tabela de resultados traz colunas de **confiabilidade** (certeza calibrada, caso limítrofe e perfil atípico) — ver [Confiabilidade da Predição](#confiabilidade-da-predição-xai).

### Formato esperado do CSV de entrada

O arquivo deve conter as 30 colunas morfológicas da base Wisconsin (sem `id` e `diagnosis`). A padronização Z-score é aplicada automaticamente.

Os arquivos de teste já estão em `data/`:

| Arquivo | Uso |
|---|---|
| `dataTeste_sem_diagnostico.csv` | Entrada para o app (Passo 1) |
| `dataTeste_com_diagnostico.csv` | Gabarito para a etapa de auditoria (Passo 4) |

---

## Rodando os Testes

Com o ambiente virtual ativo, a partir da raiz do repositório:

```bash
pytest
```

Os testes cobrem a lógica de `app/core/` e `app/utils/pdf_report.py` — não a
interface gráfica. Dois grupos:

- **Isolados** (`test_explainers.py`, `test_history_manager.py`, `test_pdf_report.py`):
  não dependem do `wisconsin.pkl` nem do `data/history.json` reais. Os
  explicadores são testados sobre modelos treinados na hora, com a base
  pública do scikit-learn (`load_breast_cancer`) — o foco é validar que a
  decisão exibida (classe, confiança, votos) sempre bate com a decisão real
  do modelo (`predict`/`predict_proba`/`decision_function`), o tipo de
  inconsistência já encontrado e corrigido no SVM e no KNN durante o
  desenvolvimento.
- **De integração** (`test_batch_processor.py`, `test_predictor.py`):
  exercitam o `data/wisconsin.pkl` versionado no repositório — servem também
  como smoke test do artefato: se o notebook for reexecutado e gerar um
  `.pkl` com um formato diferente, esses testes acusam.

---

## Estrutura do Repositório

```text
DiagnosisXAI/
├── app/
│   ├── core/
│   │   ├── batch_processor.py   # Limpeza e padronização Z-score
│   │   ├── biomarkers.py        # Glossário dos 30 atributos (descrições + tooltips)
│   │   ├── inference.py         # Carregamento dos modelos do .pkl
│   │   ├── ood_detector.py      # Aviso de perfil atípico (fora da distribuição)
│   │   ├── predictor.py         # Motor de inferência (diagnóstico + colunas de confiabilidade)
│   │   └── xai_generator.py     # Geração de gráficos SHAP/UMAP
│   ├── utils/
│   │   ├── config.py            # Constantes e caminhos globais
│   │   └── file_manager.py      # Utilitários de arquivo
│   ├── views/
│   │   ├── about_view.py        # Aba "Sobre" (info do projeto)
│   │   ├── dashboard_view.py    # Tela inicial
│   │   ├── main_window.py       # Janela principal e navegação
│   │   └── predict_view.py      # Pipeline de diagnóstico
│   └── main.py                  # Ponto de entrada do app
│
├── data/
│   ├── wisconsin.pkl                      # Modelos e scaler serializados
│   ├── dataTreinamento_com_diagnostico.csv
│   ├── dataTreinamento_sem_diagnostico.csv
│   ├── dataTeste_com_diagnostico.csv
│   └── dataTeste_sem_diagnostico.csv
│
├── notebooks/
│   └── Wisconsin.ipynb          # Pipeline científico completo
│
├── tests/                       # Testes automatizados (pytest) do app/core
│
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## Base de Dados

**Breast Cancer Wisconsin Diagnostic (WDBC)**

| Atributo | Valor |
|---|---|
| Fonte | UCI Machine Learning Repository |
| Amostras | 569 pacientes |
| Classes | Benigno (357) · Maligno (212) |
| Atributos | 30 biomarcadores morfológicos de núcleos celulares |
| Extração | Punção Aspirativa por Agulha Fina (PAAF) |
| Padronização | Z-Score (μ = 0, σ = 1) |

---

## Modelos Implementados

| Modelo | Configuração |
|---|---|
| Árvore de Decisão | Critério de entropia |
| KNN | 4 vizinhos, peso por distância, métrica Manhattan |
| Regressão Logística | `solver=liblinear`, `C=0.1`, `class_weight=balanced` |
| Random Forest | 500 estimadores, profundidade máx. 10, `class_weight=balanced_subsample` |
| SVM | Kernel RBF, `class_weight=balanced` |

Todos avaliados com: Acurácia, Sensibilidade, Especificidade, F1, ROC-AUC e PR-AUC.

### Rigor estatístico (Seção 10 do notebook)

Além do desempenho num único conjunto de teste, os modelos passam por três análises complementares que reforçam o rigor metodológico:

| Análise | O que responde | Método |
|---|---|---|
| **Validação cruzada** | Quão *estável* é o desempenho entre diferentes partições? | k-fold estratificado repetido (10×5), com padronização reajustada dentro de cada fold (sem vazamento) — reporta média ± desvio-padrão |
| **Significância estatística** | As diferenças *entre modelos* são reais ou ruído? | Teste pareado 5×2cv (Dietterich) + Friedman com pós-hoc de Nemenyi (α = 0,05) |
| **Calibração** | A *probabilidade* prevista corresponde à frequência real? | Diagrama de confiabilidade, escore de Brier e ECE; recalibração por escalonamento de Platt |

As probabilidades **recalibradas** (Seção 14) alimentam a coluna "Certeza (%)" do App — ver [Confiabilidade da Predição](#confiabilidade-da-predição-xai).

---

## Confiabilidade da Predição (XAI)

Além do diagnóstico, o App comunica **quando confiar menos na previsão** — princípio central da IA explicável aplicada ao apoio à decisão clínica. Para cada paciente, a tabela de resultados (e os arquivos CSV/PDF exportados) traz:

| Coluna | Pergunta que responde | O que informa |
|---|---|---|
| `Diagnóstico_IA` | *O quê?* | O que o sistema entrega: `Maligno`, `Benigno` ou `Revisar` (abstenção) |
| `Certeza_Maligno(%)` | *Com quanta evidência?* | Probabilidade **calibrada** de malignidade (0–100%) |
| `Zona_de_Decisão` | *Quão firme?* | `Definida`, `Limítrofe` ou `Revisar` — em que faixa da régua a certeza caiu |
| `Perfil` | *A evidência vale?* | `Atípico` quando o paciente está fora da distribuição de treino; senão `Típico` |

### A régua de decisão

O Passo 3 mostra, por extenso, **como o modelo selecionado decide** — e por quê. São três faixas contíguas que cobrem 0–100%, com a do meio sendo exatamente o intervalo em que a decisão é incerta:

| Resultado | Faixa de certeza (ex.: Comitê) | O que o sistema faz |
|---|---|---|
| Benigno | `< 1,1%` | decide Benigno, com folga |
| Revisar | `1,1% a 68,2%` | não decide — devolve o caso para revisão humana |
| Maligno | `≥ 68,2%` | decide Maligno, com folga |

A faixa do meio muda conforme o modo de operação:

- **Adiar casos incertos ligado (padrão)** — a faixa é a de recusa: `1,1%` é a menor certeza que um paciente maligno recebeu no treino e `68,2%` a maior que um benigno recebeu (probabilidades *out-of-fold*). Fora desses limites, nenhum caso do treino seria decidido errado.
- **Desligado** — a faixa é o limiar de operação ±10 pontos, e ali o sistema decide mas marca o caso como `Limítrofe`.

**Por que a recusa é o padrão:** o erro que este sistema existe para evitar é o falso negativo. Chamar de benigno um tumor maligno manda o paciente para casa sem qualquer sinal de alerta, e o custo aparece meses depois. Diante de uma probabilidade que não separa as classes, arriscar um palpite não é neutro — é escolher esse risco. Quem opera pode desligar a recusa e receber um rótulo para todos os casos, mas isso passa a ser uma decisão consciente. Modelos sem faixa calibrada (KNN e Árvore de Decisão) não oferecem a opção e decidem todos os casos.

O painel também exibe o **motivo do corte** (critério, dados usados e desempenho medido), e a régua acompanha o lote nos PDFs exportados — sem ela, uma linha "Maligno, certeza 25%" não seria conferível fora do App.

### O trio de confiabilidade

1. **Calibração** — a certeza exibida corresponde à frequência real de malignidade (validada por escore de Brier e ECE; recalibração por escalonamento de Platt, Seção 14 do notebook). Sem isso, "90%" pode não significar 90%.
2. **Aviso de perfil atípico (*out-of-distribution*)** — mede a distância de **Mahalanobis** (covariância regularizada de Ledoit-Wolf) do paciente ao centro do treino, no espaço completo dos 30 atributos padronizados. Acima do percentil 99 do treino, o perfil é marcado como atípico: o modelo está extrapolando para uma região pouco vista. *(A projeção UMAP 2D não é usada aqui — reduzir a 2D descartaria justamente a informação necessária para julgar atipicidade.)*
3. **Zona de decisão** — como a certeza é calibrada, a distância até o corte reflete incerteza real: dentro da faixa incerta, um pequeno deslocamento inverteria o diagnóstico. Esses casos são devolvidos (`Revisar`) ou, com a recusa desligada, sinalizados (`Limítrofe`).

> Em conjunto, esses sinais permitem ao sistema não apenas prever, mas **avisar quando a previsão é menos confiável** — seja porque a decisão em si é incerta (a certeza caiu na faixa do meio), seja porque o próprio paciente é atípico em relação ao que o modelo aprendeu (OOD).

### Memorial de cálculo (in-app)

A aba **Sobre** traz o card **"Como os números são calculados"**: de onde vem cada valor que o sistema exibe, com a fórmula literal ao lado. Abre justamente pela distinção que a interface provoca com mais frequência — **as duas porcentagens do programa**:

| | O que é | Natureza |
|---|---|---|
| `Certeza_Maligno(%)` | probabilidade calibrada daquele paciente | medida **no paciente**, muda linha a linha |
| Os cortes da régua | onde o modelo decide cortar | escolhidos **no treino**, fixos para todo o lote |

Confundi-las leva a comparar uma certeza de 18% com 50% e concluir "Benigno" onde o sistema decidiu "Maligno". Em seguida o card percorre, na ordem dos Passos 1 a 5: padronização Z-score, calibração de Platt, a régua (limiar, faixa de recusa e banda — e quais são medidos e qual é convenção), a distância de Mahalanobis do perfil atípico, as métricas da auditoria com o intervalo de Wilson, e a matemática de cada explicador. O conteúdo é fonte única em [`app/core/calculos.py`](app/core/calculos.py), com testes que comparam os números citados no texto com os que o código realmente usa.

### Glossário de biomarcadores (in-app)

Para completar a explicabilidade, a aba **Sobre** traz um glossário dos 30 atributos: o que cada biomarcador mede, o significado das 3 estatísticas (`_mean`/`_se`/`_worst`) e — um ponto de honestidade científica — que a maioria **não tem unidade física** (são índices adimensionais ou medidas na escala de pixel da imagem, não em milímetros), além de serem padronizados por Z-score antes da predição. As mesmas descrições aparecem como *tooltip* ao passar o mouse sobre os cabeçalhos da tabela de resultados.

### Relatórios de explicabilidade por modelo

No **Passo 5**, cada modelo abre uma janela de explicabilidade própria — com o ranking global de biomarcadores, um gráfico e o detalhamento por paciente (exportável em PDF). Cada tela é adaptada à natureza do modelo:

| Modelo | O que a tela mostra |
|---|---|
| **Árvore de Decisão** | **Ganho de informação (redução de entropia)** por biomarcador, com o valor `Δentropia` de cada um, e o caminho de regras da raiz à folha por paciente |
| **Regressão Logística** | **Curva logística** σ(z): cada paciente é posicionado em (escore `z = w·x + b`, `P(Maligno) = σ(z)`). Ao selecionar um paciente, exibe a **função matemática da decisão** — o valor de `z`, a aplicação da sigmoide e a regra `z ≷ 0` |
| **KNN** | Mapa de vizinhança 2D (PCA) destacando os *K* vizinhos que votaram, com **zoom/pan** |
| **Random Forest** | Importância por permutação + a votação das árvores por paciente |
| **SVM** | Mapa com os vetores de suporte em destaque + gráfico **interativo da margem**: a "rua" entre `z = −1` e `z = +1` que o SVM maximiza, com a fronteira em `z = 0` — torna a margem *visível* |

Os gráficos embutidos têm **zoom/pan** (lupa). Além disso, o **Mapa Populacional (UMAP)** e a **margem do SVM** oferecem versões **interativas em Bokeh** (hover com os dados do paciente, zoom, legenda clicável) abertas no navegador. O **SHAP** complementa, atribuindo a cada biomarcador sua contribuição para a decisão de cada paciente.

---

## Cronograma

| Etapa | Mar | Abr | Mai | Jun | Jul | Ago | Set |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Proposta e Revisão Bibliográfica | ✓ | | | | | | |
| EDA, Limpeza de Dados e UMAP | ✓ | | | | | | |
| Fundamentação Teórica e Metodologia | ✓ | ✓ | | | | | |
| Treinamento e Seleção do Modelo | | ✓ | | | | | |
| Interpretabilidade (SHAP) | | ✓ | ✓ | | | | |
| **Entrega PG I** | | | ✓ | | | | |
| Processamento em Lote (Upload) | | | ✓ | | | | |
| Geração de Relatórios (PDF/CSV) | | | ✓ | ✓ | | | |
| Redação de Resultados e Conclusão | | | | ✓ | ✓ | | |
| Testes e Correção de Bugs | | | | | ✓ | ✓ | |
| Formatação ABNT e Revisão Final | | | | | | ✓ | ✓ |

---

## Referências

- **Shaon et al. (2024)** — Seleção de atributos com SHAP para detecção de câncer de mama
- **Chen et al. (2023)** — Classificação com XGBoost, RF, KNN e Regressão Logística
- **Rabiei et al. (2022)** — Uso de SMOTE para balanceamento; Random Forest como melhor classificador
- **Street et al. (1993)** — Artigo fundacional da base WDBC

---

**Orientador:** Prof. Rodrigo Ramos Alves ·
