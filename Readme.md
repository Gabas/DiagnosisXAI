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
5. [Estrutura do Repositório](#estrutura-do-repositório)
6. [Base de Dados](#base-de-dados)
7. [Modelos Implementados](#modelos-implementados)
8. [Cronograma](#cronograma)
9. [Referências](#referências)

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
| **Sobre** | Informações sobre o projeto, base de dados e tecnologias |

### Formato esperado do CSV de entrada

O arquivo deve conter as 30 colunas morfológicas da base Wisconsin (sem `id` e `diagnosis`). A padronização Z-score é aplicada automaticamente.

Os arquivos de teste já estão em `data/`:

| Arquivo | Uso |
|---|---|
| `dataTeste_sem_diagnostico.csv` | Entrada para o app (Passo 1) |
| `dataTeste_com_diagnostico.csv` | Gabarito para a etapa de auditoria (Passo 4) |

---

## Estrutura do Repositório

```text
DiagnosisXAI/
├── app/
│   ├── core/
│   │   ├── batch_processor.py   # Limpeza e padronização Z-score
│   │   ├── inference.py         # Carregamento dos modelos do .pkl
│   │   ├── predictor.py         # Motor de inferência (diagnóstico)
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
├── requirements.txt
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
| KNN | 5 vizinhos |
| Regressão Logística | `solver=lbfgs`, `class_weight=balanced` |
| Random Forest | 500 estimadores, `class_weight=balanced` |
| SVM | Kernel RBF, `class_weight=balanced` |

Todos avaliados com: Acurácia, Sensibilidade, Especificidade, F1, ROC-AUC e PR-AUC.

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
