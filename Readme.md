# 🎗️ XAI para Diagnóstico Preditivo de Câncer de Mama

> **Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação**
> Universidade Tuiuti do Paraná · 2026

Software com Inteligência Artificial Explicável (XAI) para apoio ao diagnóstico preditivo de câncer de mama, integrando modelos de Machine Learning com SHAP e UMAP sobre a base pública *Breast Cancer Wisconsin (WDBC)*.

---

## 📋 Sobre o Projeto

O diagnóstico precoce é um fator determinante para o sucesso do tratamento do câncer de mama. Modelos de Machine Learning são capazes de identificar padrões complexos em dados clínicos de forma automatizada — mas sua adoção no ambiente médico esbarra no problema da **"caixa preta"**: médicos precisam entender *por que* o modelo tomou determinada decisão.

Este projeto resolverá esse problema combinando:

- **Modelos preditivos** (Por enquanto: Árvore de Decisão, KNN, Regressão Logística, Random Forest e SVM) treinados na base WDBC
- **SHAP** (*SHapley Additive exPlanations*) para explicar o impacto de cada variável clínica na previsão individual
- **UMAP** (*Uniform Manifold Approximation and Projection*) para visualização da separabilidade natural dos dados
- **Processamento em lote** via upload de planilhas, eliminando a digitação manual de dezenas de variáveis
- **Geração de relatórios automatizados** (PDF/CSV) com os diagnósticos e suas justificativas clínicas

---

## 🗂️ Estrutura do Repositório

```text
├── app/                     # Interface do usuário e lógica do sistema
│   ├── assets/              # Arquivos estáticos
│   │   └── images/          # Ícones e logos do sistema
│   ├── core/                # Processamento de dados e Inteligência Artificial
│   │   ├── batch_processor.py # Lógica de processamento em lote
│   │   ├── inference.py       # Script de carregamento do modelo treinado
│   │   └── xai_generator.py   # Lógica para geração de gráficos SHAP/UMAP
│   ├── utils/               # Funções de apoio e configurações globais
│   │   ├── config.py        # Cores padrão, fontes e caminhos do sistema
│   │   └── file_manager.py  # Funções para salvar CSVs e abrir PDFs
│   ├── views/               # Telas e componentes da interface gráfica
│   │   ├── dashboard_view.py  # Tela de resumo e métricas
│   │   ├── history_view.py    # Tela de consultas a lotes antigos
│   │   ├── main_window.py     # Menu lateral e gerenciador de telas
│   │   └── predict_view.py    # Tela de upload de arquivos e processamento
│   └── main.py              # Ponto de entrada (inicialização do aplicativo)
│
├── data/                    # Conjuntos de treino/teste processados
│   └── wisconsin.pkl        
│
├── notebooks/               # Experimentos, análises e treinamento de modelos
│   └── Wisconsin.ipynb      
│
├── reports/                 # Saídas do sistema
│   ├── figures/             
│   └── logs/     
│           
├── README.md     
│          
└── .gitignore
```

---

## 🔬 Base de Dados

**Breast Cancer Wisconsin Diagnostic (WDBC)**
- **Fonte:** [UCI Machine Learning Repository via Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Amostras:** 569 pacientes · **Atributos:** 30 características morfológicas de núcleos celulares
- **Classes:** Benigno (`B`) · Maligno (`M`)
- **Origem:** Medidas extraídas por FNA (*Fine Needle Aspiration*) com interface gráfica (Street et al., 1993)

---

## 🧠 Modelos Implementados Atualmente

| Modelo | Observação |
|---|---|
| Árvore de Decisão | Critério de entropia |
| KNN | 5 vizinhos |
| Regressão Logística | `solver=lbfgs`, `class_weight=balanced` |
| Random Forest | 500 estimadores, `class_weight=balanced` |
| SVM | Kernel RBF, `class_weight=balanced` |

Todos os modelos são avaliados com: **Acurácia, Sensibilidade (Recall+), Especificidade (Recall−), F1, ROC-AUC e PR-AUC**.

---

## ⚙️ Instalação

**Pré-requisitos:** Python 3.11+

```bash
# Clone o repositório
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio

# Instale as dependências
pip install numpy pandas matplotlib seaborn plotly Pillow scikit-learn tensorflow yellowbrick kagglehub umap-learn shap customtkinter
```

> Na primeira execução, o Kaggle Hub fará o download automático da base de dados.
> É necessário ter as credenciais do Kaggle configuradas (`~/.kaggle/kaggle.json`).

---

## 🚀 Como Usar

Abra e execute o notebook na ordem das seções:

```bash
jupyter notebook Wisconsin.ipynb
```

| Seção | Conteúdo |
|---|---|
| 1 | Importações e configurações globais |
| 2 | Funções auxiliares (ROC/PR, Matriz de Confusão, métricas) |
| 3 | Carregamento e preparação dos dados |
| 4 | Análise Exploratória — EDA (estatísticas, histogramas, boxplots) |
| 5 | Árvore de Decisão — baseline |
| 6 | KNN |
| 7 | Regressão Logística |
| 8 | Random Forest |
| 9 | SVM |
| 10 | Visualização de estrutura: UMAP e PCA |
| 11 | Interpretabilidade com SHAP |

---

## 📦 Produtos do Projeto

- [ ] Código-fonte dos modelos de ML com SHAP e UMAP
- [ ] Software de processamento em lote (upload de planilhas)
- [ ] Geração de relatórios automatizados (PDF/CSV/Cards)
- [ ] Documentação da arquitetura e diagramas de fluxo de dados
- [ ] Relatório de testes com métricas de validação

---

## 📅 Cronograma

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

## 📚 Referências Principais

- **Shaon et al. (2024)** — Seleção de atributos com SHAP para detecção de câncer de mama; acurácia de até 99,82%
- **Chen et al. (2023)** — Classificação com XGBoost, RF, KNN e LogReg; destaque para a métrica de Recall
- **Rabiei et al. (2022)** — Uso de SMOTE para balanceamento; Random Forest como melhor classificador
- **Street et al. (1993)** — Artigo fundacional da base WDBC

---

## 👤 Autor

| | |
|---|---|
| **Nome** | Gabriel Ast dos Santos |
| **E-mail** | ast.gabriel2004@gmail.com |
| **Universidade** | Universidade Tuiuti do Paraná |

---