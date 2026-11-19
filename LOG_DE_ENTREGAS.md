# Log de entregas do projeto

Este arquivo registra as entregas de cada integrante e quando foram realizadas.
Atualize a seção correspondente sempre que uma nova etapa for concluída. Datas
devem usar o formato `DD/MM/AAAA`.

## Integrante 1 — Engenharia de dados e pipeline base

### 29/10/2026

- Baixou a base **Breast Cancer Wisconsin (Diagnostic)** do UCI e organizou os
  arquivos originais em `data/raw/`:
  - [`wdbc.data`](data/raw/wdbc.data)
  - [`wdbc.names`](data/raw/wdbc.names)
- Carregou os dados com nomes de colunas e verificou a estrutura: 569 registros,
  32 colunas, sem valores faltantes ou linhas/IDs duplicados.
- Investigou tipos, escalas e possíveis valores extremos. Os valores sinalizados
  foram tratados como pontos a investigar, não como erros a remover
  automaticamente.
- Exportou a versão processada para
  [`data/processed/wdbc_limpo.csv`](data/processed/wdbc_limpo.csv), codificando
  `diagnosis` como `0` (benigno) e `1` (maligno), sem normalizar as medidas.
- Preparou [`notebooks/refinador_de_dados.ipynb`](notebooks/refinador_de_dados.ipynb)
  para importar os dados brutos em `df` e apoiar as verificações iniciais.
- Preparou [`notebooks/analise.ipynb`](notebooks/analise.ipynb) para carregar os
  dados processados, separar `X` e `y` (retirando `id` de `X`), realizar a divisão
  estratificada treino/teste e apresentar exemplos de pipelines com e sem
  `StandardScaler`.
- Orientações compartilhadas com o grupo:
  - Usar `X` com os 30 atributos numéricos; `id` é identificador e não deve ser
    usado como preditor.
  - Usar `y` como `diagnosis`, com `0 = benigno` e `1 = maligno`.
  - Para modelos sensíveis à escala (por exemplo, regressão logística, KNN e
    SVM), treinar o pipeline completo com `X_train` e `y_train`. O scaler deve
    aprender os parâmetros somente no treino e então transformar validação/teste,
    evitando vazamento de dados.
  - Modelos baseados em árvores, como Random Forest, geralmente não precisam de
    padronização.
  - Esta base não requer separação de preditores categóricos e numéricos: os 30
    atributos de `X` são numéricos e `diagnosis` é o alvo.
  - A divisão preparada usa `test_size=0.2`, `stratify=y` e `random_state=42`;
    combinar com o grupo o uso da mesma divisão.

## Integrante 2 — Análise estatística e EDA visual

Sem registro de entrega neste log.

- **Entregas planejadas:** estatísticas descritivas por classe, visualizações,
  correlações e investigação de multicolinearidade.
- **Data da entrega:** a preencher.
- **Arquivos/observações:** a preencher.

## [Data: 04/11/2026] - Integrante 3: Redução de Dimensionalidade e Feature Selection
- **Status:** Concluído
- **Entregas:**
  - Inclusão de cabeçalhos e estrutura em Markdown no notebook `Feature_selection.ipynb`.
  - Estudo de agrupamento não-supervisionado (K-Means e GMM com Silhouette Score).
  - Projeções PCA (Scree Plot, 2D e 3D) e t-SNE 2D.
  - Seleção de atributos (Filtro de Colinearidade, ANOVA, Mutual Information e RFE).
  - Exportação dos datasets reduzidos (`wdbc_train_pca.csv` e `wdbc_train_top10.csv`) na pasta `data/processed/`.

## Integrante 4 — Modelagem preditiva e ajuste de modelos

### 18/11/2026

- Treinou e ajustou 7 classificadores em
  [`notebooks/modelagem_supervisionada.ipynb`](notebooks/modelagem_supervisionada.ipynb):
  Regressão Logística (L1/L2), KNN, SVM Linear, SVM RBF, Naive Bayes Gaussiano,
  Árvore de Decisão e Random Forest.
- Usou a mesma divisão treino/teste do Integrante 1 e do Integrante 3
  (`test_size=0.2`, `stratify=y`, `random_state=42`), conferida no notebook
  contra `X_train_scaled.csv` e `y_train.csv`. O conjunto de teste não foi
  usado: fica para a avaliação do Integrante 5.
- Busca de hiperparâmetros com `GridSearchCV` e `StratifiedKFold` (10 folds),
  métrica ROC-AUC, registrando também sensibilidade, especificidade, precisão,
  F1 e acurácia.
- Comparou 4 cenários, com toda transformação dentro do pipeline (ajustada só
  nos folds de treino): A = 30 atributos padronizados; B = PCA com 90% ou 95% da
  variância, escolhido na validação cruzada (o de 90% reproduz o
  `wdbc_train_pca.csv`); C = `SelectKBest` com *k* escolhido na validação
  cruzada; D = Top-10 não redundante do Integrante 3 (lista lida de
  `wdbc_train_top10.csv`).
- Mediu quanto a lista fixa do Top-10 favorece o cenário D na validação cruzada
  (regra do Integrante 3 refeita dentro de cada fold): o efeito é pequeno na
  Regressão Logística, nos SVMs e no Random Forest (até 0,002 de ROC-AUC), mas
  grande na Árvore de Decisão (0,025).
- Comparação estatística com validação cruzada repetida (10 folds × 3), teste t
  pareado com correção de Nadeau e Bengio e ajuste de Holm para comparações
  múltiplas.
- Resultado: ROC-AUC entre 0,96 e 0,997. Regressão Logística, SVM Linear e
  SVM RBF no topo; melhor média do SVM RBF no cenário D, mas nenhuma das outras
  27 combinações difere significativamente dela depois do ajuste. Reduzir a
  dimensão (B, C ou D) não muda significativamente o desempenho de nenhum
  modelo.
- Arquivos para o Integrante 5:
  - [`models/modelos_supervisionados.joblib`](models/modelos_supervisionados.joblib):
    os 28 pipelines ajustados (chave: `(modelo, cenário)`), com `predict_proba`;
  - [`results/hiperparametros_e_cv.csv`](results/hiperparametros_e_cv.csv);
  - [`results/ranking_cv_repetida.csv`](results/ranking_cv_repetida.csv).
- Observação para o Integrante 5: os pipelines recebem os **30 atributos
  originais, sem padronizar** (`wdbc_limpo.csv` com a mesma divisão), porque
  cada um já faz a própria padronização, PCA ou seleção. Não usar
  `X_test_scaled.csv` nem `wdbc_test_*.csv` com eles (padronização dupla).
  Exemplo de uso na Seção 5 do notebook.

## Integrante 5 — Avaliação, síntese e documentação

Sem registro de entrega neste log.

- **Entregas planejadas:** comparação de métricas, análise de falsos negativos,
  discussão, síntese e compilação final.
- **Data da entrega:** a preencher.
- **Arquivos/observações:** a preencher.

## Próximos registros

Para cada atualização, acrescente uma entrada com data, descrição objetiva do
que foi concluído e links/caminhos dos arquivos. “Sem registro” significa que
nenhuma entrega foi documentada aqui ainda; não indica que a pessoa não esteja
trabalhando.
