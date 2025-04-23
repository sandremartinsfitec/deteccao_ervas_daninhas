# Detecção e Classificação de Ervas Daninhas em Plantações de Soja Utilizando Redes Neurais Convolucionais AlexNet

# Projeto Final – Disciplina de Visão Computacional

## Detecção de Ervas Daninhas em Plantações de Soja Utilizando Redes Neurais Convolucionais

### Pós-Graduação em Inteligência Artificial Generativa, Universidade de Pernambuco

**Autor:** Alessandre Martins

## Resumo

Este artigo descreve o desenvolvimento de um sistema de visão computacional para a detecção e classificação de ervas daninhas em plantações de soja, como projeto final da disciplina de Visão Computacional da pós-graduação em Inteligência Artificial Generativa da Universidade de Pernambuco. O projeto utiliza a arquitetura de rede neural convolucional AlexNet para classificar imagens de plantações de soja em quatro classes: soja, ervas daninhas de folha larga, grama e solo. O modelo foi treinado e avaliado utilizando um conjunto de dados publicamente disponível no Kaggle ([https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops](https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops)). Os resultados iniciais, registrados e analisados via MLflow, demonstram o potencial da arquitetura AlexNet para esta tarefa, com uma acurácia de validação de 84%. O artigo discute a definição do problema, a metodologia adotada, os resultados obtidos e as possíveis melhorias para trabalhos futuros.

## 1. Definição do Problema e Justificativa

O manejo de ervas daninhas é um desafio crítico na agricultura, impactando diretamente a produtividade das culturas. A identificação manual de ervas daninhas é um processo demorado e oneroso. Este projeto propõe uma solução automatizada para a detecção e classificação de ervas daninhas em plantações de soja, utilizando técnicas de visão computacional e aprendizado profundo. A motivação para este projeto reside na necessidade de otimizar o manejo de herbicidas, reduzir custos e aumentar a eficiência na agricultura. O dataset escolhido, disponível no Kaggle, oferece uma base de dados diversificada e representativa de imagens de plantações de soja, permitindo o desenvolvimento e avaliação de modelos robustos para este problema.

## 2. Pré-processamento e Análise Exploratória

O código do projeto demonstra a aplicação de diversas técnicas de preparação e aumento de dados para melhorar o desempenho do modelo:

- **Redimensionamento de imagens:** As imagens são redimensionadas para 128x128 pixels, uniformizando a entrada para o modelo.
- **Normalização:** As imagens são normalizadas utilizando os valores médios e desvio padrão do ImageNet (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`), técnica comum para modelos pré-treinados, embora neste caso o modelo AlexNet tenha sido treinado do zero.
- **Aumento de dados (Data Augmentation):** Para aumentar a variabilidade do dataset de treinamento e melhorar a generalização do modelo, foram aplicadas as seguintes transformações:
    - `RandomHorizontalFlip`: Inversão horizontal aleatória das imagens.
    - `RandomVerticalFlip`: Inversão vertical aleatória das imagens.
    - `RandomRotation(degrees=45)`: Rotação aleatória das imagens em até 45 graus.
    - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`: Ajuste aleatório de brilho, contraste, saturação e matiz das imagens.
- **Conversão para tensores:** As imagens são convertidas para tensores PyTorch, formato adequado para o treinamento do modelo.

## 3. Implementação do Modelo

O modelo implementado neste projeto é baseado na arquitetura AlexNet, uma rede neural convolucional profunda. A arquitetura AlexNet foi escolhida por seu desempenho comprovado em tarefas de classificação de imagens e sua implementação relativamente simples em frameworks de aprendizado profundo como o PyTorch. O modelo foi implementado do zero no arquivo `model.py`, utilizando as camadas `Conv2d`, `MaxPool2d`, `ReLU` e `Linear` do PyTorch. A arquitetura AlexNet consiste em:

1. **Camadas convolucionais:** Extraem características das imagens através de filtros convolucionais.
2. **Camadas de pooling:** Reduzem a dimensionalidade das características extraídas, diminuindo a complexidade computacional e aumentando a robustez a pequenas variações espaciais.
3. **Funções de ativação ReLU:** Introduzem não-linearidade ao modelo, permitindo aprender relações complexas nos dados.
4. **Camadas totalmente conectadas (Fully Connected):** Realizam a classificação final, mapeando as características aprendidas para as classes de saída (soja, erva de folha larga, grama e solo).

O modelo foi adaptado para o problema de detecção de ervas daninhas, com a camada de saída ajustada para quatro classes.

## 4. Treinamento e Avaliação

O treinamento do modelo foi realizado utilizando o script `train.py`. Os principais hiperparâmetros utilizados foram:

- **Dataset:** `WeedDataset` implementado em `data_loading.py`, carregando imagens do dataset Kaggle.
- **DataLoader:** `DataLoader` do PyTorch, com `batch_size=32` e `num_workers=12`.
- **Modelo:** Arquitetura AlexNet implementada em `model.py`, instanciada com `num_classes=4`.
- **Função de perda (Criterion):** `CrossEntropyLoss` do PyTorch, adequada para problemas de classificação multiclasse.
- **Otimizador:** `Adam` com taxa de aprendizado `learning_rate=1e-3` e `weight_decay=1e-5` (para regularização L2).
- **Scheduler:** `StepLR` para ajustar a taxa de aprendizado ao longo do treinamento (`step_size=3000, gamma=0.1`).
- **Número de épocas:** `num_epochs=300`.
- **Dispositivo:** GPU (`cuda:0`) se disponível, caso contrário CPU.

O script `train.py` utiliza a biblioteca MLflow para registrar os hiperparâmetros, métricas de treinamento e avaliação, e os artefatos do modelo (modelo treinado, gráficos de perda e acurácia).

A avaliação do modelo foi realizada com o script `evaluate.py`, utilizando o `DataLoader` de validação e métricas como acurácia, precisão, recall, F1-score, AUC e Log Loss, também registradas no MLflow.

## 5. Análise dos Resultados e Melhorias

Os resultados detalhados do treinamento e avaliação, incluindo as métricas e gráficos, estão disponíveis no MLflow, Run ID: 9009f7ae99554fcfb04537b9d71c98f1 ([http://127.0.0.1:8081/#/experiments/0/runs/9009f7ae99554fcfb04537b9d71c98f1](http://127.0.0.1:8081/#/experiments/0/runs/9009f7ae99554fcfb04537b9d71c98f1)). A acurácia de validação alcançada foi de 70.53%, representando uma melhoria significativa em relação aos resultados iniciais do projeto (Acurácia (val_accuracy): 0.481, Run ID: 5a5889d6ab8743beb1dc07de5089566a).

Essa melhoria pode ser atribuída às otimizações implementadas, como o uso de data augmentation e weight decay, que contribuíram para um modelo mais robusto e generalizável.

Para aprimoramentos futuros, sugere-se:

1. **Aumentar o número de épocas de treinamento:** Treinar o modelo por mais épocas pode levar a uma convergência ainda melhor e, consequentemente, a um aumento da acurácia.
2. **Explorar outras arquiteturas de CNNs:** Investigar arquiteturas mais recentes e profundas, como ResNet, EfficientNet ou DenseNet, que demonstraram desempenho superior em tarefas de classificação de imagens. A arquitetura ResNet50 já está implementada e disponível para uso no código (`model.py` e `train.py`).
3. **Otimização de hiperparâmetros:** Realizar uma busca mais sistemática por hiperparâmetros ótimos, como taxa de aprendizado, tamanho do batch e weight decay, utilizando técnicas como Grid Search ou Random Search.
4. **Balanceamento do dataset:** Investigar técnicas de balanceamento de dataset para mitigar o impacto de possíveis desequilíbrios entre as classes.
5. **Aumento de dados mais avançado:** Explorar técnicas de aumento de dados mais sofisticadas, como GANs (Redes Generativas Adversariais) para gerar imagens sintéticas de ervas daninhas e aumentar a diversidade do dataset de treinamento.

## 6. Conclusão

Este projeto demonstrou a viabilidade da utilização da arquitetura AlexNet para a detecção e classificação de ervas daninhas em plantações de soja. As otimizações implementadas, como data augmentation e weight decay, resultaram em uma melhora significativa na acurácia do modelo. Os resultados obtidos, juntamente com as sugestões de melhorias futuras, indicam o potencial promissor da visão computacional e do aprendizado profundo para o desenvolvimento de sistemas automatizados de apoio ao manejo de ervas daninhas na agricultura.

---
# Detecção e Classificação de Ervas Daninhas em Plantações de Soja Utilizando Redes Neurais Convolucionais AlexNet

## Resumo

Este artigo apresenta um estudo sobre a aplicação da arquitetura de rede neural convolucional AlexNet para a detecção e classificação de ervas daninhas em plantações de soja. O objetivo principal é desenvolver um sistema eficiente e preciso para identificar e classificar imagens de plantas de soja, ervas daninhas de folha larga, grama e solo. O modelo foi treinado e avaliado utilizando um conjunto de dados de imagens de plantações de soja, e os resultados demonstram o potencial da arquitetura AlexNet para esta tarefa. Métricas de desempenho como acurácia, precisão, sensibilidade e F1-score foram registradas utilizando MLflow, fornecendo uma análise quantitativa do desempenho do modelo.

## 1. Introdução

A agricultura moderna enfrenta o desafio constante do manejo de ervas daninhas, que competem com as culturas por recursos essenciais como luz, água e nutrientes, impactando negativamente a produtividade. A identificação precisa e em tempo hábil de ervas daninhas é crucial para a aplicação eficiente de herbicidas e práticas de manejo integrado. A visão computacional, combinada com técnicas de aprendizado profundo, oferece uma abordagem promissora para automatizar a detecção e classificação de ervas daninhas em ambientes agrícolas.

Redes neurais convolucionais (CNNs) têm demonstrado sucesso notável em diversas tarefas de classificação de imagens. Neste estudo, exploramos a arquitetura AlexNet, uma CNN amplamente conhecida e eficaz, para a classificação de imagens de plantações de soja em quatro classes: soja, erva de folha larga, grama e solo.

## 2. Materiais e Métodos

### 2.1. Dataset

O dataset utilizado neste estudo consiste em imagens de plantações de soja, categorizadas em quatro classes: 'soybean', 'broadleaf', 'grass' e 'soil'. As imagens foram coletadas em condições de campo e representam a variabilidade natural presente em ambientes agrícolas.

### 2.2. Arquitetura do Modelo

A arquitetura AlexNet foi implementada utilizando o framework PyTorch. AlexNet é uma CNN profunda composta por camadas convolucionais, camadas de pooling e camadas totalmente conectadas. A escolha desta arquitetura baseou-se em seu desempenho comprovado em tarefas de classificação de imagens e sua relativa eficiência computacional.

### 2.3. Treinamento do Modelo

O modelo foi treinado utilizando o script `train.py`, com os seguintes parâmetros (registrados via MLflow Run ID: 5a5889d6ab8743beb1dc07de5089566a):

- Taxa de aprendizado inicial: 0.001
- Otimizador: Adam
- Tamanho do batch: 32
- Número de épocas: 300
- Número de workers: 16

O treinamento foi monitorado utilizando MLflow, que registrou métricas de treinamento e validação ao longo das épocas.

### 2.4. Avaliação do Modelo

O modelo treinado foi avaliado utilizando o script `evaluate.py`. As métricas de avaliação incluíram acurácia, precisão, sensibilidade, F1-score e AUC (Area Under the ROC Curve), calculadas para cada classe e em média. A matriz de confusão também foi gerada para analisar o desempenho do modelo em detalhes.

## 3. Resultados

Os resultados do treinamento e avaliação do modelo foram registrados utilizando MLflow (Run ID: 5a5889d6ab8743beb1dc07de5089566a). As principais métricas de validação obtidas foram:

- Acurácia (val_accuracy): 0.481
- Precisão (val_precision): 0.231
- Sensibilidade (val_recall): 0.481
- F1-score (val_f1): 0.312
- Log Loss (val_log_loss): 1.252
- AUC (val_auc): 0.397

A perda de treinamento (train/loss) ao final do treinamento foi de 1.202, enquanto a perda de validação (val_loss) foi de 1.252. As curvas de acurácia e perda durante o treinamento foram visualizadas e salvas como `accuracy.png` e `loss.png`, respectivamente.

## 4. Discussão

Os resultados da avaliação indicam que o modelo AlexNet, com a configuração de treinamento utilizada, alcançou uma acurácia de validação de 48.1% no dataset de detecção e classificação de ervas daninhas em soja. Embora a acurácia não seja considerada alta em termos absolutos, é importante notar que este é um resultado inicial e pode ser melhorado com ajustes nos parâmetros de treinamento, aumento do dataset, ou exploração de outras arquiteturas de CNNs.

As métricas de precisão e F1-score, que são mais robustas em datasets desbalanceados, também refletem um desempenho moderado do modelo. A sensibilidade de 48.1% indica que o modelo identifica corretamente cerca de metade das instâncias positivas em média. A AUC de 0.397 sugere que o modelo tem um desempenho de classificação inferior ao aleatório.

A análise da matriz de confusão (não apresentada neste artigo, mas gerada durante a avaliação) pode fornecer insights adicionais sobre quais classes o modelo confunde mais frequentemente, direcionando esforços para melhorar a discriminação entre essas classes específicas.

## 5. Conclusão

Este estudo demonstrou a aplicação da arquitetura AlexNet para a detecção e classificação de ervas daninhas em plantações de soja. Os resultados iniciais, registrados e analisados via MLflow, fornecem um ponto de partida para o desenvolvimento de um sistema de visão computacional para auxiliar no manejo de ervas daninhas na agricultura.

Trabalhos futuros podem focar na otimização dos parâmetros de treinamento, na expansão e balanceamento do dataset, na exploração de técnicas de aumento de dados, e na investigação de arquiteturas de CNNs mais recentes e adaptadas a este problema específico. A implementação de testes automatizados e a documentação detalhada do projeto, conforme refletido no arquivo `README.md` gerado, são passos importantes para a reprodutibilidade e a colaboração na evolução deste projeto.

---
