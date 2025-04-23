# Detecção e Classificação de Ervas Daninhas em Plantações de Soja

## 1. Descrição do Projeto

Este projeto tem como objetivo desenvolver um sistema de detecção e classificação de ervas daninhas em plantações de soja utilizando técnicas de visão computacional e aprendizado profundo. O sistema é capaz de identificar e classificar imagens em quatro classes distintas: 'soybean' (soja), 'broadleaf' (erva de folha larga), 'grass' (grama) e 'soil' (solo). A arquitetura de rede neural convolucional AlexNet foi implementada para realizar a classificação das imagens.

**Tecnologias Principais:**

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 2. Arquitetura e Estrutura do Código

A estrutura do código é organizada em diretórios e arquivos principais para facilitar a manutenção e o desenvolvimento:

- **`dataset/`**: Contém as imagens do dataset, divididas em pastas por classe ('broadleaf', 'grass', 'soil', 'soybean').
- **`mlruns/` e `mlartifacts/`**: Pastas geradas pelo MLflow para rastreamento de experimentos e artefatos do modelo.
- **`src/`**: Contém o código fonte do projeto:
    - `data_loading.py`: Responsável pelo carregamento e pré-processamento dos dados.
    - `model.py`: Define a arquitetura do modelo AlexNet.
    - `train.py`: Script para treinar o modelo.
    - `evaluate.py`: Script para avaliar o modelo treinado.
    - `visualize.py`: Script para visualizar métricas de treinamento e avaliação.
- **`accuracy.png` e `loss.png`**: Imagens geradas durante o treinamento para visualização da acurácia e perda.
- **`README.md`**: Documentação do projeto (este arquivo).
- **`requirements.txt`**: Lista de dependências Python.
- **`run.sh`**: Script shell para executar o treinamento.

## 3. Requisitos e Dependências

Antes de executar o projeto, certifique-se de ter o Python instalado e as seguintes dependências:

```
torch
torchvision
opencv-python
numpy
pandas
matplotlib
scikit-learn
```

Para instalar as dependências, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

## 4. Instalação e Configuração

1. **Clone o repositório:**
   ```bash
   git clone [URL do repositório]
   cd detecção_ervas_daninhas
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## 5. Uso do Projeto

### 5.1. Treinamento do Modelo

Para treinar o modelo, execute o script `train.py`:

```bash
python src/train.py
```

Este script irá iniciar o treinamento do modelo AlexNet utilizando o dataset presente na pasta `dataset/`. As métricas de treinamento (acurácia e perda) serão visualizadas durante o processo e salvas em arquivos `.png`.

### 5.2. Avaliação do Modelo

Após o treinamento, o script avalia o modelo utilizando o script `evaluate.py`:

Este script irá carregar o modelo treinado e calcular métricas de avaliação como acurácia, precisão, sensibilidade e matriz de confusão. Os resultados detalhados serão salvos em arquivos `.csv` na pasta raiz do projeto.

### 5.3. Visualização dos Resultados

Para visualizar as curvas de acurácia e perda geradas durante o treinamento, execute o script `visualize.py`:

```bash
python src/visualize.py
```

Este script irá exibir as imagens `accuracy.png` e `loss.png` que mostram a evolução do treinamento.

## 6. Execução e Testes

Não há testes automatizados implementados neste projeto. A verificação da funcionalidade é realizada através da avaliação do modelo treinado e da visualização das métricas de desempenho.

## 7. Documentação da pesquisa

/detecção_ervas_daninhas/scientific_article.md