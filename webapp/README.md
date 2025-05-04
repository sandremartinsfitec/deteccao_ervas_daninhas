# WebApp de Detecção e Classificação de Ervas Daninhas em Soja

Este projeto apresenta um webapp interativo, desenvolvido em Streamlit, para classificação e localização de ervas daninhas em imagens de plantações de soja. O usuário pode fazer upload de qualquer imagem agrícola e obter:
- A classe predominante na imagem (`soybean`, `broadleaf`, `grass` ou `soil`)
- Probabilidades (softmax) para cada classe
- Tempo de inferência
- Heatmap de localização das áreas de ervas daninhas

O aplicativo utiliza um modelo PyTorch pré-treinado (`.pth`) e emprega o pipeline de pré-processamento e inferência do repositório original.

## Funcionalidades Principais
- Carregamento de modelo AlexNet ou ResNet50 pré-treinado
- Upload de múltiplas imagens (JPEG, PNG, TIFF)
- Pré-processamento (resize 128×128, normalização ImageNet)
- Classificação de imagem com softmax de probabilidades
- Exibição de label predito e confiança (%)
- Cálculo e exibição de tempo de inferência por imagem
- Geração de heatmap de localização via sliding-window
- Interface web responsiva e intuitiva

## Estrutura do Projeto
```
webapp/
├── app.py              # Script principal do Streamlit
├── requirements.txt    # Dependências Python
└── models/
    └── .gitkeep        # Pasta para checkpoints .pth
```

## Requisitos de Sistema
- Python 3.7 ou superior
- Streamlit
- PyTorch e torchvision
- Pillow
- NumPy
- Matplotlib
- OpenCV-Python

## Instalação
1. Acesse o diretório do webapp:
    webapp
2. (Opcional) Crie e ative um ambiente virtual:
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    .\.venv\Scripts\activate  # Windows
3. Instale as dependências:
    pip install -r requirements.txt

## Como Executar
No terminal, execute:
    streamlit run app.py

O Streamlit exibirá a URL local (ex: http://localhost:8501). Abra-a no navegador.

## Modelos Pré-Treinados
Coloque seus arquivos de checkpoint PyTorch em `webapp/models/` com nomes:
- alexnet.pth
- resnet50.pth

O app seleciona automaticamente o modelo de acordo com a opção na barra lateral.

## Uso
1. Selecione a arquitetura do modelo (AlexNet ou ResNet50) na barra lateral.
2. Faça upload de uma ou várias imagens.
3. Aguarde a inferência e visualize:
    - Label predito e confiança
    - Tempo de processamento
    - Gráfico de probabilidades
    - Heatmap de localização sobreposto

## Contribuição
1. Faça fork deste repositório.
2. Crie uma branch para sua feature: `git checkout -b minha-feature`.
3. Faça commit das alterações: `git commit -m 'feat: descrição da feature'`.
4. Envie para seu fork: `git push origin minha-feature`.
5. Abra um Pull Request.

---
Desenvolvido para apoiar práticas de agricultura de precisão, facilitando a detecção rápida de ervas daninhas em plantações de soja.