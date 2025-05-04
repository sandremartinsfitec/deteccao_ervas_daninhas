import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

# Ajusta PYTHONPATH para importar o modelo treinado
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import get_model

# Classes do modelo
CLASS_NAMES = ['soybean', 'broadleaf', 'grass', 'soil']

# Transformação de pré-processamento (mesma do treino)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache(allow_output_mutation=True)
def load_model(arch: str, device: torch.device):
    """
    Carrega o modelo pré-treinado de acordo com a arquitetura escolhida.
    Espera-se encontrar arquivo em 'models/{arch}.pth'.
    """
    # Cria instância do modelo
    model = get_model(arch, num_classes=len(CLASS_NAMES))
    # Caminho do checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), 'models', f'{arch}.pth')
    # Carrega state_dict
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def generate_heatmap(model, pil_img, device, win=128, stride=64):
    """
    Gera mapa de calor de probabilidades da classe predita usando sliding window.
    Retorna heatmap normalizado e índice da classe com maior probabilidade.
    """
    img_w, img_h = pil_img.size
    heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    counts = np.zeros((img_h, img_w), dtype=np.float32)
    # Slide window
    for top in range(0, img_h - win + 1, stride):
        for left in range(0, img_w - win + 1, stride):
            crop = pil_img.crop((left, top, left + win, top + win))
            inp = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            # Classe com maior probabilidade nesta janela
            cls = np.argmax(probs)
            p = probs[cls]
            # Acumula
            heatmap[top:top+win, left:left+win] += p
            counts[top:top+win, left:left+win] += 1
    # Evita divisão por zero
    counts[counts == 0] = 1
    heatmap = heatmap / counts
    # Classe global predita (maior média)
    global_cls = np.argmax([np.mean(heatmap)]) if False else None
    # Mas melhor usar última pred
    return heatmap

def main():
    st.title("Detecção de Ervas Daninhas em Soja")
    st.markdown("## Classificação e localização de ervas daninhas")

    # Seleção de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f"**Device:** {device}")
    # Seleção de arquitetura
    arch = st.sidebar.selectbox("Modelo", ['alexnet', 'resnet50'])

    # Carrega modelo
    with st.spinner(f"Carregando modelo {arch}..."):
        model = load_model(arch, device)

    # Upload de imagens
    uploaded = st.file_uploader("Faça upload de imagens (JPEG, PNG, TIFF)",
                                 type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                                 accept_multiple_files=True)
    if not uploaded:
        st.info("Aguardando upload de pelo menos uma imagem...")
        return

    for file in uploaded:
        # Leitura da imagem
        pil_img = Image.open(file).convert('RGB')
        st.subheader(f"Imagem: {file.name}")
        st.image(pil_img, use_column_width=True)

        # Inferência
        start = time.time()
        inp = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        elapsed = time.time() - start
        pred_idx = np.argmax(probs)
        pred_label = CLASS_NAMES[pred_idx]
        pred_conf = probs[pred_idx]

        # Exibição dos resultados
        st.write(f"**Predição:** {pred_label} ({pred_conf*100:.2f}%)")
        st.write(f"**Tempo de inferência:** {elapsed*1000:.1f} ms")

        # Gráfico de probabilidades
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probs * 100, color='skyblue')
        ax.set_ylabel('Probabilidade (%)')
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        # Heatmap de localização
        st.subheader("Heatmap de localização")
        with st.spinner("Gerando heatmap..."):
            heatmap = generate_heatmap(model, pil_img, device)
        # Exibe overlay
        fig2, ax2 = plt.subplots()
        ax2.imshow(pil_img)
        ax2.imshow(heatmap, cmap='jet', alpha=0.5)
        ax2.axis('off')
        st.pyplot(fig2)

if __name__ == '__main__':
    main()