# -*- coding: utf-8 -*-
"""
Webapp Streamlit para Classificação de Imagens Agrícolas (Soja e Ervas Daninhas)
"""
# Fechamento de blocos para evitar SyntaxError por indentação aberta
if False:
    pass
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 # Para processamento de imagem (heatmap, bounding box)
import time
import io
import os
import io

st.set_page_config(layout="wide", page_title="Classificador Agrícola")

# --- Configurações e Constantes ---
MODEL_PATH = "../models/model.pth"
IMAGE_DIR = "../images"
NUM_CLASSES = 4 # soybean, broadleaf, grass, soil
CLASS_NAMES = {0: "Soja", 1: "Folha Larga", 2: "Grama", 3: "Solo"}
WEED_CLASSES = {1, 2} # Índices das classes consideradas ervas daninhas (Folha Larga, Grama)
INPUT_SIZE = 128 # Tamanho da imagem de entrada para o modelo

# --- Funções Auxiliares ---

# Cache do modelo para evitar recarregamento
@st.cache_resource
def load_model_and_device(model_path):
    """Carrega o modelo ResNet50 pré-treinado e define o dispositivo."""
    try:
        model = resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

        # Define dispositivo e carrega checkpoint
        if torch.cuda.is_available():
            device = torch.device("cuda")
            st.sidebar.success("GPU (CUDA) disponível! Usando GPU.")
            raw_state = torch.load(model_path)
        else:
            device = torch.device("cpu")
            st.sidebar.warning("GPU (CUDA) não disponível. Usando CPU.")
            raw_state = torch.load(model_path, map_location=device)
        # Ajusta possíveis prefixos 'resnet50.' no state_dict
        new_state = {}
        for k, v in raw_state.items():
            if k.startswith("resnet50."):
                new_key = k.split("resnet50.", 1)[1]
            else:
                new_key = k
            new_state[new_key] = v
        # Carrega estado ajustado
        model.load_state_dict(new_state)

        model.to(device)
        model.eval()
        st.sidebar.success(f"Modelo 	'{os.path.basename(model_path)}' carregado em {device}.")
        return model, device
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo não encontrado em 	'{model_path}'.")
        return None, None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None

def preprocess_patch(patch_pil):
    """Pré-processa um patch (recorte) da imagem."""
    preprocess_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess_transform(patch_pil)
    batch_tensor = torch.unsqueeze(img_tensor, 0)
    return batch_tensor

def preprocess_image(image_bytes):
    """
    Converte bytes de imagem em PIL.Image e retorna (tensor, pil_image).
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    batch_tensor = preprocess_patch(pil_img)
    return batch_tensor, pil_img

def predict_patch(model, device, patch_tensor):
    """Realiza a inferência em um patch."""
    try:
        patch_tensor = patch_tensor.to(device)
        with torch.no_grad():
            outputs = model(patch_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class_idx = torch.max(probabilities, 0)
        return probabilities.cpu().numpy(), predicted_class_idx.item(), confidence.item()
    except Exception as e:
        print(f"Erro na inferência do patch: {e}")
        return None, None, None

# --- Lógica de Heatmap (CAM - Class Activation Mapping) ---
feature_maps = {}
def get_features_hook(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook

def generate_heatmap_cam(model, device, image_tensor, original_pil_image, target_class_idx):
    """Gera um heatmap usando Class Activation Mapping (CAM)."""
    handle = None
    try:
        handle = model.layer4.register_forward_hook(get_features_hook('layer4'))
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor)
        if handle: handle.remove()

        fc_weights = model.fc.weight[target_class_idx, :].detach().cpu().numpy()
        last_conv_features = feature_maps.get('layer4')
        if last_conv_features is None: return None
        last_conv_features = last_conv_features.squeeze(0).cpu().numpy()

        cam = np.zeros(last_conv_features.shape[1:], dtype=np.float32)
        for i, w in enumerate(fc_weights):
            cam += w * last_conv_features[i, :, :]

        cam = cv2.resize(cam, (original_pil_image.width, original_pil_image.height))
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0: cam = cam / np.max(cam)
        else: return None

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        original_cv_image = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        if heatmap_colored.shape != original_cv_image.shape:
             heatmap_colored = cv2.resize(heatmap_colored, (original_cv_image.shape[1], original_cv_image.shape[0]))

        superimposed_img = cv2.addWeighted(original_cv_image, 0.6, heatmap_colored, 0.4, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        return superimposed_img_rgb
    except Exception as e:
        st.error(f"Erro ao gerar heatmap CAM: {e}")
        return None
    finally:
        if handle: 
            try: handle.remove()
            except: pass
        if 'layer4' in feature_maps: del feature_maps['layer4']

# --- Lógica de Sliding Window ---
def run_sliding_window(model, device, original_pil_image, window_size=INPUT_SIZE, stride=INPUT_SIZE//2, confidence_threshold=0.7):
    """Executa a janela deslizante e retorna as detecções e o tempo."""
    width, height = original_pil_image.size
    detections = []
    total_patches = 0
    weed_patches = 0

    progress_bar = st.progress(0, text="Processando janelas deslizantes...")
    num_steps_x = (width - window_size) // stride + 1
    num_steps_y = (height - window_size) // stride + 1
    total_steps = max(1, num_steps_x * num_steps_y) # Evitar divisão por zero
    current_step = 0

    start_time_sw = time.time()

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            total_patches += 1
            current_step += 1
            patch = original_pil_image.crop((x, y, x + window_size, y + window_size))
            patch_tensor = preprocess_patch(patch)
            probs, class_idx, conf = predict_patch(model, device, patch_tensor)

            if class_idx in WEED_CLASSES and conf >= confidence_threshold:
                weed_patches += 1
                detections.append([x, y, x + window_size, y + window_size, conf, class_idx]) # Usar lista para NMS

            progress_percent = min(1.0, current_step / total_steps)
            progress_bar.progress(progress_percent, text=f"Processando janelas deslizantes... ({current_step}/{total_steps})")

    end_time_sw = time.time()
    processing_time_sw = end_time_sw - start_time_sw
    progress_bar.empty()

    st.write(f"- Janelas processadas: {total_patches}")
    st.write(f"- Detecções de ervas daninhas (conf > {confidence_threshold*100:.1f}%): {weed_patches}")
    st.write(f"- Tempo de processamento (Janela Deslizante): {processing_time_sw:.2f}s")

    return detections, processing_time_sw

def draw_sliding_window_overlay(original_pil_image, detections):
    """Desenha a sobreposição da janela deslizante com base nas detecções."""
    overlay_image = original_pil_image.copy()
    draw = ImageDraw.Draw(overlay_image)
    for det in detections:
        x1, y1, x2, y2, conf, class_idx = det
        color = (255, 0, 0) if class_idx == 1 else (255, 165, 0) # Vermelho para Folha Larga, Laranja para Grama
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    del draw
    return overlay_image

# --- Lógica de Bounding Box (com NMS) ---
def non_max_suppression(boxes, scores, iou_threshold):
    """Aplica Non-Maximum Suppression simples."""
    if not boxes:
        return []

    # Converter para numpy array
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # Ordenar por score decrescente

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calcular IoU com as caixas restantes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Manter apenas caixas com IoU abaixo do threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1] # +1 porque comparamos com order[1:]

    return keep

def generate_bounding_box_overlay(original_pil_image, detections, confidence_threshold=0.8, iou_threshold=0.1):
    """Gera uma sobreposição com bounding boxes agregadas usando NMS."""
    try:
        overlay_image = original_pil_image.copy()
        draw = ImageDraw.Draw(overlay_image)

        # Filtrar detecções pelo threshold de BBox
        filtered_detections = [d for d in detections if d[4] >= confidence_threshold]

        if not filtered_detections:
            st.info("Nenhuma detecção acima do threshold para Bounding Box.")
            return overlay_image

        # Preparar dados para NMS
        boxes = [[d[0], d[1], d[2], d[3]] for d in filtered_detections]
        scores = [d[4] for d in filtered_detections]
        class_indices = [d[5] for d in filtered_detections]

        start_time_nms = time.time()
        # Aplicar NMS
        keep_indices = non_max_suppression(boxes, scores, iou_threshold)
        end_time_nms = time.time()

        final_boxes = [filtered_detections[i] for i in keep_indices]

        st.write(f"- Detecções após NMS (IoU < {iou_threshold}): {len(final_boxes)}")
        st.write(f"- Tempo de processamento (NMS): {end_time_nms - start_time_nms:.4f}s")

        # Desenhar Bounding Boxes finais
        # Tentar carregar uma fonte
        try:
            font = ImageFont.truetype("arial.ttf", 15) # Pode precisar instalar fontes ou usar um caminho padrão
        except IOError:
            font = ImageFont.load_default()

        for box_data in final_boxes:
            x1, y1, x2, y2, conf, class_idx = box_data
            label = CLASS_NAMES.get(class_idx, "Desconhecido")
            color = (255, 0, 0) if class_idx == 1 else (0, 255, 0) # Vermelho para Folha Larga, Verde para Grama
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{label}: {conf:.2f}"
            # Obter tamanho do texto para posicionar o fundo
            # text_bbox = draw.textbbox((x1, y1 - 15), text, font=font)
            # draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 15), text, fill=color, font=font)

        del draw
        return overlay_image

    except Exception as e:
        st.error(f"Erro ao gerar overlay de bounding box: {e}")
        return None

# Alias para compatibilidade: use generate_bbox_overlay onde generate_bounding_box_overlay é chamado
generate_bbox_overlay = generate_bounding_box_overlay

# --- Função Principal de Predição (Imagem Completa) ---
def predict(model, device, image_tensor):
    """Realiza a inferência na imagem completa."""
    try:
        start_time = time.time()
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class_idx = torch.max(probabilities, 0)
        end_time = time.time()
        inference_time = end_time - start_time
        return probabilities.cpu().numpy(), predicted_class_idx.item(), confidence.item(), inference_time
    except Exception as e:
        st.error(f"Erro durante a inferência: {e}")
        return None, None, None, None

# --- Interface Streamlit ---
st.title("🌿 Classificador de Imagens Agrícolas")
st.write("Faça upload de imagens de lavouras (soja, ervas daninhas, solo) para classificação e localização aproximada de ervas daninhas.")

# Carregar modelo na barra lateral
st.sidebar.header("Configurações")
model, device = load_model_and_device(MODEL_PATH)

# Parâmetros de Localização na Sidebar
st.sidebar.subheader("Parâmetros de Localização")
loc_method = st.sidebar.radio("Método de Localização", ["Heatmap (CAM)", "Janela Deslizante", "Bounding Box"], index=0)

conf_threshold_sw = st.sidebar.slider("Threshold Confiança (Janela Deslizante)", 0.1, 1.0, 0.7, 0.05)
window_size_sw = INPUT_SIZE # Fixo por enquanto
stride_sw = INPUT_SIZE // 2 # Fixo por enquanto
conf_threshold_bbox = st.sidebar.slider("Threshold Confiança (Bounding Box)", 0.1, 1.0, 0.8, 0.05)
iou_threshold_bbox = st.sidebar.slider("Threshold IoU (Bounding Box NMS)", 0.0, 1.0, 0.1, 0.05)

if model:
    # Opção de Upload
    uploaded_files = st.file_uploader(
        "Escolha uma ou mais imagens...",
        type=["jpg", "jpeg", "png", "tiff"],
        accept_multiple_files=True
    )

    # Armazenar detecções da janela deslizante para reutilizar no BBox
    if 'sw_detections' not in st.session_state:
        st.session_state.sw_detections = {}

    if uploaded_files:
        st.info(f"{len(uploaded_files)} imagem(ns) carregada(s). Processando...")

        # Processar cada imagem
        for idx, uploaded_file in enumerate(uploaded_files):
            st.divider()
            st.subheader(f"Análise da Imagem: {uploaded_file.name}")
            # Gera ID único para esta imagem (índice + nome)
            file_id = f"{idx}_{uploaded_file.name}"  # Usar índice e nome do arquivo para estado

            col1, col2 = st.columns(2)

            image_bytes = uploaded_file.getvalue()
            image_tensor_full, original_pil_image = preprocess_image(image_bytes)

            if image_tensor_full is not None and original_pil_image is not None:
                probabilities, predicted_class_idx, confidence, inference_time = predict(model, device, image_tensor_full)

                if probabilities is not None:
                    predicted_label = CLASS_NAMES.get(predicted_class_idx, "Desconhecido")

                    with col1:
                        st.image(original_pil_image, caption="Imagem Original", use_column_width=True)

                    with col2:
                        st.metric(label="Classe Predita (Geral)", value=f"{predicted_label}", delta=f"{confidence*100:.2f}% Confiança")
                        st.metric(label="Tempo Inferência (Geral)", value=f"{inference_time:.4f} segundos")
                        st.write("**Probabilidades por Classe (%):**")
                        prob_data = {CLASS_NAMES.get(i, f"Classe {i}"): prob * 100 for i, prob in enumerate(probabilities)}
                        st.bar_chart(prob_data)

                    # --- Localização --- #
                    st.subheader(f"Localização Aproximada via {loc_method}")
                    loc_placeholder = st.empty() # Placeholder para a imagem de localização
                    info_placeholder = st.empty() # Placeholder para informações de processamento

                    with info_placeholder.container():
                        if loc_method == "Heatmap (CAM)":
                            # CAM: forward hook + fc weights → heatmap
                            heatmap_opts = {name: idx for idx, name in CLASS_NAMES.items() if idx in WEED_CLASSES}
                            default = predicted_class_idx if predicted_class_idx in WEED_CLASSES else list(heatmap_opts.values())[0]
                            default_name = [n for n,i in heatmap_opts.items() if i==default][0]
                            sel = st.selectbox("Classe para Heatmap:", list(heatmap_opts.keys()),
                                            index=list(heatmap_opts.keys()).index(default_name), key=f"hm_{file_id}")
                            cls = heatmap_opts[sel]
                            hm_img = generate_heatmap_cam(model, device, image_tensor_full, original_pil_image, cls)
                            if hm_img is not None:
                                loc_placeholder.image(hm_img, caption="Heatmap (CAM)", use_column_width=True)
                            else:
                                st.error("Falha no Heatmap")

                        elif loc_method == "Janela Deslizante":
                            dets, t_sw = run_sliding_window(model, device, original_pil_image,
                                                            window_size_sw, stride_sw, conf_threshold_sw)
                            sw_img = draw_sliding_window_overlay(original_pil_image, dets)
                            loc_placeholder.image(sw_img, caption="Sliding Window", use_column_width=True)
                            st.session_state.sw_detections[file_id] = dets
                            st.write(f"Tempo SW: {t_sw:.2f}s")

                        elif loc_method == "Bounding Box":
                            dets = st.session_state.sw_detections.get(file_id)
                            if not dets:
                                st.warning("Execute Janela Deslizante antes de BBox")
                            else:
                                bb_img = generate_bbox_overlay(original_pil_image, dets,
                                                            conf_threshold_bbox, iou_threshold_bbox)
                                if bb_img is not None:
                                    loc_placeholder.image(bb_img, caption="Bounding Box", use_column_width=True)
                                else:
                                    st.error("Falha no Bounding Box")


               