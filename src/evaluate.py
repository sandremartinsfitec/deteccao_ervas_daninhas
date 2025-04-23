import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_loading import WeedDataset
from src.model import get_model
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
import logging
import numpy as np


def evaluate_model(model, val_loader, device="cpu"):
    """
    Avalia o modelo usando o DataLoader fornecido e registra métricas no MLflow.
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_predicted = [] # Store probabilities for AUC and log_loss
    all_predictions = [] # Store predicted labels for precision, recall, f1, accuracy
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Evaluating', leave=False)
        for images, labels in progress_bar:
            images = images.to(device) # Move validation images to device
            labels = labels.to(device) # Move validation labels to device
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1) # Get probabilities
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_predicted.extend(probabilities.cpu().numpy()) # Store probabilities
            all_predictions.extend(predicted.cpu().numpy()) # Store predicted labels
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct_predictions / total_samples
    all_predicted_np = np.array(all_predicted)
    all_predictions_np = np.array(all_predictions) # Convert predicted labels to numpy array
    all_labels_np = np.array(all_labels)

    precision = precision_score(all_labels_np, all_predictions_np, average='weighted') # Use all_predictions_np
    recall = recall_score(all_labels_np, all_predictions_np, average='weighted') # Use all_predictions_np
    f1 = f1_score(all_labels_np, all_predictions_np, average='weighted') # Use all_predictions_np
    auc = roc_auc_score(all_labels_np, all_predicted_np, multi_class='ovr') # Use probabilities
    log = log_loss(all_labels_np, all_predicted_np) # Use probabilities

    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_auc", auc)
    mlflow.log_metric("val_log_loss", log)

    print(f'Accuracy on the validation set: {accuracy:.4f}')
    print(f'Precision on the validation set: {precision:.4f}')
    print(f'Recall on the validation set: {recall:.4f}')
    return accuracy

if __name__ == '__main__':
    # Define device
    device = torch.device("cuda:0") # Use GPU 0 if available, otherwise will error out
    logging.info(f"Using device for evaluation: {device}")

    # Load a trained model from MLflow
    # Replace <RUN_ID> with the actual Run ID from MLflow
    model_uri = "runs:/<RUN_ID>/alexnet_model"
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device) # Move model to device

    # Load validation dataset
    val_dataset = WeedDataset('dataset', train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    evaluate_model(model, val_loader, device)
