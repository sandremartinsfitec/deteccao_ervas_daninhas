import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import get_model, AlexNet
from src.data_loading import WeedDataset
from src.evaluate import evaluate_model
from torch.optim.lr_scheduler import StepLR
import logging
from tqdm import tqdm
import mlflow
import argparse
import mlflow.pytorch
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global lists to store losses and accuracies for plotting
epoch_losses = []
val_accuracies = []

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device="cpu"):
    """
    Treina o modelo usando o DataLoader fornecido e o otimizador.
    """
    logging.info("Starting training")
    # Registra e salva o melhor modelo localmente
    best_accuracy = float('-inf')
    best_epoch = -1
    os.makedirs('best_model', exist_ok=True)

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs} starting")
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
        loss = criterion(outputs, labels) # Assuming criterion returns a combined loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log loss for training
        mlflow.log_metric("train/loss", loss.item(), step=epoch) # Log combined loss for now
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1e-5)})

    scheduler.step()
    epoch_loss = running_loss/len(train_loader)
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')
    mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
    epoch_losses.append(epoch_loss) # Store epoch loss for plotting


    # Avaliação do modelo após cada época
    accuracy = evaluate_model(model, val_loader, device)
    mlflow.log_metric("val_accuracy", accuracy)
    logging.info(f"Epoch {epoch+1}/{num_epochs} finished, Validation Accuracy: {accuracy}")
    val_accuracies.append(accuracy) # Store validation accuracy
    # Se este for o melhor modelo até agora, salve-o
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model_path = os.path.join('best_model', 'best_model.pth')
        torch.save(model.state_dict(), model_path)
        logging.info(f"Saved new best model at epoch {epoch+1} with accuracy {accuracy:.4f} to {model_path}")

    logging.info('Training finished')
    # O modelo já foi registrado via autolog

    # Print epoch_losses and val_accuracies for debugging
    print(f"Epoch Losses: {epoch_losses}")
    print(f"Validation Accuracies: {val_accuracies}")

    # Generate and log plots
    import matplotlib.pyplot as plt

    # Loss plot
    plt.figure()
    plt.plot(epoch_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("loss.png")
    mlflow.log_artifact("loss.png")
    logging.info("Loss plot saved as mlflow artifact")
    plt.close()
    # Accuracy plot
    plt.figure()
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy.png")
    mlflow.log_artifact("accuracy.png")
    logging.info("Accuracy plot saved as mlflow artifact")
    plt.close()
    # Retorna o melhor epoch e sua acurácia
    return best_epoch, best_accuracy

if __name__ == '__main__':
    # Argument parser to handle command line arguments
    parser = argparse.ArgumentParser(description="Train weed detection model")
    parser.add_argument('--model-name', type=str, default='resnet50',
                        choices=['alexnet', 'resnet50'],
                        help='Model architecture to use (alexnet or resnet50, default: resnet50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs to train (default: 10)')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of workers for data loading (default: 12)')

    args = parser.parse_args()

    # Hiperparâmetros
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_workers = args.num_workers
    model_name = args.model_name # Get model_name from arguments
    logging.info("Hyperparameters loaded from arguments")

    mlflow.set_tracking_uri("http://127.0.0.1:8081")
    logging.info("MLflow tracking URI set to http://127.0.0.1:8081")

    # Define experimento
    mlflow.set_experiment("weed_detection")
    logging.info("MLflow experiment set to 'weed_detection'")
    # Habilita autologging do PyTorch (model registry + system metrics opcional)
    try:
        mlflow.pytorch.autolog(
            log_models=True,
            registered_model_name="WeedClassifier",
            log_system_metrics=True
        )
        logging.info("MLflow PyTorch autologging enabled with system metrics")
    except TypeError:
        mlflow.pytorch.autolog(
            log_models=True,
            registered_model_name="WeedClassifier"
        )
        logging.info("MLflow PyTorch autologging enabled without system metrics (MLflow version may not support it)")

    with mlflow.start_run(run_name=f"{model_name}_run"):
        # Hiperparâmetros
        mlflow.log_params({
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "num_workers": num_workers
        })

        # Carregamento dos dados
        train_dataset = WeedDataset('dataset', train=True)
        val_dataset = WeedDataset('dataset', train=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logging.info("Data loaders initialized")

        # Modelo e otimização
        model = get_model(model_name, num_classes=4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=3000, gamma=0.1)
        logging.info("Model, criterion, optimizer, and scheduler initialized")

        # Dispositivo
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model.to(device)
        criterion.to(device)

        # Treinamento do modelo
        # Treinamento do modelo e retorno do melhor checkpoint
        best_epoch, best_accuracy = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, scheduler, num_epochs, device
        )
        logging.info(f"Best model obtained at epoch {best_epoch} with accuracy {best_accuracy:.4f}")
        # Salva metadados do melhor modelo
        metadata_path = os.path.join('best_model', 'model_info.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"model_name: {model_name}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"learning_rate: {learning_rate}\n")
            f.write(f"num_epochs: {num_epochs}\n")
            f.write(f"num_workers: {num_workers}\n")
            f.write(f"best_epoch: {best_epoch}\n")
            f.write(f"best_accuracy: {best_accuracy:.4f}\n")
        logging.info(f"Model metadata saved to {metadata_path}")
        # Log best model files as MLflow artifacts
        mlflow.log_artifacts("best_model", artifact_path="best_model")
        logging.info("Best model artifacts logged to MLflow under 'best_model' folder")

        # O registro do modelo será feito automaticamente pelo autolog
