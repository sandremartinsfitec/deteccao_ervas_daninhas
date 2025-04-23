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
    mlflow.end_run() # End any active run before starting a new one
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("num_workers", num_workers)

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

    logging.info('Training finished')
    mlflow.pytorch.log_model(model, "alexnet_model")
    logging.info('Trained model saved as mlflow artifact')

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

    mlflow.pytorch.autolog()
    logging.info("MLflow autologging enabled")

    mlflow.log_param("model_name", model_name) # Log model_name
    mlflow.set_tag("batch_size", batch_size)
    mlflow.set_tag("learning_rate", learning_rate)
    mlflow.set_tag("num_epochs", num_epochs)
    mlflow.set_tag("model_architecture", model_name) # More descriptive tag
    logging.info("MLflow tags set for hyperparameters and model architecture")

    # Carregamento dos dados
    train_dataset = WeedDataset('dataset', train=True)
    val_dataset = WeedDataset('dataset', train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logging.info("Data loaders initialized")
    mlflow.log_param("num_workers", num_workers) # Log num_workers to MLflow

    # Modelo
    model = get_model(model_name, num_classes=4) # Use get_model function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=3000, gamma=0.1)
    logging.info("Model, criterion, optimizer, and scheduler initialized")

    # Define device - use GPU 0 if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use cuda if available
    logging.info(f"Using device: {device}")

    # Move model and criterion to device
    model.to(device)
    criterion.to(device)
    logging.info("Model and criterion moved to device")

    # Treinamento do modelo
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    logging.info("Training function called")
