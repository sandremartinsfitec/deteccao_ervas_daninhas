import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from data_loading import WeedDataset
from tqdm import tqdm  # Import tqdm, although it might not be used in this basic visualization

def visualize_dataset(dataset, num_images=5):
    """
    Visualiza algumas imagens do dataset.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    plt.figure(figsize=(15, 3 * num_images))
    for i, (image, label) in enumerate(dataloader):
        if i >= num_images:
            break
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        label_name = dataset.classes[label] # Assuming dataset.classes is defined
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(image)
        plt.title(f'Label: {label_name}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load validation dataset
    val_dataset = WeedDataset('dataset', train=False) # Ensure 'dataset' path is correct
    visualize_dataset(val_dataset)
