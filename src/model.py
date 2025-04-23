import torch.nn as nn
import torchvision.models as models
import logging

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        logging.info("Initializing ResNet50 model")
        self.resnet50 = models.resnet50(pretrained=True)
        # Modify the final fully connected layer for our number of classes
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        logging.info("ResNet50 model initialized with modified final layer")

    def forward(self, x):
        return self.resnet50(x)

def get_model(model_name, num_classes):
    if model_name == 'alexnet':
        return AlexNet(num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes)
    else:
        raise ValueError(f"Model name '{model_name}' not recognized")

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        logging.info("Initializing AlexNet model")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16384, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        logging.info("AlexNet model initialized")

    def forward(self, x):
        logging.info(f"Input shape to forward: {x.shape}")
        x = self.features(x)
        logging.info(f"Shape after features: {x.shape}")
        x = x.view(x.size(0), -1)
        logging.info(f"Shape after flatten: {x.shape}")
        x = self.classifier(x)
        logging.info(f"Output shape of classifier: {x.shape}")
        return x
