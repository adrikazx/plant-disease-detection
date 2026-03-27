import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    """
    Returns a fine-tunable ResNet18 model.
    Using transfer learning generally results in faster convergence and higher accuracy.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Replace the final fully connected classification layer to match the number of classes in our dataset.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
