import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import get_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os

def evaluate_model(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("Model checkpoint 'models/best_model.pth' not found.")
        print("Please train the model using `python train.py` first.")
        return
        
    print(f"Loading best weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    best_acc = checkpoint['best_acc']
    
    # We use validation loader for evaluation 
    _, val_loader, _ = get_dataloaders(data_dir, batch_size=32)
    
    model = get_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nRunning inference on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(f"\nModel had Validation Accuracy during training: {best_acc:.4f}\n")
    
    print("Classification Report:")
    print("----------------------")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    print("Generating Confusion Matrix Plot...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix visualization to 'confusion_matrix.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    args = parser.parse_args()
    evaluate_model(args.data_dir)
