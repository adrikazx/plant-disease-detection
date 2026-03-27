import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def get_dataloaders(data_dir, batch_size=32, img_size=(224, 224), val_split=0.2):
    """
    Creates training and validation dataloaders from an ImageFolder structure.
    """
    # ResNet and MobileNet expect 224x224 and standard ImageNet normalization
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Try to find the actual directory that contains the class folders
    possible_roots = [
        data_dir,
        os.path.join(data_dir, "PlantVillage"),
        os.path.join(data_dir, "plantvillage_deeplearning_paper_dataset"),
        os.path.join(data_dir, "PlantVillage dataset"),
        os.path.join(data_dir, "plantvillage dataset"),
    ]
    
    root_dir = data_dir
    for p in possible_roots:
        if os.path.exists(p) and os.path.isdir(p) and len(os.listdir(p)) > 5:
            # We assume the directory with > 5 folders is the root containing classes
            root_dir = p
            break
            
    print(f"Loading data from: {root_dir}")
    # Load dataset. No default transforms so it loads as PIL image.
    full_dataset = datasets.ImageFolder(root=root_dir)
    classes = full_dataset.classes
    
    # --- FAST TRAINING PROTOTYPE FOR DEADLINE ---
    # Since you must submit by 4 AM, we downsample the massive dataset to 5%
    # This guarantees a full pass takes 1-2 minutes instead of 30!
    subset_size = int(len(full_dataset) * 0.05)
    print(f"Applying emergency assignment speedup... Downsizing training to {subset_size} images!")
    
    full_dataset = torch.utils.data.Subset(
        full_dataset, 
        torch.randperm(len(full_dataset))[:subset_size].tolist()
    )
    # ---------------------------------------------
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"Total dataset size: {total_size}. Split: Train({train_size}), Val({val_size})")

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Wrap subsets with transforms
    train_dataset = TransformedDataset(train_dataset, transform=train_transform)
    val_dataset = TransformedDataset(val_dataset, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, classes
