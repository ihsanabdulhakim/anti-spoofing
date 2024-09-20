import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

def train_antispoofing_liveness():
    # Define device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
    ])

    # Load datasets with explicit class_to_idx mapping
    train_dataset = datasets.ImageFolder(root='./datasets/train', transform=transform)
    valid_dataset = datasets.ImageFolder(root='./datasets/valid', transform=transform)

    # Explicitly set class indices
    train_dataset.class_to_idx = {'spoofphone': 0, 'realhuman': 1, 'spoofpaper': 2}
    valid_dataset.class_to_idx = {'spoofphone': 0, 'realhuman': 1, 'spoofpaper': 2}

    # Ensure datasets have the correct classes
    print(f"Train Dataset Classes: {train_dataset.class_to_idx}")
    print(f"Valid Dataset Classes: {valid_dataset.class_to_idx}")
    print("Succeed to read all of train_dataset and valid_dataset")

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("Succeed to load all of train_dataset and valid_dataset")

    # Initialize the ResNet-50 model
    num_classes = len(train_dataset.classes)  # Ensure it detects 3 classes
    model = models.resnet50(pretrained=True)
    
    # Modify the fully connected layer to match the number of classes
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    
    model = model.to(device)
    print("Succeed to create ResNet-50 model")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 200
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc = 100 * correct / total
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # Print Epoch Results
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'anti-spoofing-resnet50.pt')

    # Save the training and validation results to CSV
    results_df = pd.DataFrame({
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'train_acc': train_accs,
        'valid_acc': valid_accs
    })
    results_df.to_csv('antispoofing-training-resnet50.csv', index=False)

    # Plot the training and validation loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(valid_accs, label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig('antispoofing-training-resnet50-plot.png')
    plt.show()

# Run the training function
train_antispoofing_liveness()
