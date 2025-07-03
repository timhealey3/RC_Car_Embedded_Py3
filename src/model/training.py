# This must be the very first import and code that runs
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
# Now import other required modules
import os
import torch
from torch.utils.data import DataLoader
from model import NeuralNetwork
# Set device for CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


def training_process():
    # Set up paths
    camera_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../camera'))
    csv_path = os.path.join(camera_dir, 'training_dataset.csv')
    print(f"Using training data from: {csv_path}")
    
    print(f"CSV path: {csv_path}")
    print(f"CSV exists: {os.path.exists(csv_path)}")
    
    # Use 'training' as the folder directory since that's where the images are
    data = CustomImageDataset('training_dataset.csv', camera_dir, 'training')
    train_loader, test_loader = data_split(data)

    # Create model and move to device
    model = NeuralNetwork().to(device)
    
    # Print model summary
    print("Model architecture:")
    print(model)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
    
    # Train and test the model
    train_model(model, loss_fn, optimizer, train_loader, test_loader)

def train_model(model, loss_fn, optimizer, train_loader, test_loader):
    epoch_number = 0
    num_epochs = 350
    validation_mode = True
    best_val_acc = 0
    for _ in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        avg_loss = one_epoch(epoch_number, loss_fn, optimizer, model, train_loader)
        epoch_number += 1

        if validation_mode:
            accuracy = test_model(model, loss_fn, test_loader)
            if accuracy > best_val_acc:
                print(f'\nNew best validation accuracy: {accuracy:.2f}%\n  Saving model...\n')
                best_val_acc = accuracy
                latest_model_path = 'models/latest_model.pth'
                os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
                torch.save(model.state_dict(), latest_model_path)

def test_model(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # Move data to the same device as model
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Ensure correct shape (B, C, H, W)
            if inputs.dim() == 4 and inputs.shape[-1] in [1, 3]:  # NHWC to NCHW if needed
                inputs = inputs.permute(0, 3, 1, 2).contiguous()
            
            # Convert to float32 if needed
            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item() * inputs.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy
    

def one_epoch(epoch_index, loss_fn, optimizer, model, training_data):
    running_loss = 0.0
    last_loss = 0.0
    total_samples = 0
    
    model.train(True)
    
    for batch_idx, (images, labels) in enumerate(training_data):
        # Move data to the same device as model
        images = torch.stack([transform_data(image) for image in images])
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Ensure correct shape (B, C, H, W)
        if images.dim() == 4 and images.shape[-1] in [1, 3]: 
            images = images.permute(0, 3, 1, 2)
        
        # Convert to float32 if needed
        if images.dtype != torch.float32:
            images = images.float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Print statistics
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(training_data):
            avg_loss = running_loss / total_samples
            print(f'Epoch: {epoch_index + 1} [{batch_idx + 1}/{len(training_data)}]',
                  f'Loss: {avg_loss:.6f}')
            
            # Reset running loss for next batch
            running_loss = 0.0
            total_samples = 0
    
    return last_loss

def transform_data(image):
    train_transform = transforms.RandomApply([
        transforms.RandomAffine(
            degrees=10, 
            translate=(0.2, 0.2), 
            scale=(0.8, 1.2), 
            shear=10
        )
    ], p=0.35)
    return train_transform(image)

def data_split(data):
    # split the data into training and testing sets
    all_labels = [data[i][1] for i in range(len(data))]
    indices = list(range(len(data)))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=all_labels, 
        random_state=42
    )
    train_data = Subset(data, train_idx)
    test_data = Subset(data, test_idx)
    # create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)    
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    return train_loader, test_loader

training_process()
