from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from dotenv import load_dotenv
import os
import time

if torch.cuda.is_available():
    print("Using GPU", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU found")

device = torch.device("cuda:0")

load_dotenv()

data_filepath = os.getenv("DATASET_PATH")

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)
    ], p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.RandomHorizontalFlip(p=0.3), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = ImageFolder(root=f"{data_filepath}\\train", transform=train_transforms)
test_dataset = ImageFolder(root=f"{data_filepath}\\test", transform=test_transforms)

BATCH_SIZE=128

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
    
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential( # simplifies image a bunch
            # 224 -> 112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 112 -> 56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 56->28
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 28 -> 14
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential( # 2d->1d and classifies emotions
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
def evaluate(model, loader, device=torch.device('cpu')):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
if __name__ == '__main__':
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    best_acc = 0.0
    for epoch in range(25):
        start = time.time()
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199: 
                print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/200:.3f}')
                running_loss = 0.0
        
        epoch_time = time.time() - start
        print(f'Epoch {epoch+1} took {epoch_time:.1f}s')
        val_acc = evaluate(net, test_loader, device)
        print(f"Epoch {epoch + 1} | Validation Accuracy: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), 'best_model.pth')
            print(f"Saved new best model with accuracy {val_acc:.2%}")
    
    print(f"Done training! Best model with accuracy {best_acc:.2%}")