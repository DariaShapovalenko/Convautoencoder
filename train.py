import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from convautoencodermodel import ConvAutoencoder64, ConvAutoencoder128, ConvAutoencoder256

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

transform64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset64 = ImageDataset('dataset', transform=transform64)
dataset128 = ImageDataset('dataset', transform=transform128)
dataset256 = ImageDataset('dataset', transform=transform256)

dataloader64 = DataLoader(dataset64, batch_size=32, shuffle=True)
dataloader128 = DataLoader(dataset128, batch_size=24, shuffle=True)
dataloader256 = DataLoader(dataset256, batch_size=16, shuffle=True)

model64 = ConvAutoencoder64().to(device)
model128 = ConvAutoencoder128().to(device)
model256 = ConvAutoencoder256().to(device)

optimizer64 = optim.Adam(model64.parameters(), lr=0.001)
optimizer128 = optim.Adam(model128.parameters(), lr=0.001)
optimizer256 = optim.Adam(model256.parameters(), lr=0.001)

print("Тренування моделі 64x64...")
for epoch in range(30):
    running_loss = 0.0
    for data in dataloader64:
        img = data.to(device)
        output = model64(img)
        loss = criterion(output, img)
        optimizer64.zero_grad()
        loss.backward()
        optimizer64.step()
        running_loss += loss.item()
    print(f'64x64 Epoch [{epoch+1}/30], Loss: {running_loss/len(dataloader64):.4f}')

print("Тренування моделі 128x128...")
for epoch in range(30):
    running_loss = 0.0
    for data in dataloader128:
        img = data.to(device)
        output = model128(img)
        loss = criterion(output, img)
        optimizer128.zero_grad()
        loss.backward()
        optimizer128.step()
        running_loss += loss.item()
    print(f'128x128 Epoch [{epoch+1}/30], Loss: {running_loss/len(dataloader128):.4f}')

print("Тренування моделі 256x256...")
for epoch in range(30):
    running_loss = 0.0
    for data in dataloader256:
        img = data.to(device)
        output = model256(img)
        loss = criterion(output, img)
        optimizer256.zero_grad()
        loss.backward()
        optimizer256.step()
        running_loss += loss.item()
    print(f'256x256 Epoch [{epoch+1}/30], Loss: {running_loss/len(dataloader256):.4f}')

torch.save({
    'model64': model64.state_dict(),
    'model128': model128.state_dict(),
    'model256': model256.state_dict()
}, 'convautoencoders.pth')