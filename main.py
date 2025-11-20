import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import os
from PIL import Image

num_epochs = 1

class_map = {
    "Forest": 0,
    "Jungle": 1,
    "Plains": 2
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to GoogLeNet input
    transforms.ToTensor(),          # convert to tensor
])

class BiomesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):      #return length of dataset
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = Image.fromarray(image)
        label_str = self.annotations.iloc[index,1]
        y_label = torch.tensor(class_map[label_str])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = BiomesDataset(csv_file = 'labels.csv', root_dir = 'Data_Sources', transform = transform)
train_set, test_set = torch.utils.data.random_split(dataset, [1162,289])
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True)

model = torchvision.models.googlenet(weights=None, aux_logits=False)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, model):
    correct = 0
    samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _,predictions = scores.max(1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)
        print(f'Got {correct} / {samples} correct')
    model.train()


print("Training Set: ")
check_accuracy(train_loader, model)

print("Test Set: ")
check_accuracy(test_loader, model)