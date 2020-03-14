import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms

from utils.models import CLSTM, resnet18

from utils.data import ImportDataset

import argparse
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Parameters.")
parser.add_argument('-r', '--root', type=str, default=os.path.join('./data'), nargs="?", const=True, help="Project data root.")
parser.add_argument('-m', '--model', type=str, default='CLSTM', nargs="?", const=True, help="CLSTM or alexnet")
parser.add_argument('-e', '--epochs', type=int, default=5, nargs="?", const=True, help="Epoches.")
parser.add_argument('-bs', '--batch_size', type=int, default=32, nargs="?", const=True, help="Batch Size")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, nargs="?", const=True, help="Learning rate")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = ImportDataset(os.path.join(args.root, 'data.csv'), transform)
#                                                                   True if CNN
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

model = CLSTM(True, 1000).to(device) if args.model.lower() == 'clstm' else resnet18.to(device)

criterion = nn.CrossEntropyLoss()#nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if __name__ == '__main__':
    for epoch in range(args.epochs):

        total = 0
        correct = 0
        accuracy= 0

        loadbar = tqdm(dataloader)
        for imgs, labels in loadbar:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()

            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            accuracy = 100 * correct / total

            loadbar.set_description(f"Epoch: {epoch}/{args.epochs}")

        print(f"Epoch accuracy: {accuracy}")

    torch.save(model.state_dict(), os.path.join(args.root, f'models/{args.model.lower()}.pth'))