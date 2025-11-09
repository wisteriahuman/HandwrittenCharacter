import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import get_data_loaders
from models.simple_cnn import Simple_CNN

MAX_EPOCH = 10
BATCH_SIZE = 64

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = Simple_CNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    batch_total = len(train_loader)
    for epoch in range(MAX_EPOCH):
        loss_total = 0
        for batch, (input, label) in enumerate(train_loader):
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output, label)
            loss_total += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{MAX_EPOCH}], Loss: {loss_total/batch_total:.3f}")

    torch.save(model.state_dict(), './models/simple_cnn.pth')


def main():
    train()

if __name__ == "__main__":
    main()

