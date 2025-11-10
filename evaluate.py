from os import wait
import torch
import torch.nn as nn
from data import get_data_loaders
from models.simple_cnn import Simple_CNN

MODEL_PATH = "models/simple_cnn.pth"

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple_CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    _, test_loader = get_data_loaders()
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for idx, (input, label) in enumerate(test_loader):
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output, label)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def main():
    predict()

if __name__ == "__main__":
    main()

