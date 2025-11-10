import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 5 * 5, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple_CNN()
    model = model.to(device)
    model.eval()
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    print(f"出力shape: {output.shape}")

if __name__ == "__main__":
    main()

