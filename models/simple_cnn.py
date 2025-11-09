import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=32 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        # Flatten
        x = x.view(x.size()[0], -1)
        # Linear Block 1
        x = self.fc1(x)
        x = F.relu(x)
        # Output Layer
        x = self.fc2(x)
        return x


def main():
    x = torch.randn(1, 1, 28, 28)
    model = Simple_CNN()
    output = model(x)
    print(f"出力shape: {output.shape}")

if __name__ == "__main__":
    main()

