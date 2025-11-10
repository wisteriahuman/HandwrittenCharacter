from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    # train_loader, test_loader = get_data_loaders()
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    print("=" * 3 + "train_loader" + "=" * 3)
    # print(type(train_loader))
    image, label = mnist[0]
    print(f"画像のshape: {image.shape}")

    # print("=" * 3 + "test_loader" + "=" * 3)
    # print(type(test_loader))
    

if __name__ == "__main__":
    main()

