from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_loaders(root='data', batch_size=64, small=True):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    if small:
        train = Subset(train, range(min(2000, len(train))))
        test = Subset(test, range(min(500, len(test))))
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
