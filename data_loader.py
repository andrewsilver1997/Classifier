from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as T


def get_train_loader(batch_size=16, num_workers=1):
    train_data = datasets.ImageFolder(
        'data/train_edge',
        T.Compose([
            T.CenterCrop((227, 227)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    train_loader = data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader


def get_test_loader(batch_size=16, num_workers=1):
    test_data = datasets.ImageFolder(
        'data/test_edge',
        T.Compose([
            T.CenterCrop((227, 227)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return test_loader

