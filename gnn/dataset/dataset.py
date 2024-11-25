from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Dataset


class EnzymeDataset(Dataset):
    def __init__(self, root_path):
        self.dataset = TUDataset(name="ENZYMES", root=root_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        return (self.dataset[id], self.dataset[id].y.item())


def load_data(root_path="gnn/dataset"):
    dataset = EnzymeDataset(root_path)

    train_data, test_data = random_split(dataset, [0.85, 0.15])
    train_data.classes = list(range(6))
    test_data.classes = list(range(6))

    return train_data, test_data


def get_dataloaders(train_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_data, test_data = load_data()

    print(train_data[0])
    print(len(train_data))

    train_loader, test_loader = get_dataloaders(train_data, test_data, 64)
    for graph, label in train_loader:
        print(graph)
