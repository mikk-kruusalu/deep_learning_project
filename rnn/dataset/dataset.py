from pathlib import Path
import zipfile

import requests
import numpy as np
from torch.utils.data import Dataset
import torch


def download_data(url: str, output_dir: str):
    """
    Downloads and extracts a dataset from a given URL.

    Parameters:
        url (str): The URL to download the dataset from.
        output_dir (str): The directory where the dataset will be extracted.
    """
    # Define paths
    zip_file_path = Path(output_dir) / "archive.zip"
    if zip_file_path.exists():
        return

    print("Downloading activity recognition dataset ...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download the dataset
    response = requests.get(url)
    with open(zip_file_path, "wb") as zip_file:
        zip_file.write(response.content)

    extract_data(zip_file_path)
    extract_data(Path(output_dir) / "UCI HAR Dataset.zip")
    print("Dataset ready")


def extract_data(file: Path):
    print("Unzipping...")
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(file.parent)


class ActivityDataset(Dataset):
    def __init__(self, root_path: Path):
        self.root_path = root_path

        arrays = []
        self.labels = []
        for f in root_path.glob("Inertial Signals/body*.txt"):
            array = np.genfromtxt(f)
            arrays.append(torch.tensor(array, dtype=torch.float))

            label = "_".join(f.stem.split("_")[1:3])
            self.labels.append(label)

        self.seq = torch.stack(arrays, dim=-1)

        y = np.genfromtxt(list(root_path.glob("y_*.txt"))[0])
        self.y = torch.tensor(y-1, dtype=torch.long)

        self.classes = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        ]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.seq[i], self.y[i]

    def __getitems__(self, idx):
        return [(seq, y) for seq, y in zip(self.seq[idx], self.y[idx])]


# We convert images to standard size of 128x128 pixels.
# Grayscale transform is just needed to remove the color channels,
# because the provided images are all grayscale anyway and all 3 channel have the same numbers
def load_data(path="rnn/dataset") -> tuple[Dataset]:
    download_data(
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip", path
    )

    test_root = Path(path) / "UCI HAR Dataset" / "test"
    train_root = Path(path) / "UCI HAR Dataset" / "train"

    test_data = ActivityDataset(test_root)
    train_data = ActivityDataset(train_root)

    return train_data, test_data


if __name__ == "__main__":
    train_data, _ = load_data()
    print(len(train_data))
    print(train_data[0][0].shape)
    print(train_data[0][1])
