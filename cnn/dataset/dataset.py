import os
import zipfile

import requests
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def download_data(url: str, output_dir: str):
    """
    Downloads and extracts a dataset from a given URL.

    Parameters:
        url (str): The URL to download the dataset from.
        output_dir (str): The directory where the dataset will be extracted.
    """
    # Define paths
    zip_file_path = os.path.join(output_dir, "archive.zip")
    if os.path.exists(zip_file_path):
        return

    print("Downloading brain tumor dataset ...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the dataset
    response = requests.get(url)
    with open(zip_file_path, "wb") as zip_file:
        zip_file.write(response.content)

    extract_data(zip_file_path)
    print("Dataset ready")


def extract_data(file: str):
    print("Unzipping...")
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(file))


# We convert images to standard size of 128x128 pixels.
# Grayscale transform is just needed to remove the color channels,
# because the provided images are all grayscale anyway and all 3 channel have the same numbers
def load_data(path="cnn/dataset", transform=None) -> tuple[Dataset]:
    download_data("https://www.kaggle.com/api/v1/datasets/download/bilalakgz/brain-tumor-mri-dataset", path)
    data_root = os.path.join(path, "brain_tumor_dataset", "brain_tumor_classification")

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

    train_data = datasets.ImageFolder(root=os.path.join(data_root, "Training"), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(data_root, "Testing"), transform=transform)
    return train_data, test_data
