import argparse
import os

import torch
import yaml
from dataset import load_data
from simplecnn import SimpleCNN
from torch.utils.data import DataLoader
from training import train
from unet_transfer import UnetTransfer

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True)
parser.add_argument("-o", "--output-dir", default="cnn/checkpoints")
parser.add_argument("-d", "--device", default="cpu")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--simplecnn", action="store_true", help="Use Simple CNN model")
group.add_argument("--unet", action="store_true", help="Use UNet transfer learning model")

args = parser.parse_args()

# load config file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
hyperparams = config["hyperparams"]

# prepare data
train_data, test_data = load_data()
train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"])
test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"])

if args.simplecnn:
    model = SimpleCNN()
elif args.unet:
    model = UnetTransfer()

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()

checkpoint_dir = os.path.join(args.output_dir, config["exp_name"])
os.makedirs(checkpoint_dir, exist_ok=True)
train(model, criterion, optimizer, train_loader, test_loader, checkpoint_dir, nepochs=hyperparams["nepochs"])
