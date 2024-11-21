import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification


class SentimentTransfer(nn.Module):
    def __init__(self, in_features=6, nclasses=6):
        super(SentimentTransfer, self).__init__()

        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        # freeze weights
        for param in self.sentiment_model.parameters():
            param.requires_grad = False

        # change the output layer
        self.sentiment_model.classifier.out_proj = nn.Linear(768, nclasses)
        for param in self.sentiment_model.classifier.out_proj.parameters():
            param.requires_grad = True

        self.tokenizer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1),
            nn.Flatten(start_dim=-2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        tokens = self.tokenizer(x).type(torch.int64)
        attention = torch.ones(tokens.shape)

        encodings = {"input_ids": tokens, "attention_mask": attention}

        out = self.sentiment_model(**encodings)

        return out.logits


if __name__ == "__main__":
    from dataset import load_data
    from torch.utils.data import DataLoader

    data, _ = load_data()

    train_loader = DataLoader(data, batch_size=64)

    model = SentimentTransfer()

    for seq, label in train_loader:
        print(model(seq).shape)
        print(label.shape)
        break
