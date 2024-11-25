import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree, scatter


class GNNLayer(MessagePassing):
    def __init__(self, nnodefeatures, attention_size=64):
        super(GNNLayer, self).__init__()

        self.linear = nn.Linear(nnodefeatures, nnodefeatures)
        self.attention_transform = nn.Linear(2 * nnodefeatures, attention_size, bias=False)
        self.attention_vector = nn.Parameter(torch.ones((attention_size, 1)), requires_grad=True)

    def forward(self, nodefeatures, edge_index):
        row, col = edge_index
        deg = torch.sqrt(degree(col, nodefeatures.size(0), dtype=nodefeatures.dtype))
        deg[deg == float("inf")] = 0
        norm = deg[row] * deg[col]

        return self.propagate(edge_index, x=nodefeatures, norm=norm)

    def message(self, x_j, x_i, norm, edge_index):
        # Constructs messages from node j to node i

        attention = torch.exp(
            torch.matmul(self.attention_transform(torch.cat((x_j, x_i), dim=-1)), self.attention_vector)
        )
        attention_norm = scatter(attention, edge_index[1], reduce="sum")
        attention = attention / attention_norm[edge_index[0]]

        m = self.linear(x_j * attention) / norm.view(-1, 1)
        return m

    def aggregate(self, message, index):
        # Aggregates messages from neighbors
        return scatter(message, index, reduce="sum")

    def update(self, message):
        # Updates node embeddings
        return F.selu(message)


class EnzymesGNN(nn.Module):
    def __init__(self, nnodefeatures=3, nclasses=6, nlayers=3, attention_size=64, cls_hidden_dim=128):
        super(EnzymesGNN, self).__init__()
        self.nlayers = nlayers

        self.gcn = GNNLayer(nnodefeatures, attention_size=attention_size)

        self.classifier = nn.Sequential(
            nn.Linear(nnodefeatures, cls_hidden_dim),
            nn.SELU(),
            nn.Linear(cls_hidden_dim, nclasses),
            nn.Softmax(dim=-1),
        )

    def forward(self, graph):

        for _ in range(self.nlayers):
            x = self.gcn(graph.x, graph.edge_index)

        x = global_mean_pool(x, graph.batch)

        out = self.classifier(x)

        return out


if __name__ == "__main__":
    from dataset import load_data, get_dataloaders

    train_data, test_data = load_data()
    print(train_data[0])

    train_loader, test_loader = get_dataloaders(train_data, test_data, batch_size=64)

    model = EnzymesGNN()
    for graph, label in train_loader:
        print(graph)
        print(model(graph).shape)
        print(label.shape)
        break
