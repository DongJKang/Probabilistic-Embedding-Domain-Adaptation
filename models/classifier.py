import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_node, out_node):
        super().__init__()
        self.out_node = out_node
        self.fc1 = nn.Linear(hidden_dim, num_node)
        self.fc2 = nn.Linear(num_node, out_node)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        if out_node == 1:
            self.sigm = nn.Sigmoid()
        else:
            self.lsoftm = nn.LogSoftmax(dim=1)

    def forward(self, z): # z has dimension (batch_size, hidden_dim)
        z = self.dropout(self.relu(self.fc1(z)))

        if self.out_node == 1:
            output = self.sigm(self.fc2(z))
        else:
            output = self.lsoftm(self.fc2(z))
        return output