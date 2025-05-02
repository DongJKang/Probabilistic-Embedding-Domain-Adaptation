import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_node):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, num_node)
        self.fc2 = nn.Linear(num_node, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.sigm = nn.Sigmoid()

    def forward(self, z): # z has dimension (batch_size, hidden_dim)
        z = self.dropout(self.relu(self.fc1(z)))
        output = self.sigm(self.fc2(z))

        return output
