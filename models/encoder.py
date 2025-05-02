import torch
import torch.nn as nn
from torch.distributions import Laplace, Cauchy, Uniform, SigmoidTransform, AffineTransform, TransformedDistribution


class Encoder(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, num_node, strategy, bias):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(seq_len * input_dim, num_node)
        self.fc2 = nn.Linear(num_node, num_node)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.z_loc = nn.Linear(num_node, hidden_dim)
        self.strategy = strategy
        if strategy != 'Deterministic':
            self.z_logit_scale = nn.Linear(num_node, hidden_dim)
            self.bias = bias[strategy]

    def softplus(self, z_logit_scale):
        return torch.log(1 + torch.exp(z_logit_scale + self.bias))

    def reparameterize(self, z_loc, z_scale):
        if self.strategy == "Gaussian":
            epsilon = torch.randn_like(z_scale)  # sampling from a standard normal distribution
        elif self.strategy == 'Laplace':
            epsilon = Laplace(loc=torch.tensor(0.0).to(device=z_scale.device),
                              scale=torch.tensor(1.0).to(device=z_scale.device)).sample(
                z_scale.shape)  # sampling from Laplace distribution of loc 0 and scale 1
        elif self.strategy == 'Logistic':
            epsilon = TransformedDistribution(
                Uniform(torch.tensor(0.0).to(device=z_scale.device), torch.tensor(1.0).to(device=z_scale.device)),
                [SigmoidTransform().inv, AffineTransform(loc=torch.tensor(0.0).to(device=z_scale.device),
                                                         scale=torch.tensor(1.0).to(device=z_scale.device))]).sample(
                z_scale.shape)  # sampling from Logistic distribution of loc 0 and scale 1
        elif self.strategy == "Cauchy":
            # epsilon = Cauchy(loc=torch.tensor(0.0).to(device=z_log_scale.device), scale=torch.tensor(1.0).to(device=z_log_scale.device)).sample(z_log_scale.shape) # sampling from a Cauchy distribution of loc 0 and scale 1
            epsilon = torch.rand(z_scale.shape, device=z_scale.device)  # sampling from a Uniform distribution [0, 1)
            epsilon = epsilon.clamp(1e-4, 1 - 1e-4)  # clamp into (0, 1)
            epsilon = torch.tan(torch.pi * (epsilon - 0.5))
        elif self.strategy == 'Hypsecant':
            epsilon = torch.rand(z_scale.shape, device=z_scale.device)  # sampling from a Uniform distribution [0, 1)
            epsilon = epsilon.clamp(1e-6, 1 - 1e-6)  # clamp into (0, 1)
            epsilon = 2 / torch.pi * torch.log(
                torch.tan(torch.pi / 2 * (epsilon)))  # transforming to hyperbolic secant distribution
        else:
            raise (
                "Undefined distribution. Set strategy to Deterministic, Gaussian, Laplace, Logistic, Cauchy, or Hypsecant")
        return z_loc + z_scale * epsilon

    def forward(self, sequence):  # sequence has dimension (batch_size, seq_len, input_dim)
        sequence = sequence.to(self.fc1.weight.device)
        z = self.flatten(sequence)
        z = self.dropout(self.relu(self.fc1(z)))
        z = self.fc2(z)
        z_loc = self.z_loc(z)
        if self.strategy == 'Deterministic':
            return z_loc, None
        else:
            z_scale = self.softplus(self.z_logit_scale(z))
            return self.reparameterize(z_loc, z_scale), (z_loc, z_scale)