import torch
import torch.nn as nn

class CORALLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_feature, tgt_feature):
        d = src_feature.data.shape[1]  # dim vector

        # source covariance
        xm = torch.mean(src_feature, 0, keepdim=True) - src_feature
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(tgt_feature, 0, keepdim=True) - tgt_feature
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)

        return loss

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels, device='cuda') - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class KLDLoss(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy

    def forward(self, z_loc, z_scale):
        if self.strategy == 'Gaussian':
            kl_div = -0.5 * torch.sum(1 + 2.0 * torch.log(z_scale) - z_loc.pow(2) - z_scale.pow(2), dim=1)
        elif self.strategy == 'Laplace':
            kl_div = torch.sum(z_scale * torch.exp(-z_loc.abs() / z_scale) + z_loc.abs() - torch.log(z_scale) - 1, dim=1)
        elif self.strategy == 'Logistic':
            raise NotImplementedError
        elif self.strategy == 'Cauchy':
            kl_div = torch.log(((z_scale + 1).pow(2) + z_loc.pow(2)) / (4 * z_scale))
        elif self.strategy == 'Hypsecant':
            raise NotImplementedError
        else:
            raise ValueError("Undefined distribution. Set strategy to Deterministic, Gaussian, Laplace, Logistic, Cauchy, or Hypsecant.")

        return torch.sum(kl_div) / kl_div.size(0)