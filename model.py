"""model.py"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class BetaVAE_3D(nn.Module):
    def __init__(self, z_dim=10):
        super(BetaVAE_3D, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.Conv2d(256, 2*self.z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(self.z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu.squeeze(), logvar.squeeze()


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_2D(BetaVAE_3D):
    def __init__(self, z_dim=10):
        super(BetaVAE_2D, self).__init__()
        self.z_dim = z_dim

        # Views are applied just for the consistency in shape with CONV-based models
        self.encode = nn.Sequential(
            View((-1, 4096)),
            nn.Linear(4096, 1200),
            nn.ReLU(True),
            nn.Linear(1200, 1200),
            nn.ReLU(True),
            nn.Linear(1200, 2*self.z_dim),
            View((-1, 2*self.z_dim, 1, 1)),
        )
        self.decode = nn.Sequential(
            View((-1, self.z_dim)),
            nn.Linear(self.z_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096),
            View((-1, 1, 64, 64)),
        )
        self.weight_init()

    def forward(self, x):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z).view(x.size())

        return x_recon, mu, logvar


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    net = BetaVAE(32)
    x = Variable(torch.rand(1, 3, 64, 64))
    net(x)
