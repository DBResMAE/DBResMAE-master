import torch
import torch.nn as nn
import numpy as np
class CVAE(nn.Module):
    def __init__(self, input_shape, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        self.input_dim = np.prod(input_shape)
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # 编码器网络
        self.fc1 = nn.Linear(self.input_dim + condition_dim, 256)
        self.fc2_mean = nn.Linear(256, latent_dim)
        self.fc2_log_var = nn.Linear(256, latent_dim)

        # 解码器网络
        self.fc3 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc4 = nn.Linear(256, self.input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc2_mean(h), self.fc2_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = torch.relu(self.fc3(torch.cat([z, c], dim=1)))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c), mu, log_var
