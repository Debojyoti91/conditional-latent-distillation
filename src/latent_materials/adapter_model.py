"""Conditional adapter shared by the conditioning experiments."""

from torch import nn

LATENT_DIM = 32
HIDDEN_DIM = 128


class ConditionalAdapter(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
