"""PELS-VAE models, ported line by line from their src/tcn.py and src/vae_models.py.

Encoder: a Temporal Convolutional Network (TCN: causal, dilated convolutions with residual blocks) reads
(phase, magnitude, error) of the light curve; its last time step, concatenated with the one-hot class
(and, for cVAE-P, the physical parameters), goes through Dense(16)+ReLU+Dropout to the mean and
log-variance of a small latent space (4 dimensions). Decoder: the latent vector and class are repeated at
every time step next to the phase, a second TCN runs over them, and Dense+sigmoid gives the magnitude.
A GRU version (VAE_RNN) replaces both TCNs by recurrent layers (their Fig. 4 comparison).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Drops the extra right-hand padding, which makes the convolution causal."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dilation)), Chomp1d(pad), nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_out, n_out, k, padding=pad, dilation=dilation)), Chomp1d(pad), nn.ReLU(),
            nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        # their init_weights() sets conv weights ~ N(0, 0.01), but on weight-normed convs that assignment is
        # overwritten by the next forward pass, so only the 1x1 residual conv really gets it
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(self.net(x) + res)


class TemporalConvNet(nn.Module):
    def __init__(self, n_in, channels, k, dropout):
        super().__init__()
        self.network = nn.Sequential(*[TemporalBlock(n_in if i == 0 else channels[i - 1], channels[i], k, 2 ** i,
                                                     dropout) for i in range(len(channels))])

    def forward(self, x):
        return self.network(x)


class VAE_TCN(nn.Module):
    def __init__(self, latent_dim=4, seq_len=600, kernel_size=5, hidden_dim=48, nlevels=9, n_feats=3,
                 dropout=0.2, lab_dim=8, phy_dim=0):
        super().__init__()
        self.latent_dim, self.lab_dim, self.phy_dim, self.seq_len = latent_dim, lab_dim, phy_dim, seq_len
        enc = [hidden_dim] * nlevels
        enc[0], enc[1], enc[2] = 8, 16, 32
        dec = [hidden_dim] * nlevels
        dec[-1], dec[-2], dec[-3] = 8, 16, 32
        self.enc_tcn = TemporalConvNet(n_feats, enc, kernel_size, dropout)
        self.enc_linear = nn.Sequential(nn.Linear(hidden_dim + lab_dim + phy_dim, 16), nn.ReLU(), nn.Dropout(dropout))
        self.enc_linear_mu = nn.Linear(16, latent_dim)
        self.enc_linear_std = nn.Linear(16, latent_dim)           # (it outputs the log-variance)
        self.dec_tcn = TemporalConvNet(1 + latent_dim + lab_dim, dec, kernel_size, dropout)   # feed_pp = False
        self.dec_linear = nn.Linear(dec[-1], 1)

    def encoder(self, x, label, phy=None):
        h = self.enc_tcn(x.transpose(1, 2)).transpose(1, 2)[:, -1, :]
        h = torch.cat([h, label] + ([phy] if self.phy_dim > 0 else []), 1)
        h = self.enc_linear(h)
        return self.enc_linear_mu(h), self.enc_linear_std(h)

    def decoder(self, z, dt, label):
        z = torch.cat([z, label], 1).unsqueeze(1).repeat(1, dt.shape[1], 1)
        h = torch.cat([dt.unsqueeze(-1), z], -1)
        h = self.dec_tcn(h.transpose(1, 2)).transpose(1, 2)
        return torch.sigmoid(self.dec_linear(h))

    def forward(self, x, label, phy=None):
        mu, logvar = self.encoder(x, label, phy)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)       # reparameterisation trick
        return self.decoder(z, x[:, :, 0], label), mu, logvar, z


class VAE_RNN(nn.Module):
    def __init__(self, latent_dim=4, seq_len=600, hidden_dim=48, n_layers=2, n_feats=3, dropout=0.2, lab_dim=8,
                 phy_dim=0):
        super().__init__()
        self.latent_dim, self.lab_dim, self.phy_dim, self.seq_len = latent_dim, lab_dim, phy_dim, seq_len
        self.enc_lstm = nn.GRU(n_feats, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dec_lstm = nn.GRU(1 + latent_dim + lab_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.enc_linear = nn.Sequential(nn.Linear(hidden_dim + lab_dim + phy_dim, 16), nn.ReLU(), nn.Dropout(dropout))
        self.enc_linear_mu = nn.Linear(16, latent_dim)
        self.enc_linear_std = nn.Linear(16, latent_dim)
        self.dec_linear = nn.Linear(hidden_dim, 1)
        for rnn in (self.enc_lstm, self.dec_lstm):            # their init_weights()
            for name, p in rnn.named_parameters():
                if "bias" in name:
                    nn.init.normal_(p, 0.0)                  # (std 1, as in their call)
                elif "weight" in name:
                    nn.init.xavier_normal_(p)

    def encoder(self, x, label, phy=None):
        h, _ = self.enc_lstm(x)
        h = torch.cat([h[:, -1, :], label] + ([phy] if self.phy_dim > 0 else []), 1)
        h = self.enc_linear(h)
        return self.enc_linear_mu(h), self.enc_linear_std(h)

    def decoder(self, z, dt, label):
        z = torch.cat([z, label], 1).unsqueeze(1).repeat(1, dt.shape[1], 1)
        h, _ = self.dec_lstm(torch.cat([dt.unsqueeze(-1), z], -1))
        return torch.sigmoid(self.dec_linear(h))

    def forward(self, x, label, phy=None):
        mu, logvar = self.encoder(x, label, phy)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z, x[:, :, 0], label), mu, logvar, z


def vae_loss(x, xhat, mu, logvar, beta):
    """Their Trainer._loss: plain MSE on the scaled magnitude + beta * KL / batch / 1e4.
    (The paper's Eq. 4 has an error-weighted MSE and a third KL term on the errors; the released code
    comments both out, with the note that plain MSE converges faster and better.)"""
    mse = F.mse_loss(xhat[:, :, 0], x[:, :, 1])
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] / 1e4
    return mse + beta * kld, mse, kld
