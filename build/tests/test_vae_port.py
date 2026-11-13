"""Our PELS-VAE port must match the original code: same parameter count as their best run (300,897) and the
same outputs once the weights are copied.
    python build/tests/test_vae_port.py <folder containing the original src/tcn.py and src/vae_models.py>
"""
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, sys.argv[1])
from src.vae_models import VAE_RNN as OrigRNN  # noqa: E402
from src.vae_models import VAE_TCN as OrigTCN  # noqa: E402

from astra.tcn_vae import VAE_RNN, VAE_TCN, vae_loss  # noqa: E402

torch.manual_seed(0)
count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)


def copy_by_order(src, dst):
    """Copy parameters in registration order (names differ: weight_g/v vs parametrizations.original0/1)."""
    sp = [p for _, p in src.named_parameters()]
    dp = [p for _, p in dst.named_parameters()]
    sn = [n for n, _ in src.named_parameters()]
    dn = [n for n, _ in dst.named_parameters()]
    assert len(sp) == len(dp)
    # the original registers (bias, weight_g, weight_v) per weight-normed conv, ours (bias, original0, original1)
    for (a, pa), (b, pb) in zip(zip(sn, sp), zip(dn, dp)):
        assert pa.shape == pb.shape, (a, b, pa.shape, pb.shape)
        pb.data.copy_(pa.data)


for phy in (0, 3):
    orig = OrigTCN(4, 600, 5, 48, 9, n_feats=3, dropout=0.2, return_norm=True, latent_mode="repeat",
                   lab_dim=8, phy_dim=phy, feed_pp=False).eval()
    ours = VAE_TCN(4, 600, 5, 48, 9, n_feats=3, dropout=0.2, lab_dim=8, phy_dim=phy).eval()
    print(f"TCN phy_dim={phy}: params original {count(orig):,} ours {count(ours):,}")
    assert count(orig) == count(ours)
    if phy == 0:
        assert count(ours) == 300_897, "their best run (W&B b68z1hwo) reports 300,897 parameters"
    copy_by_order(orig, ours)
    x, lab = torch.rand(3, 600, 3), torch.eye(8)[[0, 3, 7]]
    p = torch.rand(3, phy)
    mo, lo = orig.encoder(x, label=lab, phy=p if phy else None)
    mu, lv = ours.encoder(x, lab, p if phy else None)
    assert torch.allclose(mo, mu, atol=1e-6) and torch.allclose(lo, lv, atol=1e-6)
    ho = orig.decoder(mo, x[:, :, 0], label=lab)
    hu = ours.decoder(mu, x[:, :, 0], lab)
    assert torch.allclose(ho, hu, atol=1e-6)

orig = OrigRNN(4, 600, 48, 2, rnn="gru", n_feats=3, dropout=0.2, lab_dim=8, phy_dim=0).eval()
ours = VAE_RNN(4, 600, 48, 2, n_feats=3, dropout=0.2, lab_dim=8).eval()
print(f"GRU: params original {count(orig):,} ours {count(ours):,}")
assert count(orig) == count(ours)
copy_by_order(orig, ours)
mo, _ = orig.encoder(x, label=lab)
mu, _ = ours.encoder(x, lab)
assert torch.allclose(mo, mu, atol=1e-6)
assert torch.allclose(orig.decoder(mo, x[:, :, 0], label=lab), ours.decoder(mu, x[:, :, 0], lab), atol=1e-6)
loss, mse, kld = vae_loss(x, ours.decoder(mu, x[:, :, 0], lab), mu, torch.zeros_like(mu), 0.75)
assert loss > 0 and kld >= 0
print("VAE PORT TEST PASSED (same parameter counts and identical outputs as the original code)")
