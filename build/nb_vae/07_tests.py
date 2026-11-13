# ---- Self-tests ----
from astra import tcn_vae, run_vae

count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
tcn = tcn_vae.VAE_TCN(4, 600, 5, 48, 9, 3, 0.2, 8, 0)
assert count(tcn) == 300_897, count(tcn)
tcn_p = tcn_vae.VAE_TCN(4, 600, 5, 48, 9, 3, 0.2, 8, 3)
gru = tcn_vae.VAE_RNN(4, 600, 48, 2, 3, 0.2, 8, 0)
print(f"parameters: TCN {count(tcn):,} (their run: 300,897) | TCN + physics {count(tcn_p):,} | GRU {count(gru):,}")

ds = pelsvae.make_dataset(lcs, meta, "")
tr, va = pelsvae.split_indices(len(ds["y"]))
dsp = pelsvae.make_dataset(lcs, meta, "PTA")
trp, vap = pelsvae.split_indices(len(dsp["y"]))
assert len(ds["y"]) == 65_760 and len(dsp["y"]) == 43_031 and len(set(tr) & set(va)) == 0
print(f"cVAE: train {len(tr):,} / val {len(va):,} | cVAE-P ({dsp['phy_names']}): train {len(trp):,} / val {len(vap):,}")
leak = len(set(meta.OGLE_id.values[va]) & set(meta.OGLE_id.values[tr])) / len(set(meta.OGLE_id.values[va]))
print(f"validation stars that also have a copy in training (their split): {100 * leak:.1f}%")

x = torch.from_numpy(ds["x"][:4]); lab = torch.eye(8)[torch.from_numpy(ds["y"][:4])]
for m, p in ((tcn, None), (gru, None), (tcn_p, torch.from_numpy(dsp["phy"][:4]))):
    xhat, mu, logvar, z = m(x, lab, p)
    assert xhat.shape == (4, 600, 1) and mu.shape == (4, 4)
    loss, mse, kld = tcn_vae.vae_loss(x, xhat, mu, logvar, 0.75)
    assert loss.item() > 0
lrs = [run_vae.HP["lr"]] + [run_vae.cosine_lr(e - 1, run_vae.HP) for e in range(2, 151)]
plt.figure(figsize=(5, 2.5)); plt.plot(range(1, 151), lrs); plt.xlabel("epoch"); plt.ylabel("learning rate")
plt.title("their cosine schedule, stepped with the epoch"); plt.tight_layout(); plt.show()
print("all self-tests passed")
del tcn, tcn_p, gru, ds, dsp
