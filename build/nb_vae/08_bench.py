# ---- SMOKE only: time one training epoch of each architecture and estimate the queues ----
if SMOKE and DEV == "cuda":
    X = torch.from_numpy(lcs[:128 * 40]).to(DEV)
    lab = torch.eye(8, device=DEV)[torch.zeros(128, dtype=torch.long, device=DEV)]
    per_epoch = {}
    for arch, m in (("tcn", tcn_vae.VAE_TCN(4, 600, 5, 48, 9, 3, 0.2, 8, 0)),
                    ("gru", tcn_vae.VAE_RNN(4, 600, 48, 2, 3, 0.2, 8, 0))):
        m = m.to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
        for i in range(45):
            if i == 5:
                torch.cuda.synchronize(); t = time.time()
            xb = X[(i % 40) * 128:(i % 40 + 1) * 128]
            loss = tcn_vae.vae_loss(xb, *m(xb, lab)[:3], 0.75)[0]
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        step = (time.time() - t) / 40
        per_epoch[arch] = step * (52_608 / 128) * 1.15              # + validation
        print(f"{arch}: {step * 1000:.0f} ms/step -> ≈ {per_epoch[arch] / 60:.1f} min/epoch -> "
              f"≈ {150 * per_epoch[arch] / 3600:.1f} h for 150 epochs")
    for gpu, q in QUEUES.items():
        h = sum(150 * per_epoch["gru" if j.startswith("gru") else "tcn"] / 3600 for j in q)
        print(f"GPU {gpu}: {q} ≈ {h:.1f} h")
    json.dump(per_epoch, open(TABLES / "benchmark.json", "w"))
    del X, m, opt
