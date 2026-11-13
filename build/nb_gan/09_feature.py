# ---- Train (or reuse) the feature CNN ----
from astra import gan as gan_mod

FEAT = RUNS / "feature_cnn.pt"
data = gan_mod.load_data(NPZ, DEV)
tr, va, te = (gan_mod.subset(data, data[f"{s}_idx"]) for s in ("train", "val", "test"))
if not FEAT.exists():
    torch.manual_seed(0)
    feat, curve = classifier.train_feature_cnn(tr, va, len(CLASSES), max_epochs=3 if SMOKE else 20, device=DEV)
    torch.save(feat.state_dict(), FEAT)
    json.dump(curve, open(RUNS / "feature_cnn_curve.json", "w"))
feat = classifier.CNNEnsemble(1, len(CLASSES), L).to(DEV)
feat.load_state_dict(torch.load(FEAT, map_location=DEV)); feat.eval()
curve = np.array(json.load(open(RUNS / "feature_cnn_curve.json")))
p = classifier.predict_proba(feat, classifier.stack_input(te["x"], te["ph"]), amp=False)[0].argmax(1).cpu().numpy()
print(f"feature CNN: {len(curve)} epochs | test accuracy of this single CNN: {np.mean(p == te['y'].cpu().numpy()):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(curve[:, 0], label="train (balanced)"); ax[0].plot(curve[:, 1], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend()
ax[1].plot(curve[:, 2]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy")
fig.tight_layout(); fig.savefig(FIGS / "feature_cnn_curves.png", dpi=110); plt.show()
del data
