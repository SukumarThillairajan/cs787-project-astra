# ---- Train (or reuse) the feature CNN of every split in the queues ----
from astra import classifier

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for s in SEEDS:
    data = bench_data.load(NPZ, s, DEV)
    feat = bench_run.feature_cnn(RUNS, data, DEV, max_epochs=3 if SMOKE else 20)
    te = bench_data.subset(data, data["test_idx"])
    p = classifier.predict_proba(feat, classifier.stack_input(te["x"], te["ph"]), amp=False)[0].argmax(1).cpu().numpy()
    sc = bench_eval.summarise(data["y_np"][data["test_idx_np"]], p, len(CLASSES))
    curve = np.array(json.load(open(RUNS / f"feature_cnn_s{s}_curve.json")))
    print(f"split {s}: feature CNN trained {len(curve)} epochs | this single CNN on the test stars: "
          f"balanced accuracy {100 * sc['bal_acc']:.1f}%, accuracy {100 * sc['acc']:.1f}%")
    ax[0].plot(curve[:, 0], label=f"split {s} train (balanced)"); ax[0].plot(curve[:, 1], "--", label=f"split {s} validation")
    ax[1].plot(curve[:, 2], label=f"split {s}")
    del data, feat
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIGS / "feature_cnn_curves.png", dpi=110); plt.show()
