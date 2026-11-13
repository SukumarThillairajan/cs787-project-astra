# ---- 5 · Which classes does a classifier trained on real data confuse? (one small CNN per dataset, CPU) ----
for name in Z_ALL:
    def run():
        conf, acc = audit.confusion_and_map(NPZ[name], name, FIGS / f"{name}_confusion.png", seed=SEED)
        show(f"{name}: confusion matrix of one CNN on the test stars, % of each true class (rows)", conf,
             f"{name}_confusion.csv", True)
        print({k: round(v, 1) for k, v in acc.items()})
    safely(f"{name} confusion", run)
