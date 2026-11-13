# ---- 4 · Splits: curves of one star on both sides, and how finely a test set measures a class ----
for name, Z in Z_ALL.items():
    show(f"{name}: stars with several curves, and held-out curves whose star is also in training",
         safely(f"{name} leakage", lambda: audit.leakage_table(Z, SEED)), f"{name}_leakage.csv")
    show(f"{name}: test-set size per class and what it can resolve (paper's split against ours)",
         safely(f"{name} power", lambda: audit.power_table(Z, SEED)), f"{name}_test_power.csv")
