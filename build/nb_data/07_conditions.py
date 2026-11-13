# ---- 3 · Period, amplitude, sampling in phase, noise ----
for name, Z in Z_ALL.items():
    show(f"{name}: period and amplitude per class",
         safely(f"{name} period/amplitude", lambda: audit.plot_period_amplitude(Z, name, FIGS / f"{name}_period_amplitude.png")),
         f"{name}_period_amplitude.csv")
    show(f"{name}: largest phase gap per curve",
         safely(f"{name} sampling", lambda: audit.plot_sampling(Z, name, FIGS / f"{name}_sampling.png")),
         f"{name}_sampling.csv")
    show(f"{name}: photometric error per class",
         safely(f"{name} noise", lambda: audit.plot_noise(Z, name, FIGS / f"{name}_noise.png")),
         f"{name}_noise.csv")
    u = Z["mag"] / Z["amp"][:, None]
    print(f"\n{name}: magnitudes span {Z['mag'].min():.2f} to {Z['mag'].max():.2f} mag (std {Z['mag'].std():.3f}); "
          f"divided by each curve's amplitude they span {u.min():.2f} to {u.max():.2f} (std {u.std():.3f}) for every class")
