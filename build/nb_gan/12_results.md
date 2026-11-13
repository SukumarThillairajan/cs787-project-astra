## 5 · Results: Table 5 (ours vs paper) and the paper's figures

The cell below reads whatever has finished so far (so it also works on a paused session) and produces:
- **Table 5**: test accuracy, mean ± std over 15 runs, next to the paper's numbers, with two-sided Welch's t-test p-values as in the paper. Macro-F1 and per-class recall are added for the rarity analysis.
- **Fig. 3/4-style** curves: validation TSTR accuracy and FID over GAN training, and the γ comparison.
- **Fig. 5/7/11-style** plots: per-class density and recall, the G-score curve, and the density-recall plane with the two model-selection choices marked.
- **Fig. 9-style** samples: real curves and generated curves with the *same* condition (phases, amplitude, class).
- The TRTR classifiers' loss curves, to check over- and under-fitting.
