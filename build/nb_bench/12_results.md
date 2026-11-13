## 5 · Results

**Search stage** (split 0, validation stars only): one row per configuration, best first, with the hour of training at which it peaked, why it stopped (no gain or the cap), and its rare-tier copies. The best row becomes `FINAL` in `astra/bench_grid.py`. Test scores are not computed in the search.

**Final stage**
- *Reference rows on real data* (A = oversampling, B–D = soft warping), per split.
- *One row per split*: the selected checkpoint and when it was reached, TSTR and real + synthetic balanced accuracy (± over the 15 classifier runs), the difference to row A of the same split with a 95% paired-bootstrap interval, **the rare-tier copies and memorisation ratio next to them**, fidelity and sampling time. An interval that excludes 0 is a difference the test stars can support.
- *The compute-matched view*: the TSTR test score of the best checkpoint within 5, 10, 15, 30, 60 minutes and 2, 5, 10 h of training, and at the family's own best point.
- *Conditions no training star has*: TSTR (3 classifier runs) at the selected and at the last checkpoint, for the training stars' own conditions and for new ones. A model that only copies its training curves scores well on its own conditions and badly on new ones.
- *Per-class recall*, classes from most to least common: the question of the project is the bottom rows.
- *Other sampling settings* of the selected checkpoints (3 classifier runs each): sensitivity checks, never picked by their test score.

**Figures:** validation score, FID, recall and density, rare-tier copies and the model's own losses against hours of training; the budget figure; real and generated curves with the same condition; the final classifiers' loss curves. **The four-way comparison** appears once the final outputs of the other benchmark notebooks are attached.
