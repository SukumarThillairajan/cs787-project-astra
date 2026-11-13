## 4 · Training: one worker per GPU

Each GPU runs `python -m astra.run_gan` on its queue:
- `trtr` trains the classifiers of rows A–D (15 runs × 5 CNNs each, on real data). The 75 CNNs of a row train side by side as one network with a member axis.
- A `gan_*` job trains one GAN for 400k generator iterations, saving the EMA generator every 10k iterations. For every checkpoint it measures FID, per-class density and recall (5 repetitions), and the validation TSTR accuracy (3 runs × 5 CNNs trained on 43k generated curves).
- At the end, the G-score is computed over all checkpoints. The best checkpoint by each criterion (validation accuracy, G-score) gets the final TSTR test score: 15 runs × 5 CNNs.

In the real run, a 2-minute *canary* first runs a tiny version of each GPU's first job, so a broken setup fails early instead of after hours. The cell prints a short status line every 10 minutes; full logs are in `runs/<job>/log.txt`.
