## 7 · Observations

From the run of 6 October 2026 (CPU, both files as used by notebook 01; split 0 of the benchmark). All numbers are in `tables_data/`.

**The datasets.** Catalina: 43,417 curves, 6 classes, 100 points per curve. ZTF: 56,135 curves of 30,125 stars, 5 classes, 40 points per curve. Both are dominated by eclipsing binaries: the largest class is 166 × the smallest on Catalina and 66 × on ZTF.

**Rare classes.** After a 70/10/20 split the rare tier has 347 (LPV), 172 (DSCT) and 127 (CEP) training curves on Catalina, and 365 (DSCT) and 329 (CEP) on ZTF. Together they are 2.1% of Catalina and 1.7% of ZTF.

**Period and amplitude.** The period alone almost separates the classes: DSCT near 0.09 days, the binaries and RR Lyrae between 0.3 and 0.9 days, Cepheids at 2–4 days, LPVs at 200–300 days. Amplitudes overlap (medians 0.24–0.42 mag) except for the LPVs (0.9 mag on Catalina, 1.7 on ZTF). The paper's classifier sees magnitudes and phases only; the period is used up by the folding. So a class that is easy to tell by its period can still be hard for this classifier, and the benchmark, which follows the paper, measures augmentation under that handicap. This belongs in the report's limitations.

**Scale.** Magnitudes span −3.7 to 4.4 mag on Catalina and −6.7 to 12.4 on ZTF; divided by each curve's amplitude they lie in [−2, 2] for every class. This is why the benchmark's target is magnitude / amplitude.

**Sampling in phase.** Catalina is close to what random phases would give: the median largest gap is 0.05–0.08 of a cycle against 0.05 for 100 random phases, and under 2% of the curves have a gap above 0.2. ZTF is much less regular: the median largest gap is 0.13–0.20 against 0.10 for 40 random phases, and a gap above 0.2 appears in 8% of the EB curves, 28% of RRL and DSCT, 17% of CEP and 50% of the LPVs. If telling a model the phases matters, it should show on ZTF; the phase ablation (`fm_nophase_s0`, notebook 06) should be read there first.

**Noise.** Catalina is noisy: the median photometric error is 0.19–0.27 of the amplitude for every class except the LPVs (0.06). ZTF is clean: 0.03–0.04. A generated Catalina curve without noise would not look like a real one, which is why the benchmark's VAE learns a noise level and adds it when sampling.

**Splits.**
- *ZTF, the paper's split (over curves):* 42% (CEP) to 85% (RRL) of the held-out curves belong to a star that also has a curve in training (the same star in the other band, or under a second object id). In our split over stars it is 0%.
- *Catalina:* every curve is its own object. 19% of the held-out EBSD/D curves share their period value with a training curve, but with 30,000 contact binaries between 0.2 and 0.5 days that is what chance gives, not the same star twice. Grouping by period is only the cautious choice there.
- *Resolution:* the paper tests on 36 curves per class on Catalina (one curve = 2.8 points; a recall of 75% has a standard error of 7.2 points) and 94 on ZTF (4.5 points). Our test set brings the common classes to 0.6–1.4 points. For the rarest classes it cannot do better than the data: 36 CEP and 49 DSCT test curves on Catalina, 96 and 89 on ZTF, so their recalls stay uncertain by 4–7 points on one split. Three splits and the paired bootstrap are how the benchmark deals with that.

**Which classes are confused** (one CNN, six epochs, not the benchmark's ensemble).
- Catalina: accuracy 88.8% but balanced accuracy 70.0%. Recall is 86–91% for the four larger classes, 59% for DSCT and **8% for CEP**: 64% of the Cepheids are called RR Lyrae and 17% contact binaries; 35% of the DSCT are called RR Lyrae.
- ZTF: accuracy 90.5%, balanced accuracy 65.2%. Recall 83–97% for EB, RRL, LPV, 45% for DSCT and **11.5% for CEP** (60% called RR Lyrae, 16% LPV).
- So the rare classes are not spread over all classes: they are absorbed by RR Lyrae, whose folded shape they resemble. This is where synthetic curves have room to help, and the per-class recall of CEP and DSCT is the number to watch in notebooks 03–06.
- On ZTF this single CNN is about 7 points below the paper's row A (72.6%), on Catalina about 3 below (73.4%). The paper's rows are ensembles on its own split; whether the larger gap on ZTF comes from the leakage above is a question for the `real_s` jobs of the benchmark, not answered here.

**The PELS-VAE file (OGLE-III).** 65,760 curves of 33,108 stars. The small classes exist mostly as copies: 70 curves per star for ACEP (71 stars), 121 for DSCT (42 stars), 12 for T2CEP. In their random split every validation star of ACEP, DSCT and T2CEP, and 98% of ELL, has a copy in training; overall 47.5%. Only 1.4% of the ACEP curves and 29% of the RR Lyrae have all three physical parameters, so the cVAE-P of notebook 02 is trained on a different mix of classes than the cVAE.

