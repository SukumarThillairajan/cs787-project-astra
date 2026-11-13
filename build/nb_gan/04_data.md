## 1 · Data (Sec. 3 of the paper)

### Catalina Surveys DR1 periodic variables

The catalog (47,055 periodic variables with period, amplitude and type) and all their photometry (MJD, magnitude, error) are public on the Catalina Surveys site. The paper keeps 6 classes (its Table 1) and pre-processes every light curve in four steps:
1. **Period folding**: phase = (t mod T) / T, so the curve's shape becomes visible in [0, 1).
2. **Outlier filtering**: a sliding window of 20 points (in phase order) drops points with z-score > 3, done twice. Then a curve is dropped if more than 90% of its magnitudes lie outside the class median ± k·σ. The paper does not give k, and **k = 2 reproduces its class counts** (table below).
3. **Time sub-sampling**: keep L = 100 randomly chosen observations. This gives a fixed length, so convolutions can be used.
4. **Median centring**: subtract the curve's median magnitude, since the generator's tanh output is symmetric around 0.

### ZTF (only when `DATASET = "ztf"`)

The paper uses the periodic variable stars of ALeRCE's ZTF training set, with 40 points per curve. Neither their processed data nor their code is public, and the paper does not state the band, the dates, or where the period comes from. We recovered these from the paper itself: the real light curves it plots (Figs. 8 and 9, and the same in the first author's thesis) were digitised and matched against ALeRCE's detections, which identified all 21 stars shown. What they tell us:
- **One curve per star and band.** Each curve is single-band, and the dataset holds both g curves and r curves (14 and 7 of the 21).
- **Period:** the `period` column of ALeRCE's internal label file, i.e. the period given by the catalogue the label came from. It equals the folding period of all 21 curves to five digits. For eclipsing binaries it is the full orbital period (ALeRCE's own `Multiband_period` is half of it). Most Gaia-labelled stars have no period in that file (78% of the RR Lyrae, 74% of the LPVs); they cannot be folded and drop out, which is why the paper has so few of both.
- **Dates:** no plotted point is later than 19 July 2020 (MJD 59048.3), and many are that late: the data is one snapshot of that date.
- **Filtering:** only the magnitude threshold of 30.

So the dataset is rebuilt from public sources as follows:
- **Labels:** ALeRCE's labeled set (Sánchez-Sáez et al. 2021, Zenodo record 4279623): 85,809 periodic stars in the classes EB, RRL, LPV, DSCT and CEP.
- **Periods:** `ZTF_labels_v8.0.1.csv`, a public copy of ALeRCE's label file (Pérez-Carrasco et al. 2023, AnomalyALeRCE repository).
- **Light curves:** ALeRCE's public TAP service (table `ztf.detection`), corrected magnitudes below 30, up to MJD 59048.4, repeated epochs dropped. One curve per star and band with at least 40 points.
- Then, as for Catalina: fold, keep 40 random points, median-centre. No z-score or class-range filter, since the paper applies only the magnitude threshold to ZTF.

This rule gives EB 31,174, RRL 18,748 and LPV 5,470 curves, against the paper's 31,477, 18,729 and 5,245. For DSCT and CEP it gives 768 and 716 against the paper's 507 and 471, which we could not explain. We therefore **draw the paper's number of curves per class at random** where more qualify (for EB, all 31,174 are used). The class imbalance and the split sizes then equal the paper's; most stars should be the paper's too, but the 40 points of each curve are a new random draw.

Checked offline on the classifier trained on real data (row A), over random draws of the curves: this recipe gives the paper's accuracy (72.61) within the spread between draws, about 2 points.

The download takes over an hour, so it is done once offline (`build/ztf/download_ztf.py`, `add_label_periods.py`, then `make_ztf_dataset.py`) and the resulting `ztf_L40.npz` (24 MB) is attached as a Kaggle dataset.

### Split (Sec. 4.1)

Validation and test each hold 20% of the smallest class, *for every class*: 36 stars per class for Catalina (CEP has 181) and 94 curves per class for ZTF (CEP has 471). For ZTF the split is over curves, as in the paper, so the g and the r curve of one star can fall on different sides. Both sets are balanced, so accuracy equals balanced accuracy. Everything else is training data, still heavily imbalanced (Catalina: EBSD/D ≈ 30k vs CEP ≈ 109).
