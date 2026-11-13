## 1 · Data and split

The curves are those of notebook 01 (period folding, outlier filtering, 100 or 40 random points, median-centred), so any difference to its numbers comes from the protocol, not from the data. If a ready `catalina_L100.npz` or `ztf_L40.npz` is attached under *Add Input* it is used; otherwise Catalina is downloaded and pre-processed here (about 3 minutes). **Attach the same file to all four notebooks**: the results carry a hash of the data and split, and the comparison refuses to pair results with different hashes.

The table shows the curves per class and side for split 0, and which classes form the rare tier. The figure shows, per class, a real curve in magnitudes (top) and the amplitude-normalised curve the models are trained on (bottom).
