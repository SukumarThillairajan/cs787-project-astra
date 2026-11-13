## 3 · The feature classifier (one per split)

As in the paper (Sec. 4.3), one CNN is trained on the real training curves of a split, and the output of its last conv block is the feature space in which density, recall, coverage and FID are measured. It is trained once per split and then reused by every job on that split.
