## 2 · Self-tests (seconds)

Small checks with a known answer, run before any GPU time is spent:
- balanced accuracy of a constant guess is 1 / (number of classes), and the paired bootstrap agrees;
- the model of this notebook: parameter counts, one training step, sampling repeats exactly for a fixed seed, and a check of the sampler against a case with a known answer.
