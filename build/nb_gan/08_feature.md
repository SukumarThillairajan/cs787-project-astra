## 3 · The feature classifier (Sec. 4.3)

Density, recall, FID and the G-score all compare real and generated curves in a **feature space**: the output of the last convolution block (4 × 64 = 256 numbers) of one base CNN trained on the balanced real training set. It is trained once, here, and shared by all GAN jobs so their metrics are comparable. Its training and validation losses are plotted below to check for over- or under-fitting.
