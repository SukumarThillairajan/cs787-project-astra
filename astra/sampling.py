"""The resampling block of Garcia-Jara et al. (2022), Sec. 4.6.

Each class keeps its own shuffled queue. A class is picked with probability
    p_i = gamma / N + (1 - gamma) * S_i / S          (Eq. 12)
and the next sample is taken from its queue; an empty queue is reshuffled.
gamma = 0 keeps the original class mix, gamma = 1 gives a uniformly balanced stream.
"""
import numpy as np


class ClassResampler:
    def __init__(self, y, gamma, seed=0, n_classes=None):
        y = np.asarray(y)
        self.n_classes = n_classes or int(y.max()) + 1
        self.rng = np.random.default_rng(seed)
        self.pools = [np.flatnonzero(y == c) for c in range(self.n_classes)]
        sizes = np.array([len(p) for p in self.pools], float)
        self.p = gamma / self.n_classes + (1 - gamma) * sizes / sizes.sum()
        self.p[sizes == 0] = 0
        self.p /= self.p.sum()
        self.queues = [self.rng.permutation(p) for p in self.pools]
        self.pos = np.zeros(self.n_classes, int)

    def _take(self, c, k):
        out = []
        while k > 0:
            if self.pos[c] == len(self.queues[c]):
                self.queues[c] = self.rng.permutation(self.pools[c])
                self.pos[c] = 0
            n = min(k, len(self.queues[c]) - self.pos[c])
            out.append(self.queues[c][self.pos[c]:self.pos[c] + n])
            self.pos[c] += n
            k -= n
        return np.concatenate(out)

    def sample(self, batch):
        """Indices of one batch (classes drawn with p, samples without replacement inside a class)."""
        cls = self.rng.choice(self.n_classes, size=batch, p=self.p)
        idx = np.empty(batch, np.int64)
        for c in np.unique(cls):
            where = np.flatnonzero(cls == c)
            idx[where] = self._take(c, len(where))
        return idx
