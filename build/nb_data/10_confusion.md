## 5 · Which classes are confused?

One CNN (the paper's base classifier) is trained on the real training curves with class-balanced batches and scored on the test stars of the benchmark's split. Rows of the matrix are true classes; a row shows where that class's curves go. The map on the right is a t-SNE of the classifier's features: classes that overlap there are the ones it cannot separate. It is one small network, not the 15 × 5 ensemble of the benchmark, so its numbers are only a guide.
