## 4 · Results: Figs. 4–7 (ours vs paper)

- **Fig. 4**: validation loss per epoch, with the paper's approximate epoch-100 values marked. Training and validation loss are shown side by side for every model, to check for over- or under-fitting. Remember that the validation curves share stars with the training set (their split).
- The table also gives each model's lowest validation loss and its epoch: their schedule raises the learning rate again after epoch 50, and a model can get worse after that.
- **Figs. 5 and 7**: reconstructions from the latent *mean*, for 3 validation curves per class, with and without physical parameters, and for the second seed.
- **Fig. 6**: the latent means of all curves, coloured by class. We also report how close the latent codes are to the N(0, I) prior. With the KL term divided by 10⁴ we expect small posterior σ, which matters later: sampling new curves from the prior is only safe if the codes actually fill N(0, I).
