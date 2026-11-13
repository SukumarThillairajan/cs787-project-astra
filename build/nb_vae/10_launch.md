## 3 · Training: one worker per GPU

Each GPU runs `python -m astra.run_vae` on its queue, with the settings of their best run: Adam at lr 10⁻³ with their cosine schedule, batch 128, 150 epochs, β = 0.75, latent size 4, dropout 0.2. After every epoch the worker records the training and validation loss (and its MSE and KL parts) and saves a resumable checkpoint.

After the last epoch, it saves the final weights, the latent mean and standard deviation of all 65,760 curves (for Fig. 6 and Table 3), and reconstructions of 3 validation curves per class (Figs. 5 and 7). In the real run, a short canary with tiny jobs runs first.

Time: a worker pauses its job 15 minutes before the limit (`SESSION_HOURS` minus 0.4 h, kept for the results cells) and the next session continues it. A worker that is still alive at the limit is stuck and is stopped, so the results cells always run. When every job is already done, no worker starts.
