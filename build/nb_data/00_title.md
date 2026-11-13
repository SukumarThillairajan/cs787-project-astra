# ASTRA · Notebook 00 — The datasets: an audit in figures

**CS787 (Generative AI) project, Group 4.** This notebook needs **no GPU** (set Accelerator to *None*; it costs no GPU quota) and runs in about 10 minutes.

**Goal.** Show what the data look like and check, with numbers, the reasons behind the choices of the benchmark notebooks (03–06). Each section answers one question:

| Section | Question | Used for |
|---|---|---|
| 1 | Which datasets, how many curves, how many classes? | the data section of the report |
| 2 | How unbalanced are the classes, and which are rare? What does a curve of each class look like? What does pre-processing do to a raw star? | motivation: rare classes |
| 3 | What do period and amplitude say about the class? How irregular is the sampling in phase? How noisy are the curves? | the conditions given to the models; the amplitude-normalised target; irregular sampling |
| 4 | Do curves of one star sit on both sides of a split? How finely can each test set measure a class? | why the benchmark splits over stars and tests on 20% of them |
| 5 | Which classes does a classifier trained on real data confuse? | where augmentation has room to help |
| 6 | What is in the PELS-VAE training file (OGLE-III)? | the leakage found in notebook 02 |

**Data.** The pre-processed files of notebook 01: `catalina_L100.npz` (built here from the public Catalina release if it is not attached, about 3 minutes) and `ztf_L40.npz` (attach the dataset that holds it; without it the ZTF parts are skipped). Section 6 downloads the PELS-VAE file from Zenodo (296 MB).

**How to run.** Accelerator *None*, Internet *On*, Run All. Figures go to `figures_data/`, tables to `tables_data/`.
