"""Assemble the Kaggle notebooks from build/nb_*/ cells and the astra/ package (the source of truth).
Never hand-edit the .ipynb files: edit the sources and rebuild.

    python build/build_notebooks.py            # writes all notebooks into notebooks/
    python build/build_notebooks.py BENCH      # only those whose file name contains BENCH

The four benchmark notebooks (03-06) share the cells of build/nb_bench/; what differs per model family is the
title cell, one self-test cell, the family's modules, and the @@...@@ fields filled in below.
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PKG = ROOT / "astra"

SHARED = ["@__init__.py", "@common.py", "@catalina.py", "@ztf.py", "@sampling.py", "@classifier.py", "@warping.py",
          "@metrics.py"]


sys.path.insert(0, str(ROOT))
from astra import bench_grid  # noqa: E402  (pure Python: the grids, the winners, the measured speeds)


def bench(fam, modules, modules_md, obs):
    order = ([f"00_title_{fam}.md", "00b_protocol.md", "01_config.py", "02_env.py", "03_package.md"] + SHARED
             + ["@bench_data.py", "@bench_eval.py", "@bench_grid.py", "@bench_run.py"] + [f"@{m}.py" for m in modules]
             + ["@bench_report.py"]
             + ["04_data.md", "05_data.py", "06_tests.md", "07_tests.py", f"07b_tests_{fam}.py", "08_feature.md",
                "09_feature.py", "09b_bench.py", "10_launch.md", "11_launch.py", "12_results.md", "13_results.py",
                "14_checks.py", "15_observations.md"])
    jobs = {"GRID_TAGS": repr(list(bench_grid.GRID[fam])), "FINAL_TAG": repr(bench_grid.FINAL[fam])}
    return "nb_bench", order, {"FAMILY": fam, "MODULES": modules_md, "OBS": obs, **jobs}


NOTEBOOKS = {
    "ASTRA_00_DATA_audit.ipynb": (
        "nb_data",
        ["00_title.md", "01_config.py", "@__init__.py", "@common.py", "@catalina.py", "@sampling.py", "@classifier.py",
         "@pelsvae.py", "@bench_data.py", "@data_audit.py",
         "02_data.md", "03_data.py", "04_classes.md", "05_classes.py", "06_conditions.md", "07_conditions.py",
         "08_split.md", "09_split.py", "10_confusion.md", "11_confusion.py", "12_ogle.md", "13_ogle.py",
         "14_checks.py", "15_observations.md"]),
    "ASTRA_01_GAN_GarciaJara_replication.ipynb": (
        "nb_gan",
        ["00_title.md", "01_config.py", "02_env.py", "03_package.md"] + SHARED + ["@gan.py", "@run_gan.py",
         "04_data.md", "05_data.py", "06_tests.md", "07_tests.py", "08_feature.md", "09_feature.py", "09b_bench.py",
         "10_launch.md", "11_launch.py", "12_results.md", "13_results.py", "14_checks.py", "15_observations.md"]),
    "ASTRA_02_VAE_PELSVAE_replication.ipynb": (
        "nb_vae",
        ["00_title.md", "01_config.py", "02_env.py", "03_package.md",
         "@__init__.py", "@common.py", "@pelsvae.py", "@tcn_vae.py", "@run_vae.py",
         "04_data.md", "05_data.py", "06_tests.md", "07_tests.py", "08_bench.py",
         "10_launch.md", "11_launch.py", "12_results.md", "13_results.py", "14_regression.md", "15_regression.py",
         "15b_prior.md", "15c_prior.py", "16_checks.py", "17_observations.md"]),
    "ASTRA_03_BENCH_GAN.ipynb": bench(
        "gan", ["gan", "bench_gan"],
        "- `gan`: the paper's generator and critic, as in notebook 01.\n"
        "- `bench_gan`: **this notebook's model**: the GAN's training step and its sampler.",
        "**GAN-specific.** *(Placeholder: critic loss and the W-distance estimate over training; learning rate and "
        "batch size in the search; where the validation score peaks (notebook 01 found 10–20k iterations on the "
        "paper's split) and whether the rare-tier copies rise after it, as they did in notebook 01.)*"),
    "ASTRA_04_BENCH_VAE.ipynb": bench(
        "vae", ["gan", "bench_vae"],
        "- `gan`: the paper's generator and critic, reused here as decoder and encoder.\n"
        "- `bench_vae`: **this notebook's model**: encoder, decoder with a noise level, the ELBO step and the sampler.",
        "**VAE-specific.** *(Placeholder: the learnt noise level σ and the KL term over training; is z used, or is the "
        "KL near 0? Sampling without the noise (`noise = False`, the classic smooth VAE output): how much does TSTR "
        "lose? β = 0.1, 1 and 4 in the search: reconstruction against a latent space that matches the prior.)*"),
    "ASTRA_05_BENCH_DDPM.ipynb": bench(
        "ddpm", ["bench_net", "bench_ddpm"],
        "- `bench_net`: the 1-D U-Net shared by diffusion and flow matching.\n"
        "- `bench_ddpm`: **this notebook's model**: the noise schedule, the training step and the DDIM sampler.",
        "**Diffusion-specific.** *(Placeholder: score against the number of sampling steps (4, 8, 16, 32, 64) and the "
        "sampling time; does guidance (1.5, 2) help the classifier, and what does it cost in recall? Noise prediction "
        "against v-prediction and the learning rate in the search.)*"),
    "ASTRA_06_BENCH_FM.ipynb": bench(
        "fm", ["bench_net", "bench_fm"],
        "- `bench_net`: the 1-D U-Net shared by diffusion and flow matching.\n"
        "- `bench_fm`: **this notebook's model**: the straight-line path, the training step and the ODE sampler.",
        "**Flow-matching-specific.** *(Placeholder: score against the number of Euler steps (4, 8, 16, 32, 64): does "
        "it need fewer steps than diffusion (notebook 05) for the same score, as straight paths suggest? Euler against "
        "midpoint at equal cost; guidance; uniform against logit-normal t.)*"),
}


def build(name, folder, order, subs=None):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    for item in order:
        if item.startswith("@"):
            src = f"%%writefile astra/{item[1:]}\n" + (PKG / item[1:]).read_text(encoding="utf-8")
            nb.cells.append(nbformat.v4.new_code_cell(src.rstrip() + "\n"))
            continue
        text = (BUILD / folder / item).read_text(encoding="utf-8").rstrip()
        for key, value in (subs or {}).items():
            text = text.replace(f"@@{key}@@", value)
        assert "@@" not in text, f"unfilled field in {folder}/{item}"
        nb.cells.append(nbformat.v4.new_markdown_cell(text) if item.endswith(".md") else nbformat.v4.new_code_cell(text))
    for i, cell in enumerate(nb.cells):          # fixed ids, so a rebuild without changes gives no git diff
        cell.id = f"{folder}-{i:02d}"
    return nb


if __name__ == "__main__":
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    which = sys.argv[1:] or list(NOTEBOOKS)
    for name in NOTEBOOKS:
        if not any(w in name for w in which):
            continue
        nb = build(name, *NOTEBOOKS[name])
        nbformat.write(nb, out_dir / name)
        print("wrote", out_dir / name, len(nb.cells), "cells")
