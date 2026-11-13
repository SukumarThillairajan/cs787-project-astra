# ---- SMOKE only: time the real-size jobs and estimate each GPU queue's hours ----
if SMOKE:
    from astra import run_gan
    data = gan_mod.load_data(NPZ, DEV)
    b = run_gan.benchmark(data, feat, DEV, WORK / "bench_tmp", LATENT)
    shutil.rmtree(WORK / "bench_tmp", ignore_errors=True)
    del data
    c = run_gan.GAN
    n_ckpt = c["iters"] // c["ckpt_every"]
    ep_sel, ep_fin = b[f"clf_epoch_s_{c['sel_runs'] * c['ens']}cnn"], b[f"clf_epoch_s_{c['final_runs'] * c['ens']}cnn"]
    ep_sel32 = b[f"clf_epoch_s_{c['sel_runs'] * c['ens']}cnn_fp32"]
    SEL_EPOCHS, FIN_EPOCHS = 4, 5            # typical epochs before early stopping ends a training
    est = {"gan training": c["iters"] / b["gan_it_per_s"] / 3600,
           "checkpoint evaluations": n_ckpt * (b["ckpt_eval_1epoch_s"] + (SEL_EPOCHS - 1) * ep_sel) / 3600,
           "final TSTR (2 criteria)": 2 * FIN_EPOCHS * ep_fin / 3600}
    gan_h = sum(est.values())
    trtr_h = 4 * FIN_EPOCHS * ep_fin * 1.1 / 3600
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
    print(f"\nmixed precision: GAN {b['gan_it_per_s']:.1f} it/s (full precision {b['gan_it_per_s_fp32']:.1f}), "
          f"{b['amp_skipped_D_steps']:.0f} skipped D steps in 80; classifier epoch (15 CNNs) {ep_sel:.1f} s "
          f"(full precision {ep_sel32:.1f} s)")
    print("one GAN job ≈ " + ", ".join(f"{k} {v:.1f} h" for k, v in est.items()) + f" -> {gan_h:.1f} h")
    print(f"trtr (rows A-D) ≈ {trtr_h:.1f} h")
    for gpu, q in QUEUES.items():
        tot = sum(trtr_h if j == "trtr" else gan_h for j in q)
        print(f"GPU {gpu}: {q} ≈ {tot:.1f} h  (≈ {tot / (SESSION_HOURS - 0.7):.1f} sessions)")
    json.dump({"bench": b, "gan_job_h": gan_h, "trtr_h": trtr_h}, open(TABLES / "benchmark.json", "w"), indent=1)
