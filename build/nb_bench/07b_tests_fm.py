# ---- Self-tests: flow matching ----
ratio = N_PARAMS["net"] / bench_data.G_PARAMS[L]
assert abs(ratio - 1) < 0.10, f"the velocity network should match the GAN generator's size (ratio {ratio:.2f})"
u1 = data_cpu["u"][idx]
# known answer: with the exact velocity field towards the curves u1, v = (u1 - x) / (1 - t), the solver must
# land on u1 for any number of steps (a straight path needs no small steps)
learnt, model.velocity = model.velocity, lambda x, t, cond, w: (u1 - x) / (1 - t)
for kw in ({"nfe": 1}, {"nfe": 8}, {"nfe": 8, "solver": "midpoint"}):
    assert torch.allclose(model.sample(idx, seed=1, **kw), data_cpu["x"][idx], atol=1e-4), kw
model.velocity = learnt
print(f"flow-matching tests passed | velocity network {N_PARAMS['net']:,} parameters "
      f"({100 * ratio:.0f}% of the GAN generator's {bench_data.G_PARAMS[L]:,})")
del data_cpu, model
