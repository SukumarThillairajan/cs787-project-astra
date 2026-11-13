## 3 · What the models are given, and what they have to produce

A model receives a real star's phases, amplitude and class, and generates the magnitudes.
- **Period and amplitude.** The left panel is the classic way to tell variable stars apart. The period itself is not given to the models: it is used up by the folding.
- **Sampling in phase.** The largest empty stretch of the phase circle for every curve, against what the same number of uniformly random phases would give. A gap much larger than that means the star's period is close to a multiple of the observing cadence.
- **Noise.** The median photometric error of a curve, and the same divided by the curve's amplitude: how much of the curve is signal.
- **Scale.** Raw magnitudes differ in scale by class; divided by the amplitude they do not. This is why the benchmark's target is magnitude / amplitude.
