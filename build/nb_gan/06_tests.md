## 2 · Self-tests (seconds)

Cheap checks that would otherwise only show up hours into a run:
- the resampler draws classes with the probabilities of Eq. 12;
- the grouped network gives exactly the same outputs as separate CNNs, and one member never sees another member's data;
- warping keeps phases in order and inside the window;
- the metrics behave as expected on a copy, an independent sample and a collapsed generator;
- G and D produce the right shapes, and the gradient penalty back-propagates.
