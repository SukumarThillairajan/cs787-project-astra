## 6 · Observations (to fill in after the real run)

**Does it beat real data only?** *(Placeholder: TSTR and real + synthetic balanced accuracy against row A, per split, with the paired intervals. Does an interval exclude 0 on all five splits? Which mix of real and synthetic curves did validation choose?)*

**Rare classes.** *(Placeholder: recall of the rare-tier classes for row A, TSTR and real + synthetic. This is the question ASTRA is about. Do the common classes pay for it?)*

**Fidelity and diversity.** *(Placeholder: density / precision against recall / coverage per class; FID; the amplitude error. Which classes are covered worst?)*

**Over/under-fitting and memorisation.** *(Placeholder: does the validation TSTR score peak and then fall? Does the model's loss on validation stars rise above its loss on training stars? Do the rare-tier copies rise above 5%, and when? Did selection pick an early checkpoint? Classifier curves: training against validation loss.)*

**New conditions.** *(Placeholder: TSTR for new conditions against the training stars' own, at the selected and at the last checkpoint. Does the last checkpoint, which copies, still generate useful curves for conditions it has not seen?)*

@@OBS@@

**Training length and compute.** *(Placeholder: did the runs stop by 'no gain' or at the 20 h cap? At how many hours of training on one T4 did each split peak? The compute-matched view: which family is ahead at 5, 10, 15, 30, 60 minutes and 2, 5, 10 h, and does the order change?)*

**The search.** *(Placeholder: the configurations on split 0, validation only: how far apart are they, compared with the ± of one checkpoint's score? Which knob mattered?)*

**Split-to-split spread.** *(Placeholder: how much do the scores move between splits 0–4, compared with the differences between methods?)*

**Deviations and their possible effect.** *(Placeholder: the amplitude-normalised target, the stopping rule and its caps, 3 selection runs per checkpoint, a search of six configurations on one split, mixed precision, the "ours" choices of the harness.)*

