# Data layout

```
data/test/{clear,partial,blocked}/    real photos used ONLY for evaluation
data/train/{clear,partial,blocked}/   real photos used for training the optional model (keep disjoint from test)
data/synthetic/                       generated demo images (smoke tests only - not evidence)
```

Label guide used by the detector: **clear** < ~8 % of the grate covered by debris, **partial** ~8-30 %,
**blocked** > ~30 %. Label your photos by eye using the same rule so the evaluation is consistent.
Record where and when each photo was taken (Southbank, date, wet/dry) in your notes for the log.
