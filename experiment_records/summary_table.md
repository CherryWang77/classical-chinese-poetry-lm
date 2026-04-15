# Experiment Summary Table

| Experiment | Best practical checkpoint / region | Best validation loss | Final validation loss | Main judgement |
|---|---:|---:|---:|---|
| Real Song-ci baseline | around step 10000 | not singled out as one exact best saved checkpoint in the same way as later lines | smaller gains by step 15000 | earliest successful real Song-ci baseline and historical reference line |
| 6-layer / BS32 / Dim256 / Heads4 | step 10000 | 4.035873 | 4.092119 | strong 6-layer reference baseline |
| 6-layer / BS64 / Dim256 / Heads4 | step 7000 | 4.052221 | 4.416212 | useful batch-size comparison; BS64 did not clearly improve the line |
| 6-layer / BS32 / Dim256 / Heads8 | step 10000 (best observed region around 10600) | 4.021207 | 4.191617 | strongest balanced candidate |
| 6-layer / Dim384 / Heads4 | step 6000 (best observed region around 5800) | 4.026370 | 5.056804 | stronger peak than dim256/h4, but much stronger overfitting |
| 6-layer / BS32 / Dim384 / Heads8 | step 5000 | 3.974154 | 4.671842 | strongest peak candidate, but highest overfitting risk |