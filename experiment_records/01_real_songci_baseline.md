# Experiment: Real Song-ci baseline and continuation line

## 1. Purpose
This experiment established the first main real training line on the final Song-ci corpus and was used to verify that the project could move from corpus preparation to stable poetry-oriented language-model training on the selected dataset.

## 2. Configuration
- Corpus: songci_main_5mb.txt
- Context length: 256
- Batch size: 32
- Hidden dimension: 256
- Number of layers: 4
- Number of heads: 4
- Dropout: 0.1
- Learning rate: 3e-4
- Training steps: baseline 0–5000, continuation to 10000, and continuation to 15000
- Device: cuda

## 3. Main comparison target
This line serves as the earliest main Song-ci real-experiment reference line and provides the historical baseline for later 6-layer comparisons.

## 4. Key quantitative results
- Baseline run (0–5000) trained stably and improved validation substantially
- Continuation from 5000 to 10000 improved the line further
- Continuation from 10000 to 15000 remained stable, but gains became smaller and less decisive than in earlier stages
- The strongest practical checkpoint region in this line appeared around step 10000 rather than at the final continuation endpoint

## 5. Key log excerpts
```text
=== Continue real experiment training to 15000 (Colab) ===
Device: cuda
Context length: 256
Batch size: 32
Embed dim: 256
Num layers: 4
Num heads: 4
Dropout: 0.1
Learning rate: 0.0003

Step 10800 | train loss = 3.588542 | train BPC = 5.177172 | avg val loss = 4.070608 | avg val BPC = 5.872646
Step 11000 | train loss = 3.620312 | train BPC = 5.223006 | avg val loss = 4.064797 | avg val BPC = 5.864262
Step 12400 | train loss = 3.638612 | train BPC = 5.249407 | avg val loss = 4.075722 | avg val BPC = 5.880024
Step 12800 | train loss = 3.590711 | train BPC = 5.180301 | avg val loss = 4.074052 | avg val BPC = 5.877615
Step 15000 | train loss = 3.404876 | train BPC = 4.912197 | avg val loss = 4.070005 | avg val BPC = 5.871775
```

## 6. Qualitative sample evidence

### Earlier strong continuation sample region
```text
气和玉烛，睿化著鸿明。
缇管一阳生。

秋色早天空，正钱塘草木，还是秋光。
不见老人无此日，十分春、几度更无愁。
君心肯似我，长是江南第一，柳带春风。
但无人问我，不如今、不老为盟。

小圃池亭。
风撼松江。
正莼头一笛，江空没个人声。
江湖正好，一洗一襟愁。
```

### Later continuation sample tendency
```text
The later continuation stages remained recognisably poetic and stable, but the qualitative gains after the strongest earlier region were smaller than the gains observed in the baseline-to-10000 development. The line appeared to be approaching a practical plateau rather than collapsing.
```