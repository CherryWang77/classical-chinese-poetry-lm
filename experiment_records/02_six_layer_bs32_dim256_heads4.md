# Experiment: 6-layer / BS32 / Dim256 / Heads4

## 1. Purpose
This experiment was designed to establish a clear 6-layer reference line under the batch size 32 / hidden dimension 256 / heads 4 configuration and to serve as the main comparison baseline for later controlled tuning experiments.

## 2. Configuration
- Corpus: songci_main_5mb.txt
- Context length: 256
- Batch size: 32
- Hidden dimension: 256
- Number of layers: 6
- Number of heads: 4
- Dropout: 0.1
- Learning rate: 3e-4
- Training steps: 15000
- Device: cuda

## 3. Main comparison target
This line mainly serves as a reference line for later comparisons against:

- 6-layer / BS64 / Dim256 / Heads4
- 6-layer / BS32 / Dim256 / Heads8
- 6-layer / Dim384 / Heads4

## 4. Key quantitative results
- Best observed validation region: around step 10000
- Validation loss at step 10000: 4.035873
- Validation BPC at step 10000: 5.822534
- Best practical saved checkpoint: step 10000
- Final checkpoint: step 15000
- Final validation loss: 4.092119
- Final validation BPC: 5.903680

## 5. Key log excerpts
```text
=== Continue 6-layer training to 10000 (Colab) ===
Device: cuda
Context length: 256
Batch size: 32
Embed dim: 256
Num layers: 6
Num heads: 4
Dropout: 0.1
Learning rate: 0.0003

Step 07400 | train loss = 3.789575 | train BPC = 5.467201 | avg val loss = 4.058059 | avg val BPC = 5.854542
Step 07800 | train loss = 3.734446 | train BPC = 5.387666 | avg val loss = 4.052771 | avg val BPC = 5.846913
Step 08400 | train loss = 3.698603 | train BPC = 5.335956 | avg val loss = 4.074890 | avg val BPC = 5.878823
Step 10000 | train loss = 3.447323 | train BPC = 4.973436 | avg val loss = 4.035873 | avg val BPC = 5.822534

=== Continue 6-layer training to 15000 (Colab) ===
Step 15000 | train loss = 3.156161 | train BPC = 4.553378 | avg val loss = 4.092119 | avg val BPC = 5.903680
```

## 6. Qualitative sample evidence

### Best practical sample excerpt (step 10000)
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
正莼乡。
任浊酒渊明，青蓑笠过，白鸥归去舟。
```
### Later-stage sample excerpt (step 15000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。

秋色早天空，正钱塘草木，雁声催晓。
白鸟山光有烂漫，十年梦里人心未得。
谩悔懊。
算不如、如今老。
怕僮仆揶揄，风斤可惜。
便纵有、天也为人意，何为有、人情可爱。
算今岁、万里相逢，是好元宵好。

江头好，风雨洗、怎教是愁地。
向南国。
看使君、匆匆过了，此情未惬。
```