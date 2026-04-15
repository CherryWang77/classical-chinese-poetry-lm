# Experiment: 6-layer / Dim384 / Heads4

## 1. Purpose
This experiment was designed to test whether increasing hidden dimension from 256 to 384 would improve model behaviour while keeping the stronger 6-layer architecture and the heads=4 setting unchanged.

## 2. Configuration
- Corpus: songci_main_5mb.txt
- Context length: 256
- Batch size: 32
- Hidden dimension: 384
- Number of layers: 6
- Number of heads: 4
- Dropout: 0.1
- Learning rate: 3e-4
- Training steps: 20000
- Device: cuda

## 3. Main comparison target
This line is mainly compared against:

- 6-layer / BS32 / Dim256 / Heads4

## 4. Key quantitative results
- Best observed validation region: around step 5800
- Best validation loss: approximately 4.026370
- Best validation BPC: approximately 5.808824
- Best practical saved checkpoint: step 6000
- Validation loss at step 6000: 4.027541
- Validation BPC at step 6000: 5.810513
- Final checkpoint: step 20000
- Final validation loss: 5.056804
- Final validation BPC: 7.295425

## 5. Key log excerpts
```text
=== Continue 6-layer + 384-dim training to 20000 (Colab) ===
Device: cuda
Context length: 256
Batch size: 32
Embed dim: 384
Num layers: 6
Num heads: 4
Dropout: 0.1
Learning rate: 0.0003

Step 05800 | train loss = 3.321490 | train BPC = 4.791898 | avg val loss = 4.026370 | avg val BPC = 5.808824
Step 06000 | train loss = 3.328189 | train BPC = 4.801562 | avg val loss = 4.027541 | avg val BPC = 5.810513
Step 10000 | train loss = 2.734832 | train BPC = 3.945529 | avg val loss = 4.248361 | avg val BPC = 6.129090
Step 15000 | train loss = 2.448226 | train BPC = 3.532044 | avg val loss = 4.664452 | avg val BPC = 6.729381
Step 20000 | train loss = 2.096793 | train BPC = 3.025033 | avg val loss = 5.056804 | avg val BPC = 7.295425
```

## 6. Qualitative sample evidence

### Best practical sample excerpt (step 6000 region)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。
人人天上清。

翠盖笼山绕。
绿鬓亸吟笑。
当年野老身难老。
长亭短帽。
今断九秋霜早。
明月清风台外到。
风庭梧桐凉月悄。
```

### Later-stage sample excerpt (step 20000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。

帝乡春晚，长昼帘垂。
天风不动帘栊。
酒醒初断，春眠尚迟。
重门深闭斜晖。
又是主人贤。
香冷人闲，日暮携壶。
愁重潘郎，独倚危樯。
秋色入窗纱。
```