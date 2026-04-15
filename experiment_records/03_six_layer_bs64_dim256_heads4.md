# Experiment: 6-layer / BS64 / Dim256 / Heads4

## 1. Purpose
This experiment was designed to test whether increasing batch size from 32 to 64 would improve model behaviour under the 6-layer / hidden dimension 256 / heads 4 setup.

## 2. Configuration
- Corpus: songci_main_5mb.txt
- Context length: 256
- Batch size: 64
- Hidden dimension: 256
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
- Best observed checkpoint: around step 7000
- Best validation loss: approximately 4.052221
- Best validation BPC: approximately 5.846120
- Best practical saved checkpoint: step 7000
- Final checkpoint: step 20000
- Final validation loss: 4.416212
- Final validation BPC: 6.371247

## 5. Key log excerpts
```text
=== 6-layer / BS64 / Dim256 / Heads4 experiment training (20000, Colab) ===
Device: cuda
Context length: 256
Batch size: 64
Embed dim: 256
Num layers: 6
Num heads: 4
Dropout: 0.1
Learning rate: 0.0003
Num steps: 20000

Step 07000 | train loss = 3.573637 | train BPC = 5.155668 | avg val loss = 4.052221 | avg val BPC = 5.846120
Step 10000 | train loss = 3.277767 | train BPC = 4.728818 | avg val loss = 4.090427 | avg val BPC = 5.901238
Step 15000 | train loss = 3.007609 | train BPC = 4.339063 | avg val loss = 4.279082 | avg val BPC = 6.173410
Step 20000 | train loss = 2.788145 | train BPC = 4.022443 | avg val loss = 4.416212 | avg val BPC = 6.371247
```

## 6. Qualitative sample evidence

### Best-region sample excerpt (step 7000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。
岁岁春浓，花上第，年少年时。

平生富贵，几度长江水。
叹飘飘，漂泊何时，天涯无限，高歌一醉。
满目东篱，倚云愁冉，是谁能继。
江涵落影，冷澹闲亭柳。
孤竹依高，独有相呼伴侣，且同看、千年春梦。
笑此处、新来几度。
天涯倦客，无处可，几度江云伴。
空暗忆同云，翠屏深处。
```

### Later-stage sample excerpt (step 20000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。

龙虎精衷，臣来忠孝，鼎盛平如。
圣主一宗，中路入崇新。
玄圃中兴县，玉坛论。
明堂玉帝出扶调。
玉帝下天颜。
九重渴深，八诏歌迟。
华阳将驾，大将指点祖诸民。

朝元时候，佳气葱葱瑞。
今日五云来至昼。
今朝祝寿。
称寿处，天难老。
祥烟舞袖香凝袖。
```