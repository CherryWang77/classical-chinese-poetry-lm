# Experiment: 6-layer / BS32 / Dim384 / Heads8

## 1. Purpose
This experiment was designed to test whether combining two previously productive tuning directions—using 8 attention heads and increasing hidden dimension to 384—would produce a stronger model than the earlier strongest candidate lines.

## 2. Configuration
- Corpus: songci_main_5mb.txt
- Context length: 256
- Batch size: 32
- Hidden dimension: 384
- Number of layers: 6
- Number of heads: 8
- Dropout: 0.1
- Learning rate: 3e-4
- Training steps: 15000
- Device: cuda

## 3. Main comparison target
This line is mainly compared against:

- 6-layer / BS32 / Dim256 / Heads8
- 6-layer / Dim384 / Heads4

## 4. Key quantitative results
- Best observed checkpoint: step 5000
- Best validation loss: 3.974154
- Best validation BPC: 5.733492
- Final checkpoint: step 15000
- Final validation loss: 4.671842
- Final validation BPC: 6.740043

## 5. Key log excerpts
```text
=== 6-layer / BS32 / Dim384 / Heads8 experiment training (15000, Colab) ===
Device: cuda
Context length: 256
Batch size: 32
Embed dim: 384
Num layers: 6
Num heads: 8
Dropout: 0.1
Learning rate: 0.0003
Num steps: 15000

Step 05000 | train loss = 3.450335 | train BPC = 4.977781 | avg val loss = 3.974154 | avg val BPC = 5.733492
Step 06000 | train loss = 3.341772 | train BPC = 4.821158 | avg val loss = 4.006255 | avg val BPC = 5.779804
Step 07400 | train loss = 3.153047 | train BPC = 4.548885 | avg val loss = 4.021567 | avg val BPC = 5.801895
Step 10000 | train loss = 2.830903 | train BPC = 4.084130 | avg val loss = 4.231749 | avg val BPC = 6.105124
Step 15000 | train loss = 2.316365 | train BPC = 3.341808 | avg val loss = 4.671842 | avg val BPC = 6.740043
```

## 6. Quantitative sample evidence
### Best-region sample excerpt  (step 5000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。
何须重睹驭重城。

天地雷声杂铁。
天籁奏严禋晓露。
万载无疆语。
香满五年无暑。
春满捧金卮。
祝寿如春长恁困。
年年今日称觞处。
且看大江春。

西崑照海，望中云，太无际。
五福从来，真水西崑阙。
我自无虚无地，更不是、明河无雪。
好在广寒仙，何日驾、丹霄阙。
```

### Later-stage sample excerpt  (step 15000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。
万代无穷人杰，千载自由天下，留作黑云轻。
富贵已官皆宰，指日一封荣。
何惜饮中仙。

良宵好，明日杏花枝。
春酒十分斟玉斝，香泛碧桃溪。
潋滟九霞飞翠幕，春滟滟金卮。
斟美酒，寿延低。
寿阳更着窦郎归。
何似亲郎并愿、一点灵犀。
```