# Experiment: 6-layer / BS32 / Dim256 / Heads8

## 1. Purpose
This experiment was designed to test whether increasing the number of attention heads from 4 to 8 would improve model behaviour under the 6-layer / batch size 32 / hidden dimension 256 setup.

## 2. Configuration
- Corpus: songci_main_5mb.txt
- Context length: 256
- Batch size: 32
- Hidden dimension: 256
- Number of layers: 6
- Number of heads: 8
- Dropout: 0.1
- Learning rate: 3e-4
- Training steps: 20000
- Device: cuda

## 3. Main comparison target
This line is mainly compared against:

- 6-layer / BS32 / Dim256 / Heads4

## 4. Key quantitative results
- Best observed validation region: around step 10600
- Best validation loss: 4.021207
- Best validation BPC: 5.801375
- Best practical saved checkpoint: step 10000
- Validation loss at step 10000: 4.070869
- Validation BPC at step 10000: 5.873023
- Final checkpoint: step 20000
- Final validation loss: 4.191617
- Final validation BPC: 6.047225

## 5. Key log excerpts
```text
=== 6-layer / BS32 / Dim256 / Heads8 experiment training (20000, Colab) ===
Device: cuda
Context length: 256
Batch size: 32
Embed dim: 256
Num layers: 6
Num heads: 8
Dropout: 0.1
Learning rate: 0.0003
Num steps: 20000

Step 10000 | train loss = 3.567725 | train BPC = 5.147139 | avg val loss = 4.070869 | avg val BPC = 5.873023
Step 10600 | train loss = 3.514323 | train BPC = 5.070096 | avg val loss = 4.021207 | avg val BPC = 5.801375
Step 11000 | train loss = 3.517274 | train BPC = 5.074354 | avg val loss = 4.101842 | avg val BPC = 5.917707
Step 15000 | train loss = 3.187013 | train BPC = 4.597887 | avg val loss = 4.131286 | avg val BPC = 5.960185
Step 20000 | train loss = 3.048259 | train BPC = 4.397708 | avg val loss = 4.191617 | avg val BPC = 6.047225
```

## 6. Quantitative sample evidence
### Best practical sample excerpt(step 10000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。
看八千椿算来现，同姓显，大椿年。

十月三春春色到，见花面、先生四日。
金尊满引，一气如虹白。
金城里，画楼深处，依旧画梁间。
金针侍，簇香环、醉拥东屏，应想像。
歌醉舞，红袖小，金缕歌喉。
翠袖掩金钗。
更阑上、屏山月落。

秋色渐满，月移来、还是有仙人。
人意正好，天长地久，一气清辉。
几度鸾鉴水晶盘。
正满天台。
好风吹雨，几点云深。
夜深一叶明芜。
```

### Later-stage sample excerpt(step 20000)
```text
气和玉烛，睿化著鸿明。
缇管一阳生。
应羡南山寿域，蔚蓝袍、福产瑶觥。

寿筵开宴，对华筵。
香烬回风，翠被红摇。
金兽喷沈烟。
兽炉烟断，笙调簧飐，满院深深。
人意在，绮筵红浸宝杯。

双盘鸾凤，宛转双蝉语。
玉帐虚堂，锦瑟初弦脆。
瑶钿皱重，玉殿珠筵醉。
银烛交辉，金炉细、玉炉香炷。
满斟玉斝，缓歌金缕，千岁寿龄，共祝千千岁。
```