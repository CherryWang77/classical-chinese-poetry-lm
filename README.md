# Classical Chinese Poetry Generation with a Character-Level Transformer

This repository contains a computational linguistics course project on classical Chinese poetry generation. The project implements a decoder-only character-level transformer and studies whether structural and statistical properties of an edited Song-ci corpus reappear in generated continuations. Rather than treating generation as a purely creative task, the project focuses on controlled comparison, checkpoint selection, and qualitative analysis of generated poems.

Both Tang poetry and Song ci were explored during corpus selection. Tang regulated verse was useful in the exploratory phase because its visible formal regularity makes structure-oriented reasoning comparatively straightforward. The final real training line, however, was based on Song ci. The edited Song-ci materials provided clear line breaks, stanza-like segmentation, richer lyric variation, and a corpus size that was practical for repeated Colab-based experiments.

## Main findings

- **Strongest balanced model:** `6-layer / BS32 / Dim256 / Heads8`, with the safest practical checkpoint at **step 10000**.
- **Strongest peak model:** `6-layer / BS32 / Dim384 / Heads8`, with the strongest peak validation result at **step 5000**.
- **Core methodological conclusion:** the best checkpoint was often more important than the final checkpoint; training longer did not necessarily produce a better poetry model.

## Fixed generation prompt

A fixed short prompt was used across major checkpoints in order to keep qualitative comparison interpretable and controlled. The prompt was taken manually from the very beginning of `final_corpus/songci_main_5mb.txt`:

> 气和玉烛，睿化著鸿明。  
> 缇管一阳生。

This prompt was kept fixed for three reasons:

1. It is short, so the continuation quickly reflects the model’s own learned behaviour rather than being heavily constrained by a long context.
2. It already contains a clear two-line lyric opening with punctuation and line breaks, which encourages structurally legible continuations.
3. Using the same prompt across checkpoints makes cross-checkpoint comparison much more credible.

## Repository structure

```text
.
├── experiment_records/          # controlled-comparison summaries and experiment notes
├── final_corpus/                # final processed corpora used in the project
├── results/                     # metrics, plots, and representative generation outputs
├── scripts_corpus_prep/         # corpus inspection, export, filtering, and dataset-building scripts
├── scripts_training_final/      # final training / continuation / generation / plotting scripts
├── 宋词/                         # raw source materials retained from the original corpus repository
├── 全唐诗/                       # raw source materials retained from the original corpus repository
├── 御定全唐詩/                   # raw source materials retained from the original corpus repository
├── 水墨唐诗/                     # raw source materials retained from the original corpus repository
├── README.md
├── requirements.txt
└── LICENSE
