# Classical Chinese Poetry Generation with a Character-Level Transformer

This repository contains a computational linguistics course project on classical Chinese poetry generation. The project implements a decoder-only character-level transformer and studies whether structural and statistical properties of an edited Song-ci corpus reappear in generated continuations. Rather than treating generation as a purely creative task, the project focuses on controlled comparison, checkpoint selection, and qualitative analysis of generated poems.

Both Tang poetry and Song ci were explored during corpus selection. Tang regulated verse was useful in the exploratory phase because its visible formal regularity makes structure-oriented reasoning comparatively straightforward. The final real training line, however, was based on Song ci. The edited Song-ci materials provided clear line breaks, stanza-like segmentation, richer lyric variation, and a corpus size that was practical for repeated Colab-based experiments.

The corpus is selected and modified through https://github.com/chinese-poetry/chinese-poetry/tree/master

## Project goals

The project had three main goals:

1. to construct a usable character-level corpus for classical Chinese poetry generation;
2. to train and compare several decoder-only transformer configurations under a controlled experimental design;
3. to analyse whether some structural and stylistic regularities of the training corpus reappear in generated texts.

The project therefore combines corpus preparation, neural language-model training, hyperparameter exploration, and close qualitative analysis of generated outputs.

## Main findings

- **Strongest balanced model:** `6-layer / BS32 / Dim256 / Heads8`, with the safest practical checkpoint at **step 10000**.
- **Strongest peak model:** `6-layer / BS32 / Dim384 / Heads8`, with the strongest peak validation result at **step 5000**.
- **Core methodological conclusion:** the best checkpoint was often more important than the final checkpoint; training longer did not necessarily produce a better poetry model.

## Final corpus choice

The final main training corpus is:

- `final_corpus/songci_main_5mb.txt`

Tang poetry remained important during corpus exploration and structural reasoning, but Song ci became the main final corpus for three reasons:

- the edited materials provided clear punctuation and visible line breaks;
- the corpus showed richer lyric and stanza-like variation;
- the corpus size was practical for repeated Colab-based experiments.

This choice was linguistic as well as computational. Song ci offered enough visible structure to support form-sensitive analysis, while also retaining stylistic variation that made the generation task more challenging and more interesting than simple surface imitation.

## Fixed generation prompt

A fixed short prompt was used across major checkpoints in order to keep qualitative comparison interpretable and controlled. The prompt was taken manually from the very beginning of `final_corpus/songci_main_5mb.txt`:

> 气和玉烛，睿化著鸿明。  
> 缇管一阳生。

This prompt was kept fixed for three reasons:

1. it is short, so the continuation quickly reflects the model’s own learned behaviour rather than being heavily constrained by a long context;
2. it already contains a clearly segmented lyric opening, which encourages structurally legible continuation;
3. using the same prompt across checkpoints makes qualitative comparison across models and training stages more credible.

## Repository structure

```text
.
├── experiment_records/          # controlled-comparison summaries and experiment notes
├── final_corpus/                # final processed corpora used in the project
├── results/                     # metrics, plots, representative samples, and report evidence
├── scripts_corpus_prep/         # corpus inspection, export, filtering, and dataset-building scripts
├── scripts_training_final/      # final training / continuation / generation / plotting scripts
├── model_prototypes/            # earlier prototype and small-scale model experiments
├── archive_notes/               # archived planning / interpretation notes
├── archive_helpers/             # archived helper scripts
├── 宋词/                         # raw source materials retained from the original corpus repository
├── 全唐诗/                       # raw source materials retained from the original corpus repository
├── 御定全唐詩/                   # raw source materials retained from the original corpus repository
├── 水墨唐诗/                     # raw source materials retained from the original corpus repository
├── README.md
├── requirements.txt
└── LICENSE
