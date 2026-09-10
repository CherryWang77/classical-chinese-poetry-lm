# Every Character Counts: Pre-registered Experiment Plan

Status: implementation and CPU validation. The held-out test split must not be used for
checkpoint selection, rhyme-weight selection, debugging, or prompt revision.

## Research questions

1. Can an online character-level finite-state constraint guarantee valid Song-ci structure
   when the generator uses a subword tokenizer?
2. Does online constraint enforcement produce more valid outputs per GPU-minute than
   prompt-only generation or Best-of-8 post-hoc selection?
3. Can a soft Cilin Zhengyun reward improve rhyme consistency without materially reducing
   diversity?
4. What constraint overhead is introduced by character and subword tokenization?

## Frozen protocol

- Data and rhyme resources are pinned in `configs/research/data.json`.
- Target forms are the ten variants listed in that config.
- Split seed: `2026`; fold 0 is test, fold 1 is validation, folds 2-9 are training.
- Model/training seeds: `42`, `1729`, `2026`.
- Decoding seeds: `42`, `1729`, `2026`.
- The rhyme-weight sweep is `{0, 0.5, 1.0, 2.0, 4.0}` and is selected on validation only.
- The test grid is ten forms by six themes. Test evaluation is run once after the selected
  checkpoints and rhyme weight have been written to a frozen selection manifest.

## Primary outcomes

- Structural exact match, segment accuracy, punctuation/stanza accuracy, and dead-end rate.
- Cilin Zhengyun rhyme consistency.
- Valid outputs per GPU-minute, latency, throughput, and peak CUDA allocation.
- Character-level diversity, repetition, and training-corpus overlap.

All differences are aggregated to the 60 independent form-theme prompts before paired
bootstrap confidence intervals are calculated. A positive result is not required; negative
or mixed results will be reported without changing the protocol.

## Human evaluation gate

Human evaluation is included only if two independent native-Chinese annotators can complete
the same 60 blinded pairs. Otherwise, the report will make no claim that literary quality was
human-validated and will label automatic judge scores as descriptive only.
