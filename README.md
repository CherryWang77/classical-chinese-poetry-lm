# Classical Chinese Poetry Generation with a Character-Level Transformer

This repository contains the code and evidence for a computational linguistics project on character-level Song-ci generation. It implements a decoder-only Transformer, controlled hyperparameter comparisons, checkpointed generation, and structural comparison between the edited training corpus and generated text.

The raw poetry records come from the [`chinese-poetry/chinese-poetry`](https://github.com/chinese-poetry/chinese-poetry) collection. This repository keeps only the processed 5 MiB Song-ci corpus used by the reported experiments, rather than duplicating the complete upstream archive.

## Main findings

| Experiment | Best step | Best validation loss | Final step | Final validation loss |
|---|---:|---:|---:|---:|
| Real Song-ci baseline (4L / BS32 / D256 / H4) | 10000 | 4.058797 | 15000 | 4.070005 |
| 6L / BS32 / D256 / H4 | 10000 | 4.035873 | 15000 | 4.092119 |
| 6L / BS64 / D256 / H4 | 7000 | 4.052221 | 20000 | 4.416212 |
| 6L / BS32 / D256 / H8 | 10600 | 4.021207 | 20000 | 4.191617 |
| 6L / BS32 / D384 / H4 | 5000 | 4.004561 | 20000 | 5.056804 |
| 6L / BS32 / D384 / H8 | 5000 | 3.974154 | 15000 | 4.671842 |

The strongest peak model was `6L / BS32 / D384 / H8` at step 5000. The strongest balanced model was `6L / BS32 / D256 / H8`: it combined a strong validation region with substantially less late-stage deterioration than either D384 line.

## Design choices

- **Character-level modelling:** avoids imposing a modern word segmentation scheme on Classical Chinese and directly preserves punctuation and line boundaries.
- **Learned positional embeddings:** match the compact fixed-context training setting and add only a small number of parameters at context length 256.
- **Context length 256:** usually spans several edited Song-ci lines, allowing the model to observe line transitions and local stanza-like structure while remaining practical on a Colab GPU.
- **Two attention backends:** `manual` reproduces the triangular-mask implementation used for the historical results; `sdpa` uses `torch.nn.functional.scaled_dot_product_attention` for the required speed and memory comparison.
- **Fixed validation windows:** the refactored trainer evaluates every configuration on identical precomputed validation positions. Training, validation, and sample generation use separate random states.

The CSV files under `results/metrics/` are preserved historical experiment artifacts produced before the validation sampler was refactored. They support the submitted report, while new runs use the stricter fixed-window protocol described above.

## Repository structure

```text
.
├── configs/               # one JSON configuration per reported experiment
├── data/                  # processed Song-ci training corpus
├── results/
│   ├── figures/           # final comparison figures and numeric summary
│   ├── metrics/           # one merged CSV per experiment line
│   └── samples/           # representative checkpoint generations
├── scripts/
│   ├── train.py
│   ├── generate.py
│   ├── prepare_songci.py
│   ├── analyse_corpus.py
│   ├── analyse_generation.py
│   ├── benchmark_attention.py
│   └── plot_results.py
├── src/poetry_lm/         # reusable model, data, generation, and analysis code
├── tests/                 # fast unit tests
├── EXPERIMENTS.md
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the tests, install `requirements-dev.txt` instead.

## Corpus preparation

The exact included corpus is `data/songci_main_5mb.txt`:

```text
SHA-256  758890f5ba594ee67fdc6706e745a8d6149f1dc88b7965ea9108af384af48a22
```

To rebuild it from a local checkout of the upstream data:

```bash
python scripts/prepare_songci.py \
  --input-dir /path/to/chinese-poetry/宋词 \
  --output data/songci_main_5mb.txt \
  --target-mib 5
```

Inspect its structural statistics with:

```bash
python scripts/analyse_corpus.py data/songci_main_5mb.txt \
  --output results/corpus_statistics.json
```

## Training and continuation

Start a reported configuration:

```bash
python scripts/train.py --config configs/6l_bs32_d256_h8.json
```

Training outputs are written to the ignored `artifacts/<run_name>/` directory. Resume from a saved checkpoint without changing the random training stream:

```bash
python scripts/train.py \
  --config configs/6l_bs32_d256_h8.json \
  --resume artifacts/6l_bs32_d256_h8/checkpoints/step_10000.pt \
  --max-steps 20000
```

## Generation

```bash
python scripts/generate.py \
  --checkpoint artifacts/6l_bs32_d256_h8/checkpoints/step_10000.pt \
  --prompt $'气和玉烛，睿化著鸿明。\n缇管一阳生。\n' \
  --temperature 1.0 \
  --top-k 50 \
  --output generated.txt
```

## Attention benchmark

Run this command on the same GPU for both backends. It writes comparable forward/backward time and peak CUDA memory measurements:

```bash
python scripts/benchmark_attention.py \
  --config configs/6l_bs32_d256_h8.json \
  --batch-size 32 \
  --warmup 10 \
  --steps 50 \
  --device cuda \
  --output results/attention_benchmark.csv
```

## Rebuild figures and analyse generated structure

```bash
python scripts/plot_results.py

python scripts/analyse_generation.py \
  --reference data/songci_main_5mb.txt \
  --generated results/samples/six_layer_bs32_dim256_heads8_step_10000.txt \
  --generated results/samples/six_layer_bs32_dim384_heads8_step_5000.txt \
  --output results/generation_structure_comparison.json
```

## Tests

```bash
python -m pytest -q
```

See `EXPERIMENTS.md` for experiment provenance, checkpoint selection, and the corrected artifact-level comparison.
