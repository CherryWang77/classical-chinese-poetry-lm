from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .benchmark_generation import run_benchmark
from .lora_training import train_lora
from .research_data import build_dataset, fetch_sources, load_config
from .research_evaluation import evaluate_generations, select_validation_lambda
from .research_judge import run_judge
from .research_training import train_character_research


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poetry-lm",
        description="Every Character Counts reproducible research CLI",
    )
    parser.add_argument("--project-root", type=_path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="fetch and build the locked research dataset")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_build = data_commands.add_parser("build")
    data_build.add_argument("--config", type=_path, required=True)
    data_build.add_argument("--fetch", action="store_true")

    train = commands.add_parser("train", help="train a registered model")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    char = train_commands.add_parser("char")
    char.add_argument("--config", type=_path, required=True)
    char.add_argument("--seed", type=int, required=True)
    char.add_argument("--device", default="auto")
    char.add_argument("--resume", type=_path)
    char.add_argument("--pretrain-tokens", type=int)
    char.add_argument("--finetune-tokens", type=int)
    lora = train_commands.add_parser("lora")
    lora.add_argument("--config", type=_path, required=True)
    lora.add_argument("--seed", type=int, required=True)
    lora.add_argument("--resume")

    generate = commands.add_parser("generate", help="run registered generation systems")
    generate_commands = generate.add_subparsers(dest="generate_command", required=True)
    benchmark = generate_commands.add_parser("benchmark")
    benchmark.add_argument("--config", type=_path, required=True)
    benchmark.add_argument("--system", required=True)
    benchmark.add_argument("--training-seed", type=int)
    benchmark.add_argument("--split", choices=("validation", "test"), default="test")
    benchmark.add_argument("--rhyme-lambda", type=float)
    benchmark.add_argument("--output", type=_path)
    benchmark.add_argument("--limit", type=int)

    evaluate = commands.add_parser("evaluate", help="evaluate or freeze rhyme lambda")
    evaluate.add_argument("--manifest", type=_path, required=True)
    evaluate.add_argument(
        "--config",
        type=_path,
        default=_path("configs/research/benchmark.json"),
    )
    evaluate.add_argument(
        "--output-dir", type=_path, default=_path("results/research/evaluation")
    )
    evaluate.add_argument("--select-rhyme", action="store_true")
    evaluate.add_argument("--selection-output", type=_path)

    report = commands.add_parser("report", help="build tables, figures, and the PDF draft")
    report.add_argument(
        "--results-dir", type=_path, default=_path("results/research/evaluation")
    )
    report.add_argument("--output", type=_path)

    judge = commands.add_parser("judge", help="run the frozen descriptive Qwen3-4B judge")
    judge.add_argument("--manifest", type=_path, required=True)
    judge.add_argument("--config", type=_path, required=True)
    judge.add_argument("--output", type=_path, required=True)
    judge.add_argument("--limit", type=int)

    human = commands.add_parser("human-pack", help="build a 60-pair blinded evaluation pack")
    human.add_argument("--manifest", type=_path, required=True)
    human.add_argument("--form", type=_path, required=True)
    human.add_argument("--key", type=_path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "data":
        config = load_config(args.config)
        if args.fetch:
            fetch_sources(config, root)
        manifest = build_dataset(args.config, root)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train" and args.train_command == "char":
        output = train_character_research(
            args.config,
            root,
            args.seed,
            device_name=args.device,
            resume=args.resume,
            pretrain_tokens_override=args.pretrain_tokens,
            finetune_tokens_override=args.finetune_tokens,
        )
        print(output)
        return 0
    if args.command == "train" and args.train_command == "lora":
        print(train_lora(args.config, root, args.seed, resume=args.resume))
        return 0
    if args.command == "generate":
        output = run_benchmark(
            args.config,
            root,
            args.system,
            args.training_seed,
            dataset_split=args.split,
            rhyme_lambda=args.rhyme_lambda,
            output_override=args.output,
            limit=args.limit,
        )
        print(output)
        return 0
    if args.command == "evaluate":
        if args.select_rhyme:
            output = args.selection_output or root / "results/research/selected_rhyme_lambda.json"
            print(select_validation_lambda(args.manifest, output))
        else:
            print(
                evaluate_generations(
                    args.manifest,
                    args.config,
                    root,
                    args.output_dir,
                )
            )
        return 0
    if args.command == "report":
        from .reporting import build_research_report

        output = args.output or root / "output/pdf/every_character_counts_report.pdf"
        print(build_research_report(root, args.results_dir, output))
        return 0
    if args.command == "judge":
        print(run_judge(args.manifest, args.config, args.output, args.limit))
        return 0
    if args.command == "human-pack":
        from .human_evaluation import build_blind_evaluation_pack

        form, key = build_blind_evaluation_pack(args.manifest, args.form, args.key)
        print(form)
        print(key)
        return 0
    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
