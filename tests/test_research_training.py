from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from poetry_lm.research_training import train_character_research
from poetry_lm.research_types import PoemRecord, SegmentSpec, TemplateSpec


def _write_jsonl(path: Path, records: list[PoemRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _record(identifier: str, split: str, text: str) -> PoemRecord:
    return PoemRecord(
        id=identifier,
        author=identifier,
        rhythmic="测试",
        paragraphs=(text,),
        text=text,
        normalized_text=text,
        template_id="测试:v0",
        template_exact=True,
        split=split,
        fold=2 if split == "train" else 1,
        source_file="fixture.json",
        source_index=0,
        sha256=identifier,
        group_id=identifier,
    )


def _fixture_project(root: Path) -> Path:
    dataset = root / "data/research/processed"
    template = TemplateSpec(
        template_id="测试:v0",
        cipai="测试",
        variant=0,
        source_commit="fixture",
        segments=(SegmentSpec(char_count=4, punctuation="。", stanza_end=True),),
    )
    dataset.mkdir(parents=True)
    (dataset / "templates.json").write_text(
        json.dumps({"source_commit": "fixture", "templates": [template.to_dict()]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (dataset / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    train = _record("train", "train", "山河日月。" * 8)
    validation = _record("validation", "validation", "春江花月。" * 8)
    _write_jsonl(dataset / "splits/train.jsonl", [train])
    _write_jsonl(dataset / "selected/train.jsonl", [train])
    _write_jsonl(dataset / "selected/validation.jsonl", [validation])
    config = {
        "model": {
            "context_length": 4,
            "embed_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "attention_backend": "sdpa",
            "positional_encoding": "learned",
        },
        "training_seeds": [42],
        "conditioning_themes": ["春日相思"],
        "pretrain_tokens": 16,
        "finetune_tokens": 16,
        "batch_size": 2,
        "learning_rate": 0.0003,
        "finetune_learning_rate": 0.0001,
        "warmup_steps": 1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "eval_every": 1,
        "save_every": 1,
        "dataset_dir": "data/research/processed",
        "output_dir": "artifacts/research/char",
    }
    path = root / "char.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_checkpoint_resume_reproduces_uninterrupted_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("poetry_lm.research_training.git_commit", lambda _: "fixture")
    config = _fixture_project(tmp_path)
    run_dir = train_character_research(config, tmp_path, 42, device_name="cpu")
    checkpoint_dir = run_dir / "checkpoints"
    uninterrupted = torch.load(
        checkpoint_dir / "finetune_step_2.pt", map_location="cpu", weights_only=False
    )
    partial_checkpoint = tmp_path / "pretrain_step_1.pt"
    partial_best = tmp_path / "pretrain_best.pt"
    shutil.copy2(checkpoint_dir / "pretrain_step_1.pt", partial_checkpoint)
    shutil.copy2(checkpoint_dir / "pretrain_best.pt", partial_best)
    first_metric = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]

    shutil.rmtree(run_dir)
    checkpoint_dir.mkdir(parents=True)
    shutil.copy2(partial_best, checkpoint_dir / "pretrain_best.pt")
    (run_dir / "metrics.jsonl").write_text(first_metric + "\n", encoding="utf-8")
    train_character_research(
        config,
        tmp_path,
        42,
        device_name="cpu",
        resume=partial_checkpoint,
    )
    resumed = torch.load(
        checkpoint_dir / "finetune_step_2.pt", map_location="cpu", weights_only=False
    )
    for name, tensor in uninterrupted["model_state_dict"].items():
        torch.testing.assert_close(tensor, resumed["model_state_dict"][name], rtol=0, atol=0)
    metric_steps = [
        (row["stage"], row["step"])
        for row in (
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        )
    ]
    assert metric_steps == [
        ("pretrain", 1),
        ("pretrain", 2),
        ("finetune", 1),
        ("finetune", 2),
    ]
