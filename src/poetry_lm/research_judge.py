from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from .manifests import append_jsonl
from .research_evaluation import read_generations


def _generation_id(record: Any) -> str:
    payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_scores(text: str, fields: set[str]) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        scores = {field: int(payload[field]) for field in fields}
    except (KeyError, TypeError, ValueError):
        return None
    if any(value < 1 or value > 5 for value in scores.values()):
        return None
    return {**scores, "rationale": str(payload.get("rationale", ""))}


def run_judge(
    generation_path: Path,
    config_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-4B judging requires CUDA; run it from Kaggle or Colab")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("install the research dependencies before judging") from error
    config = json.loads(config_path.read_text(encoding="utf-8"))
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(config["load_in_4bit"]),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
    )
    model.eval()
    completed: set[str] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            completed = {
                str(json.loads(line)["generation_id"])
                for line in handle
                if line.strip()
            }
    records = read_generations(generation_path)
    if limit is not None:
        records = records[:limit]
    fields = set(config["rubric"])
    rubric = "\n".join(
        f"- {name}: {description}" for name, description in config["rubric"].items()
    )
    for record in records:
        generation_id = _generation_id(record)
        if generation_id in completed:
            continue
        instruction = (
            "你是古典文学评审。只根据给出的主题与词作，从1到5整数评分。"
            "不要判断词牌格式，因为格式由独立程序评测。\n"
            f"主题：{record.theme}\n词作：\n{record.text}\n\n评分项：\n{rubric}\n"
            "只输出一个JSON对象，键为上述四个英文评分项和rationale。"
        )
        parsed: dict[str, Any] | None = None
        raw = ""
        for attempt in range(2):
            messages = [
                {"role": "system", "content": "你是严格、一致的宋词质量评审。"},
                {
                    "role": "user",
                    "content": instruction
                    + ("\n上一次格式无效；本次不得输出Markdown。" if attempt else ""),
                },
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(config["max_new_tokens"]),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(
                output[0, inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            parsed = _parse_scores(raw, fields)
            if parsed is not None:
                break
        append_jsonl(
            output_path,
            {
                "generation_id": generation_id,
                "system": record.system,
                "template_id": record.template_id,
                "theme": record.theme,
                "training_seed": record.training_seed,
                "decoding_seed": record.decoding_seed,
                "judge_model_id": config["model_id"],
                "judge_model_revision": config["model_revision"],
                "valid_json": parsed is not None,
                "scores": parsed,
                "raw_response": raw,
                "evidence_role": "descriptive_only",
            },
        )
        completed.add(generation_id)
    return output_path
