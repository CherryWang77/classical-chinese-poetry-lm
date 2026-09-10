from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .constraints import ConstraintDeadEndError, RhymeLexicon, SongCiAutomaton
from .manifests import append_jsonl
from .model import DecoderOnlyLM
from .prompts import SYSTEM_PROMPT, character_control_prefix, user_prompt
from .research_metrics import rhyme_consistency, structural_metrics
from .research_types import GenerationRecord, TemplateSpec
from .templates import load_templates
from .token_constraints import ConstrainedLogitsProcessor, TokenSurfaceTable

CHAR_SYSTEMS = {"char_unconstrained", "char_hard_structure"}
QWEN_SYSTEMS = {
    "qwen_zero_shot",
    "qwen_lora_unconstrained",
    "qwen_lora_best_of_8",
    "qwen_lora_hard_structure",
    "qwen_lora_hard_structure_rhyme",
}
CONSTRAINED_SYSTEMS = {
    "char_hard_structure",
    "qwen_lora_hard_structure",
    "qwen_lora_hard_structure_rhyme",
}


@dataclass(frozen=True)
class SampleResult:
    text: str
    elapsed_seconds: float
    generated_tokens: int
    peak_memory_mb: float | None


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_key(record: GenerationRecord) -> tuple[Any, ...]:
    return (
        record.system,
        record.dataset_split,
        record.template_id,
        record.theme,
        record.training_seed,
        record.decoding_seed,
        record.rhyme_lambda,
    )


def _completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.add(_record_key(GenerationRecord.from_dict(json.loads(line))))
    return keys


def _seed(base: int, candidate: int = 0) -> int:
    payload = f"{base}:{candidate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _apply_sampling_filters(
    logits: torch.Tensor,
    generated: Iterable[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    scores = logits.clone()
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    for token_id in set(generated):
        value = scores[token_id]
        scores[token_id] = value * repetition_penalty if value < 0 else value / repetition_penalty
    scores /= temperature
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    scores[sorted_indices[remove]] = float("-inf")
    return scores


class CharacterGenerator:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.vocab = list(checkpoint["vocab"])
        self.stoi = {character: index for index, character in enumerate(self.vocab)}
        self.model = DecoderOnlyLM(
            len(self.vocab), ModelConfig(**checkpoint["model_config"])
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device
        self.checkpoint_path = checkpoint_path

    def _prompt_ids(self, prompt: str) -> list[int]:
        missing = sorted(set(prompt).difference(self.stoi))
        if missing:
            raise ValueError(
                "character checkpoint lacks prompt characters: " + "".join(missing)
            )
        return [self.stoi[character] for character in prompt]

    def sample(
        self,
        template: TemplateSpec,
        theme: str,
        decoding_seed: int,
        constrained: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> SampleResult:
        automaton = SongCiAutomaton(template)
        prompt = character_control_prefix(template, theme) + "\n"
        prompt_ids = self._prompt_ids(prompt)
        generated = list(prompt_ids)
        new_ids: list[int] = []
        state = automaton.initial_state
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(decoding_seed)
        maximum = len(automaton.slots)
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(maximum):
                conditioned = generated[-self.model.config.context_length :]
                inputs = torch.tensor([conditioned], dtype=torch.long, device=self.device)
                logits = self.model(inputs)[0, -1]
                if constrained:
                    expected = automaton.next_expected(state)
                    legal: list[tuple[int, Any]] = []
                    for token_id, character in enumerate(self.vocab):
                        transition = automaton.transition(state, character)
                        if transition is not None:
                            legal.append((token_id, transition))
                    if not legal:
                        raise ConstraintDeadEndError(
                            f"no legal character at offset {state.offset}; expected {expected!r}"
                        )
                    masked = torch.full_like(logits, float("-inf"))
                    ids = torch.tensor(
                        [token_id for token_id, _ in legal], device=self.device
                    )
                    masked[ids] = logits[ids]
                    logits = masked
                scores = _apply_sampling_filters(
                    logits,
                    new_ids,
                    temperature,
                    top_p,
                    repetition_penalty,
                )
                probabilities = torch.softmax(scores, dim=-1)
                token_id = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
                generated.append(token_id)
                new_ids.append(token_id)
                if constrained:
                    transition = automaton.transition(state, self.vocab[token_id])
                    if transition is None:
                        raise AssertionError("masked character produced an illegal transition")
                    state = transition.state
        elapsed = time.perf_counter() - start
        if constrained and not automaton.is_accepting(state):
            raise ConstraintDeadEndError("character generation did not reach an accepting state")
        return SampleResult(
            text="".join(self.vocab[token_id] for token_id in new_ids),
            elapsed_seconds=elapsed,
            generated_tokens=len(new_ids),
            peak_memory_mb=None,
        )


class QwenGenerator:
    def __init__(
        self,
        model_id: str,
        revision: str,
        adapter_path: Path | None,
        rhyme_lexicon: RhymeLexicon,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen benchmark requires CUDA; run it from Kaggle or Colab")
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("install the research dependencies before Qwen generation") from error
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
        )
        self.model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
        self.model.eval()
        self.model.config.use_cache = True
        self.model_id = model_id
        self.revision = revision
        self.adapter_path = adapter_path
        self.rhyme_lexicon = rhyme_lexicon
        self.surface_table = TokenSurfaceTable.from_tokenizer(self.tokenizer)

    def sample(
        self,
        template: TemplateSpec,
        theme: str,
        decoding_seed: int,
        constrained: bool,
        rhyme_lambda: float,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_unconstrained_new_tokens: int,
    ) -> SampleResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(template, theme)},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        prompt_length = int(inputs["input_ids"].shape[-1])
        automaton = SongCiAutomaton(template, self.rhyme_lexicon)
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if constrained:
            generation_kwargs["max_new_tokens"] = len(automaton.slots) + 1
            generation_kwargs["logits_processor"] = [
                ConstrainedLogitsProcessor(
                    tokenizer=self.tokenizer,
                    automaton=automaton,
                    prompt_length=prompt_length,
                    eos_token_id=int(self.tokenizer.eos_token_id),
                    rhyme_lambda=rhyme_lambda,
                    surface_table=self.surface_table,
                )
            ]
        else:
            generation_kwargs["max_new_tokens"] = max_unconstrained_new_tokens

        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(decoding_seed)
        torch.cuda.manual_seed_all(decoding_seed)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        elapsed = time.perf_counter() - start
        new_ids = output[0, prompt_length:].tolist()
        if new_ids and new_ids[-1] == self.tokenizer.eos_token_id:
            new_ids = new_ids[:-1]
        text = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if constrained:
            automaton.require_complete(text)
        return SampleResult(text, elapsed, len(new_ids), peak)


def _metrics(text: str, template: TemplateSpec, lexicon: RhymeLexicon) -> dict[str, float | bool]:
    return {
        **structural_metrics(text, template),
        "rhyme_consistency": rhyme_consistency(text, template, lexicon),
    }


def _best_of_n(
    generator: QwenGenerator,
    count: int,
    template: TemplateSpec,
    theme: str,
    decoding_seed: int,
    config: dict[str, Any],
    lexicon: RhymeLexicon,
) -> SampleResult:
    candidates: list[tuple[SampleResult, dict[str, float | bool]]] = []
    for candidate_index in range(count):
        sample = generator.sample(
            template=template,
            theme=theme,
            decoding_seed=_seed(decoding_seed, candidate_index),
            constrained=False,
            rhyme_lambda=0.0,
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            repetition_penalty=float(config["repetition_penalty"]),
            max_unconstrained_new_tokens=int(config["max_unconstrained_new_tokens"]),
        )
        candidates.append((sample, _metrics(sample.text, template, lexicon)))
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            bool(candidates[index][1]["structural_exact_match"]),
            float(candidates[index][1]["segment_length_accuracy"]),
            float(candidates[index][1]["rhyme_consistency"]),
            -index,
        ),
    )
    chosen = candidates[selected_index][0]
    return SampleResult(
        text=chosen.text,
        elapsed_seconds=sum(row[0].elapsed_seconds for row in candidates),
        generated_tokens=sum(row[0].generated_tokens for row in candidates),
        peak_memory_mb=max(
            (row[0].peak_memory_mb or 0.0 for row in candidates), default=0.0
        ),
    )


def run_benchmark(
    config_path: Path,
    project_root: Path,
    system: str,
    training_seed: int | None,
    dataset_split: str = "test",
    rhyme_lambda: float | None = None,
    output_override: Path | None = None,
    limit: int | None = None,
) -> Path:
    config = _load_config(config_path)
    if system not in config["systems"]:
        raise ValueError(f"system is not pre-registered: {system}")
    if dataset_split not in {"validation", "test"}:
        raise ValueError("benchmark split must be validation or test")
    if system == "qwen_zero_shot":
        training_seed = None
    elif training_seed not in config["training_seeds"]:
        raise ValueError("a pre-registered training seed is required")

    if system == "qwen_lora_hard_structure_rhyme" and rhyme_lambda is None:
        selected_path = _resolve(project_root, config["selected_rhyme_lambda_file"])
        if not selected_path.exists():
            raise FileNotFoundError(
                "freeze lambda on validation before running the rhyme-constrained test"
            )
        rhyme_lambda = float(json.loads(selected_path.read_text())["selected_lambda"])
    rhyme_lambda = float(rhyme_lambda or 0.0)
    if rhyme_lambda not in [float(value) for value in config["rhyme_lambda_candidates"]]:
        raise ValueError("rhyme lambda is outside the pre-registered sweep")

    templates_path = _resolve(project_root, config["templates_file"])
    rhyme_path = _resolve(project_root, config["rhyme_file"])
    templates = load_templates(templates_path)
    lexicon = RhymeLexicon.from_json(rhyme_path)
    output_path = output_override or _resolve(project_root, config["output_file"])
    completed = _completed_keys(output_path)

    model_id: str
    revision: str
    artifact: Path | None
    char_generator: CharacterGenerator | None = None
    qwen_generator: QwenGenerator | None = None
    if system in CHAR_SYSTEMS:
        artifact = (
            _resolve(project_root, config["char_artifact_dir"])
            / f"seed_{training_seed}"
            / "checkpoints"
            / "finetune_best.pt"
        )
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        char_generator = CharacterGenerator(
            artifact, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model_id = "poetry_lm.DecoderOnlyLM"
        revision = "local"
    elif system in QWEN_SYSTEMS:
        model_id = str(config["qwen_model_id"])
        revision = str(config["qwen_model_revision"])
        artifact = None
        if system != "qwen_zero_shot":
            artifact = (
                _resolve(project_root, config["lora_artifact_dir"])
                / f"seed_{training_seed}"
                / "best_adapter"
            )
            if not artifact.exists():
                raise FileNotFoundError(artifact)
        qwen_generator = QwenGenerator(model_id, revision, artifact, lexicon)
    else:
        raise AssertionError(system)

    jobs = [
        (template, theme, decoding_seed)
        for template in templates.values()
        for theme in config["themes"]
        for decoding_seed in config["decoding_seeds"]
    ]
    if limit is not None:
        jobs = jobs[:limit]
    for template, theme, decoding_seed in jobs:
        prompt = user_prompt(template, theme)
        placeholder = GenerationRecord(
            system=system,
            template_id=template.template_id,
            theme=theme,
            training_seed=training_seed,
            decoding_seed=int(decoding_seed),
            prompt=prompt,
            text="",
            elapsed_seconds=0.0,
            generated_tokens=0,
            peak_memory_mb=None,
            dead_end=False,
            metrics={},
            model_id=model_id,
            model_revision=revision,
            artifact=str(artifact) if artifact else None,
            dataset_split=dataset_split,
            rhyme_lambda=rhyme_lambda,
        )
        if _record_key(placeholder) in completed:
            continue
        try:
            if char_generator is not None:
                sample = char_generator.sample(
                    template,
                    theme,
                    _seed(int(decoding_seed)),
                    system in CONSTRAINED_SYSTEMS,
                    float(config["temperature"]),
                    float(config["top_p"]),
                    float(config["repetition_penalty"]),
                )
            elif system == "qwen_lora_best_of_8":
                assert qwen_generator is not None
                sample = _best_of_n(
                    qwen_generator,
                    int(config["best_of_n"]),
                    template,
                    theme,
                    int(decoding_seed),
                    config,
                    lexicon,
                )
            else:
                assert qwen_generator is not None
                sample = qwen_generator.sample(
                    template=template,
                    theme=theme,
                    decoding_seed=_seed(int(decoding_seed)),
                    constrained=system in CONSTRAINED_SYSTEMS,
                    rhyme_lambda=rhyme_lambda,
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                    repetition_penalty=float(config["repetition_penalty"]),
                    max_unconstrained_new_tokens=int(
                        config["max_unconstrained_new_tokens"]
                    ),
                )
            record = GenerationRecord(
                **{
                    **placeholder.to_dict(),
                    "text": sample.text,
                    "elapsed_seconds": sample.elapsed_seconds,
                    "generated_tokens": sample.generated_tokens,
                    "peak_memory_mb": sample.peak_memory_mb,
                    "metrics": _metrics(sample.text, template, lexicon),
                }
            )
        except ConstraintDeadEndError:
            record = GenerationRecord(**{**placeholder.to_dict(), "dead_end": True})
        append_jsonl(output_path, record.to_dict())
        completed.add(_record_key(record))
    return output_path
