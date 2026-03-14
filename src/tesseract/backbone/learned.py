from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tesseract.backbone.interface import Backbone, BackboneOutput
from tesseract.compiler.synthetic import RESULT_REGISTER

from .datasets import NaturalLanguageTask


@dataclass(frozen=True)
class BackboneVocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def from_tasks(cls, tasks: list[NaturalLanguageTask]) -> BackboneVocabulary:
        tokens = sorted({token for task in tasks for token in task.prompt.lower().split()})
        itos = ["<pad>", "<unk>", *tokens]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, prompt: str) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in prompt.lower().split()]

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]


@dataclass(frozen=True)
class CanonicalPromptVocabulary:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_tasks(cls, tasks: list[NaturalLanguageTask]) -> CanonicalPromptVocabulary:
        prompts = sorted({task.canonical_prompt for task in tasks})
        return cls(stoi={prompt: index for index, prompt in enumerate(prompts)}, itos=prompts)

    @property
    def size(self) -> int:
        return len(self.itos)


@dataclass
class BackboneTrainingBatch:
    prompts: list[str]
    target_ids: list[int]


class LearnedBackboneModel(nn.Module):
    def __init__(
        self,
        *,
        vocabulary: BackboneVocabulary,
        canonical_vocabulary: CanonicalPromptVocabulary,
        hidden_dim: int = 128,
        conditioning_dim: int = 8,
        learning_rate: float = 0.05,
        max_prompt_length: int = 24,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.canonical_vocabulary = canonical_vocabulary
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
        self.learning_rate = learning_rate
        self.max_prompt_length = max_prompt_length
        input_dim = vocabulary.size * max_prompt_length
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.canonical_head = nn.Linear(hidden_dim, canonical_vocabulary.size)
        self.conditioning_head = nn.Linear(hidden_dim, conditioning_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_prompt(self, prompt: str, repair_hint: str | None = None) -> Tensor:
        full_prompt = prompt if repair_hint is None else f"{prompt} {repair_hint}"
        token_ids = self.vocabulary.encode(full_prompt)
        features = torch.zeros(self.max_prompt_length * self.vocabulary.size, dtype=torch.float32, device=self.device)
        for position, token_id in enumerate(token_ids[: self.max_prompt_length]):
            features[(position * self.vocabulary.size) + token_id] = 1.0
        return features

    def hidden_state(self, prompt: str, repair_hint: str | None = None) -> Tensor:
        return self.encoder(self.encode_prompt(prompt, repair_hint)).squeeze(0)

    def canonical_logits(self, prompt: str, repair_hint: str | None = None) -> Tensor:
        return self.canonical_head(self.hidden_state(prompt, repair_hint))

    def predict_canonical_prompt(self, prompt: str, repair_hint: str | None = None) -> str:
        with torch.no_grad():
            logits = self.canonical_logits(prompt, repair_hint)
            index = int(torch.argmax(logits).item())
        return self.canonical_vocabulary.itos[index]

    def conditioning_vector(self, prompt: str, repair_hint: str | None = None) -> tuple[float, ...]:
        with torch.no_grad():
            hidden = self.hidden_state(prompt, repair_hint)
            conditioning = torch.tanh(self.conditioning_head(hidden))
        return tuple(float(value) for value in conditioning.cpu().tolist())


@dataclass
class LearnedBackbone(Backbone):
    model: LearnedBackboneModel
    result_register: int = RESULT_REGISTER

    def encode(
        self,
        prompt: str,
        *,
        repair_hint: str | None = None,
    ) -> BackboneOutput:
        canonical_prompt = self.model.predict_canonical_prompt(prompt, repair_hint)
        conditioning = self.model.conditioning_vector(prompt, repair_hint)
        task_type, values, metadata = _parse_canonical_prompt(canonical_prompt)
        if repair_hint is not None:
            metadata = {**metadata, "repair_hint": repair_hint}
        return BackboneOutput(
            original_prompt=prompt,
            canonical_prompt=canonical_prompt,
            task_type=task_type,
            result_register=self.result_register,
            values=values,
            conditioning=conditioning,
            metadata=metadata,
        )


def build_backbone_training_batch(
    tasks: list[NaturalLanguageTask],
    *,
    canonical_vocabulary: CanonicalPromptVocabulary,
) -> BackboneTrainingBatch:
    return BackboneTrainingBatch(
        prompts=[task.prompt for task in tasks],
        target_ids=[canonical_vocabulary.stoi[task.canonical_prompt] for task in tasks],
    )


def train_backbone_step(
    model: LearnedBackboneModel,
    batch: BackboneTrainingBatch,
    *,
    epochs: int = 256,
) -> dict[str, float]:
    if not batch.prompts:
        return {"loss": 0.0, "accuracy": 0.0}

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        model.optimizer.zero_grad()
        logits = torch.stack([model.canonical_logits(prompt) for prompt in batch.prompts], dim=0)
        targets = torch.tensor(batch.target_ids, dtype=torch.long, device=model.device)
        loss = model.loss_fn(logits, targets)
        loss.backward()
        model.optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        if final_loss < 1e-4:
            break

    model.eval()
    correct = sum(
        1
        for prompt, target_id in zip(batch.prompts, batch.target_ids, strict=True)
        if model.predict_canonical_prompt(prompt) == model.canonical_vocabulary.itos[target_id]
    )
    return {
        "loss": final_loss,
        "accuracy": correct / len(batch.prompts),
    }


def build_learned_backbone(tasks: list[NaturalLanguageTask]) -> LearnedBackbone:
    vocabulary = BackboneVocabulary.from_tasks(tasks)
    canonical_vocabulary = CanonicalPromptVocabulary.from_tasks(tasks)
    model = LearnedBackboneModel(vocabulary=vocabulary, canonical_vocabulary=canonical_vocabulary)
    return LearnedBackbone(model=model)


def _parse_canonical_prompt(canonical_prompt: str) -> tuple[str, tuple[int, ...], dict[str, int | float | str | bool]]:
    parts = canonical_prompt.split()
    if not parts:
        raise ValueError("canonical prompt must not be empty")
    if parts[0] == "arith" and len(parts) == 4:
        operation = parts[1]
        lhs = int(parts[2])
        rhs = int(parts[3])
        return "arithmetic", (lhs, rhs), {"operation": operation}
    if parts[0] == "max" and len(parts) == 3:
        return "max", (int(parts[1]), int(parts[2])), {}
    if parts[0] == "sum_to_n" and len(parts) == 2:
        return "sum_to_n", (int(parts[1]),), {}
    raise ValueError(f"unsupported canonical prompt {canonical_prompt!r}")
