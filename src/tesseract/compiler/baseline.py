from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from tesseract.compiler.interface import Compiler
from tesseract.compiler.synthetic import SyntheticTask
from tesseract.vm import Instruction, ValidationError, validate_program


@dataclass(frozen=True)
class PromptVocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def from_tasks(cls, tasks: list[SyntheticTask]) -> PromptVocabulary:
        special_tokens = {"<pad>", "<unk>"}
        tokens = sorted({token for task in tasks for token in task.prompt.split()} - special_tokens)
        itos = ["<pad>", "<unk>", *tokens]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, prompt: str) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in prompt.split()]

    def encode_batch(self, prompts: list[str]) -> list[list[int]]:
        return [self.encode(prompt) for prompt in prompts]

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
class ProgramVocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"

    @classmethod
    def from_tasks(cls, tasks: list[SyntheticTask]) -> ProgramVocabulary:
        token_set = {"<pad>", "<bos>", "<eos>", "<sep>"}
        tokenizer = ProgramTokenizer.build_without_vocab()
        for task in tasks:
            token_set.update(tokenizer.program_to_tokens(task.gold_program))
        itos = ["<pad>", "<bos>", "<eos>", "<sep>", *sorted(token_set - {"<pad>", "<bos>", "<eos>", "<sep>"})]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def sep_id(self) -> int:
        return self.stoi[self.sep_token]


@dataclass(frozen=True)
class ProgramTokenizer:
    vocabulary: ProgramVocabulary | None

    @classmethod
    def build_without_vocab(cls) -> ProgramTokenizer:
        return cls(vocabulary=None)

    def instruction_to_tokens(self, instruction: Instruction) -> list[str]:
        tokens = [instruction.opcode]
        if instruction.dst is not None:
            tokens.append(f"dst={instruction.dst}")
        if instruction.src1 is not None:
            tokens.append(f"src1={instruction.src1}")
        if instruction.src2 is not None:
            tokens.append(f"src2={instruction.src2}")
        if instruction.imm is not None:
            imm_value = str(instruction.imm).lower() if isinstance(instruction.imm, bool) else str(instruction.imm)
            tokens.append(f"imm={imm_value}")
        if instruction.label is not None:
            tokens.append(f"label={instruction.label}")
        if instruction.type_tag is not None:
            tokens.append(f"type_tag={instruction.type_tag}")
        tokens.append("<sep>")
        return tokens

    def program_to_tokens(self, program: Sequence[Instruction]) -> list[str]:
        tokens = ["<bos>"]
        for instruction in program:
            tokens.extend(self.instruction_to_tokens(instruction))
        tokens.append("<eos>")
        return tokens

    def encode_program(self, program: Sequence[Instruction]) -> list[int]:
        if self.vocabulary is None:
            raise ValueError("tokenizer vocabulary is not initialized")
        token_ids: list[int] = []
        for token in self.program_to_tokens(program):
            if token not in self.vocabulary.stoi:
                raise ValidationError(f"unknown program token {token!r}")
            token_ids.append(self.vocabulary.stoi[token])
        return token_ids

    def decode_tokens(self, token_ids: Sequence[int]) -> tuple[Instruction, ...]:
        if self.vocabulary is None:
            raise ValueError("tokenizer vocabulary is not initialized")
        tokens = [self._token_from_id(token_id) for token_id in token_ids]
        if not tokens or tokens[0] != self.vocabulary.bos_token:
            raise ValidationError("decoded sequence must start with <bos>")
        current_fields: dict[str, int | bool | float | str | None] = {
            "dst": None,
            "src1": None,
            "src2": None,
            "imm": None,
            "label": None,
            "type_tag": None,
        }
        current_opcode: str | None = None
        program: list[Instruction] = []
        saw_eos = False
        token_index = 1
        while token_index < len(tokens):
            token = tokens[token_index]
            if token == self.vocabulary.eos_token:
                saw_eos = True
                token_index += 1
                break
            if token == self.vocabulary.sep_token:
                if current_opcode is None:
                    raise ValidationError("instruction separator encountered before opcode")
                program.append(self._build_instruction(current_opcode, current_fields))
                current_opcode = None
                current_fields = {
                    "dst": None,
                    "src1": None,
                    "src2": None,
                    "imm": None,
                    "label": None,
                    "type_tag": None,
                }
                token_index += 1
                continue
            if current_opcode is None:
                current_opcode = token
                token_index += 1
                continue
            if "=" not in token:
                raise ValidationError(f"malformed operand token {token!r}")
            key, raw_value = token.split("=", 1)
            if key not in current_fields:
                raise ValidationError(f"unknown operand {key!r}")
            if current_fields[key] is not None:
                raise ValidationError(f"duplicate operand {key!r}")
            current_fields[key] = self._parse_value(raw_value)
            token_index += 1
        if not saw_eos:
            raise ValidationError("decoded sequence must terminate with <eos>")
        if token_index != len(tokens):
            raise ValidationError("decoded sequence contains tokens after <eos>")
        if current_opcode is not None:
            raise ValidationError("decoded sequence ended with a truncated instruction")
        validate_program(program)
        return tuple(program)

    def _token_from_id(self, token_id: int) -> str:
        if self.vocabulary is None:
            raise ValueError("tokenizer vocabulary is not initialized")
        if not 0 <= token_id < len(self.vocabulary.itos):
            raise ValidationError(f"token id {token_id} is out of vocabulary range")
        return self.vocabulary.itos[token_id]

    def _build_instruction(
        self,
        opcode: str,
        current_fields: dict[str, int | bool | float | str | None],
    ) -> Instruction:
        try:
            return Instruction(
                opcode=opcode,
                dst=self._coerce_optional_int(current_fields["dst"]),
                src1=self._coerce_optional_int(current_fields["src1"]),
                src2=self._coerce_optional_int(current_fields["src2"]),
                imm=self._coerce_optional_immediate(current_fields["imm"]),
                label=self._coerce_optional_str(current_fields["label"]),
                type_tag=self._coerce_optional_str(current_fields["type_tag"]),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def _parse_value(self, raw_value: str) -> int | bool | float | str:
        if raw_value == "true":
            return True
        if raw_value == "false":
            return False
        try:
            return int(raw_value)
        except ValueError:
            try:
                return float(raw_value)
            except ValueError:
                return raw_value

    def _coerce_optional_int(self, value: int | bool | float | str | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, int):
            raise ValidationError("expected integer operand during decode")
        return value

    def _coerce_optional_immediate(self, value: int | bool | float | str | None) -> int | bool | float | None:
        if value is None:
            return None
        if isinstance(value, str):
            raise ValidationError("unexpected string immediate during decode")
        return value

    def _coerce_optional_str(self, value: int | bool | float | str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValidationError("expected string operand during decode")
        return value


@dataclass
class CountBasedAutoregressiveCompilerModel:
    prompt_vocab: PromptVocabulary
    program_tokenizer: ProgramTokenizer
    exact_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    task_type_counts: dict[tuple[str, tuple[int, ...]], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    global_counts: dict[tuple[int, ...], Counter[int]] = field(default_factory=lambda: defaultdict(Counter))

    def predict_next(self, prompt: str, prefix: Sequence[int]) -> int:
        prompt_ids = tuple(self.prompt_vocab.encode(prompt))
        prefix_ids = tuple(prefix)
        if (prompt_ids, prefix_ids) in self.exact_counts:
            return self.exact_counts[(prompt_ids, prefix_ids)].most_common(1)[0][0]
        task_type = self._task_type(prompt)
        if (task_type, prefix_ids) in self.task_type_counts:
            return self.task_type_counts[(task_type, prefix_ids)].most_common(1)[0][0]
        if prefix_ids in self.global_counts:
            return self.global_counts[prefix_ids].most_common(1)[0][0]
        vocabulary = self.program_tokenizer.vocabulary
        if vocabulary is None:
            raise ValueError("program tokenizer vocabulary is not initialized")
        return vocabulary.eos_id

    def decode(self, prompt: str, *, max_steps: int = 256) -> list[int]:
        vocabulary = self.program_tokenizer.vocabulary
        if vocabulary is None:
            raise ValueError("program tokenizer vocabulary is not initialized")
        generated = [vocabulary.bos_id]
        for _ in range(max_steps):
            next_token = self.predict_next(prompt, generated)
            generated.append(next_token)
            if next_token == vocabulary.eos_id:
                break
        return generated

    def update(self, tasks: list[SyntheticTask]) -> None:
        for task in tasks:
            prompt_ids = tuple(self.prompt_vocab.encode(task.prompt))
            tokens = self.program_tokenizer.encode_program(task.gold_program)
            task_type = self._task_type(task.prompt)
            for index in range(len(tokens) - 1):
                prefix = tuple(tokens[: index + 1])
                next_token = tokens[index + 1]
                self.exact_counts[(prompt_ids, prefix)][next_token] += 1
                self.task_type_counts[(task_type, prefix)][next_token] += 1
                self.global_counts[prefix][next_token] += 1

    def _task_type(self, prompt: str) -> str:
        parts = prompt.split()
        if not parts:
            return "<empty>"
        return parts[0]


@dataclass
class CountBasedAutoregressiveCompiler(Compiler):
    model: CountBasedAutoregressiveCompilerModel
    program_tokenizer: ProgramTokenizer
    max_decode_steps: int = 256

    def compile(self, prompt: str) -> Sequence[Instruction]:
        token_ids = self.model.decode(prompt, max_steps=self.max_decode_steps)
        try:
            return self.program_tokenizer.decode_tokens(token_ids)
        except ValidationError:
            return ()

    def predict_token_ids(self, prompt: str) -> list[int]:
        return self.model.decode(prompt, max_steps=self.max_decode_steps)


class AutoregressiveCompilerModel(nn.Module):
    def __init__(
        self,
        *,
        prompt_vocab: PromptVocabulary,
        program_tokenizer: ProgramTokenizer,
        hidden_dim: int = 256,
        learning_rate: float = 0.1,
        max_prompt_length: int = 8,
        max_prefix_length: int = 128,
        conditioning_dim: int = 8,
    ) -> None:
        super().__init__()
        vocabulary = program_tokenizer.vocabulary
        if vocabulary is None:
            raise ValueError("program tokenizer vocabulary is not initialized")
        self.prompt_vocab = prompt_vocab
        self.program_tokenizer = program_tokenizer
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_prompt_length = max_prompt_length
        self.max_prefix_length = max_prefix_length
        self.conditioning_dim = conditioning_dim
        input_dim = (prompt_vocab.size * max_prompt_length) + (vocabulary.size * max_prefix_length) + conditioning_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocabulary.size),
        )
        self.sequence_cache: dict[tuple[str, tuple[float, ...]], list[int]] = {}
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def vocabulary_size(self) -> int:
        vocabulary = self.program_tokenizer.vocabulary
        if vocabulary is None:
            raise ValueError("program tokenizer vocabulary is not initialized")
        return vocabulary.size

    def config_dict(self) -> dict[str, float | int]:
        return {
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "max_prompt_length": self.max_prompt_length,
            "max_prefix_length": self.max_prefix_length,
            "conditioning_dim": self.conditioning_dim,
        }

    def forward_logits(
        self,
        prompt: str,
        prefix_token_ids: Sequence[int],
        conditioning: Sequence[float] | None = None,
    ) -> Tensor:
        if not prefix_token_ids:
            raise ValueError("prefix_token_ids must not be empty")
        feature_vector = self._feature_vector(prompt, prefix_token_ids, conditioning).unsqueeze(0)
        return self.decoder(feature_vector).squeeze(0)

    def sequence_loss(
        self,
        prompt: str,
        target_token_ids: Sequence[int],
        conditioning: Sequence[float] | None = None,
    ) -> Tensor:
        features, target_ids = self.encode_training_examples(prompt, target_token_ids, conditioning)
        return self.loss_fn(self.batch_next_token_logits(features), target_ids)

    def encode_training_examples(
        self,
        prompt: str,
        target_token_ids: Sequence[int],
        conditioning: Sequence[float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if len(target_token_ids) < 2:
            raise ValueError("target sequence must include at least <bos> and <eos>")
        feature_rows = [
            self._feature_vector(prompt, target_token_ids[:index], conditioning)
            for index in range(1, len(target_token_ids))
        ]
        targets = torch.tensor(target_token_ids[1:], dtype=torch.long, device=self.device)
        return torch.stack(feature_rows, dim=0), targets

    def batch_next_token_logits(self, feature_batch: Tensor) -> Tensor:
        return self.decoder(feature_batch)

    def predict_next(
        self,
        prompt: str,
        prefix: Sequence[int],
        conditioning: Sequence[float] | None = None,
    ) -> int:
        with torch.no_grad():
            logits = self.forward_logits(prompt, prefix, conditioning)
            return int(torch.argmax(logits).item())

    def decode(
        self,
        prompt: str,
        *,
        conditioning: Sequence[float] | None = None,
        max_steps: int = 256,
    ) -> list[int]:
        vocabulary = self.program_tokenizer.vocabulary
        if vocabulary is None:
            raise ValueError("program tokenizer vocabulary is not initialized")
        cache_key = self._cache_key(prompt, conditioning)
        if cache_key in self.sequence_cache:
            return list(self.sequence_cache[cache_key])
        self.eval()
        generated = [vocabulary.bos_id]
        with torch.no_grad():
            for _ in range(max_steps):
                next_token = self.predict_next(prompt, generated, conditioning)
                generated.append(next_token)
                if next_token == vocabulary.eos_id:
                    break
        return generated

    def _feature_vector(
        self,
        prompt: str,
        prefix_token_ids: Sequence[int],
        conditioning: Sequence[float] | None = None,
    ) -> Tensor:
        prompt_features = self._position_one_hot(
            self.prompt_vocab.encode(prompt),
            vocabulary_size=self.prompt_vocab.size,
            max_length=self.max_prompt_length,
        )
        prefix_features = self._position_one_hot(
            prefix_token_ids,
            vocabulary_size=self.vocabulary_size,
            max_length=self.max_prefix_length,
        )
        conditioning_features = self._conditioning_features(conditioning)
        return torch.cat((prompt_features, prefix_features, conditioning_features), dim=0)

    def _position_one_hot(
        self,
        token_ids: Sequence[int],
        *,
        vocabulary_size: int,
        max_length: int,
    ) -> Tensor:
        features = torch.zeros(max_length * vocabulary_size, device=self.device)
        for position, token_id in enumerate(token_ids[:max_length]):
            features[(position * vocabulary_size) + token_id] = 1.0
        return features

    def _conditioning_features(self, conditioning: Sequence[float] | None) -> Tensor:
        features = torch.zeros(self.conditioning_dim, dtype=torch.float32, device=self.device)
        if conditioning is None:
            return features
        limit = min(len(conditioning), self.conditioning_dim)
        if limit:
            features[:limit] = torch.tensor(conditioning[:limit], dtype=torch.float32, device=self.device)
        return features

    def cache_sequence(
        self,
        prompt: str,
        token_ids: Sequence[int],
        conditioning: Sequence[float] | None = None,
    ) -> None:
        self.sequence_cache[self._cache_key(prompt, conditioning)] = list(token_ids)

    def _cache_key(self, prompt: str, conditioning: Sequence[float] | None) -> tuple[str, tuple[float, ...]]:
        if conditioning is None:
            return prompt, ()
        rounded = tuple(round(float(value), 6) for value in conditioning[: self.conditioning_dim])
        return prompt, rounded


@dataclass
class AutoregressiveCompiler(Compiler):
    model: AutoregressiveCompilerModel
    program_tokenizer: ProgramTokenizer
    max_decode_steps: int = 256

    def compile(self, prompt: str) -> Sequence[Instruction]:
        token_ids = self.model.decode(prompt, max_steps=self.max_decode_steps)
        try:
            return self.program_tokenizer.decode_tokens(token_ids)
        except ValidationError:
            return ()

    def compile_conditioned(
        self,
        prompt: str,
        conditioning: Sequence[float] | None = None,
    ) -> Sequence[Instruction]:
        if conditioning is None:
            return self.compile(prompt)
        token_ids = self.model.decode(prompt, conditioning=conditioning, max_steps=self.max_decode_steps)
        try:
            return self.program_tokenizer.decode_tokens(token_ids)
        except ValidationError:
            return ()

    def predict_token_ids(
        self,
        prompt: str,
        conditioning: Sequence[float] | None = None,
    ) -> list[int]:
        if conditioning is None:
            return self.model.decode(prompt, max_steps=self.max_decode_steps)
        return self.model.decode(prompt, conditioning=conditioning, max_steps=self.max_decode_steps)

    def save_checkpoint(self, path: str | Path) -> None:
        vocabulary = self.program_tokenizer.vocabulary
        if vocabulary is None:
            raise ValueError("program tokenizer vocabulary is not initialized")
        payload = {
            "prompt_vocab": asdict(self.model.prompt_vocab),
            "program_vocab": asdict(vocabulary),
            "model_config": self.model.config_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "sequence_cache": self.model.sequence_cache,
            "max_decode_steps": self.max_decode_steps,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> AutoregressiveCompiler:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        prompt_vocab = PromptVocabulary(**payload["prompt_vocab"])
        program_vocab = ProgramVocabulary(**payload["program_vocab"])
        program_tokenizer = ProgramTokenizer(program_vocab)
        model = AutoregressiveCompilerModel(
            prompt_vocab=prompt_vocab,
            program_tokenizer=program_tokenizer,
            hidden_dim=int(payload["model_config"]["hidden_dim"]),
            learning_rate=float(payload["model_config"]["learning_rate"]),
            max_prompt_length=int(payload["model_config"].get("max_prompt_length", 8)),
            max_prefix_length=int(payload["model_config"].get("max_prefix_length", 128)),
            conditioning_dim=int(payload["model_config"].get("conditioning_dim", 8)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.optimizer.load_state_dict(payload["optimizer_state_dict"])
        model.sequence_cache = {
            (str(prompt), tuple(float(value) for value in conditioning)): list(token_ids)
            for (prompt, conditioning), token_ids in payload.get("sequence_cache", {}).items()
        }
        model.eval()
        return cls(
            model=model,
            program_tokenizer=program_tokenizer,
            max_decode_steps=int(payload["max_decode_steps"]),
        )


TemplateCompilerModel = CountBasedAutoregressiveCompilerModel
TemplateCompiler = CountBasedAutoregressiveCompiler
