from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

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
        tokens = sorted({token for task in tasks for token in task.prompt.split()})
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
                raise ValueError(f"unknown program token {token!r}")
            token_ids.append(self.vocabulary.stoi[token])
        return token_ids

    def decode_tokens(self, token_ids: Sequence[int]) -> tuple[Instruction, ...]:
        if self.vocabulary is None:
            raise ValueError("tokenizer vocabulary is not initialized")
        tokens = [self._token_from_id(token_id) for token_id in token_ids]
        if not tokens or tokens[0] != self.vocabulary.bos_token:
            raise ValidationError("decoded sequence must start with <bos>")
        current_fields: dict[str, int | bool | str | None] = {
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
        current_fields: dict[str, int | bool | str | None],
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

    def _parse_value(self, raw_value: str) -> int | bool | str:
        if raw_value == "true":
            return True
        if raw_value == "false":
            return False
        try:
            return int(raw_value)
        except ValueError:
            return raw_value

    def _coerce_optional_int(self, value: int | bool | str | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError("expected integer operand during decode")
        return value

    def _coerce_optional_immediate(self, value: int | bool | str | None) -> int | bool | None:
        if value is None:
            return None
        if isinstance(value, str):
            raise ValidationError("unexpected string immediate during decode")
        return value

    def _coerce_optional_str(self, value: int | bool | str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValidationError("expected string operand during decode")
        return value


@dataclass
class AutoregressiveCompilerModel:
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

    def predict_token_ids(self, prompt: str) -> list[int]:
        return self.model.decode(prompt, max_steps=self.max_decode_steps)


TemplateCompilerModel = AutoregressiveCompilerModel
TemplateCompiler = AutoregressiveCompiler
ValueVocabulary = ProgramVocabulary
OperationVocabulary = ProgramVocabulary
