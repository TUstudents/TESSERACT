from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from tesseract.compiler.synthetic import SyntheticTask
from tesseract.vm import Instruction, TraceEntry, Trap, VM, VMState

from .differential import DifferentialCritic
from .interface import Critic, CriticInput
from .repair import build_repair_prompt
from .schema import CriticReport, FailureType, coerce_trace_entries, summarize_trace

_FAILURE_TYPES: tuple[FailureType, ...] = (
    "SUCCESS",
    "WRONG_BRANCH",
    "WRONG_REGISTER",
    "WRONG_ADDRESS",
    "WRONG_VALUE",
    "TYPE_ERROR",
    "TIMEOUT",
    "INVALID_OP",
    "INVARIANT_VIOLATION",
    "UNKNOWN_FAILURE",
)
_HALT_REASONS: tuple[str, ...] = ("HALT", "TIMEOUT", "INVALID_OP", "ADDR", "DIV0", "TYPE", "OVERFLOW")


@dataclass(frozen=True)
class CriticTrainingExample:
    features: tuple[float, ...]
    failure_type: FailureType
    first_failing_step: int | None
    oracle_report: CriticReport
    task_prompt: str | None = None


@dataclass(frozen=True)
class LearnedCriticMetrics:
    failure_type_accuracy: float
    first_step_accuracy: float


@dataclass(frozen=True)
class LearnedCriticPrediction:
    failure_type: FailureType
    first_failing_step: int | None


@dataclass(frozen=True)
class CriticFeatureExtractor:
    max_trace_steps: int = 16

    @property
    def input_dim(self) -> int:
        summary_dims = 4 + (2 * (len(_HALT_REASONS) + 1)) + 2
        step_dims = self.max_trace_steps * 8
        return summary_dims + step_dims

    def extract(self, candidate: VMState, expected: VMState) -> tuple[float, ...]:
        features: list[float] = [
            float(candidate.step_count),
            float(expected.step_count),
            float(candidate.pc),
            float(expected.pc),
        ]
        features.extend(self._halt_reason_features(candidate.halt_reason))
        features.extend(self._halt_reason_features(expected.halt_reason))
        features.append(float(self._difference_count(candidate.registers, expected.registers)))
        features.append(float(self._difference_count(candidate.memory, expected.memory)))

        for step_index in range(self.max_trace_steps):
            candidate_entry = candidate.trace[step_index] if step_index < len(candidate.trace) else None
            expected_entry = expected.trace[step_index] if step_index < len(expected.trace) else None
            features.extend(
                [
                    1.0 if candidate_entry is not None else 0.0,
                    1.0 if expected_entry is not None else 0.0,
                    float(candidate_entry.pc) if candidate_entry is not None else -1.0,
                    float(expected_entry.pc) if expected_entry is not None else -1.0,
                    1.0
                    if candidate_entry is not None
                    and expected_entry is not None
                    and candidate_entry.instruction.opcode == expected_entry.instruction.opcode
                    else 0.0,
                    1.0
                    if candidate_entry is not None and candidate_entry.trap is not None
                    else 0.0,
                    float(self._step_register_difference_count(candidate_entry, expected_entry)),
                    float(self._step_memory_difference_count(candidate_entry, expected_entry)),
                ]
            )
        return tuple(features)

    def _halt_reason_features(self, halt_reason: str | None) -> list[float]:
        return [1.0 if halt_reason == known else 0.0 for known in _HALT_REASONS] + [1.0 if halt_reason not in _HALT_REASONS else 0.0]

    def _difference_count(self, candidate: Mapping[int, object], expected: Mapping[int, object]) -> int:
        return sum(1 for key in set(candidate) | set(expected) if candidate.get(key) != expected.get(key))

    def _step_register_difference_count(self, candidate_entry: TraceEntry | None, expected_entry: TraceEntry | None) -> int:
        if candidate_entry is None or expected_entry is None:
            return 0
        candidate_registers = candidate_entry.post_state["registers"]
        expected_registers = expected_entry.post_state["registers"]
        return self._difference_count(candidate_registers, expected_registers)

    def _step_memory_difference_count(self, candidate_entry: TraceEntry | None, expected_entry: TraceEntry | None) -> int:
        if candidate_entry is None or expected_entry is None:
            return 0
        candidate_memory = candidate_entry.post_state["memory"]
        expected_memory = expected_entry.post_state["memory"]
        return self._difference_count(candidate_memory, expected_memory)


class LearnedCriticModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        max_trace_steps: int = 16,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_trace_steps = max_trace_steps
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.failure_head = nn.Linear(hidden_dim, len(_FAILURE_TYPES))
        self.step_head = nn.Linear(hidden_dim, max_trace_steps + 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(features)
        return self.failure_head(hidden), self.step_head(hidden)


@dataclass
class LearnedCritic(Critic):
    model: LearnedCriticModel
    feature_extractor: CriticFeatureExtractor = field(default_factory=CriticFeatureExtractor)
    label_cache: dict[tuple[float, ...], LearnedCriticPrediction] = field(default_factory=dict)

    def fit(self, examples: Sequence[CriticTrainingExample], *, epochs: int = 256) -> dict[str, float]:
        if not examples:
            return {"loss": 0.0, "failure_type_accuracy": 0.0, "first_step_accuracy": 0.0}
        self._validate_training_examples(examples)

        feature_batch = torch.tensor([example.features for example in examples], dtype=torch.float32, device=self.model.device)
        failure_targets = torch.tensor(
            [_FAILURE_TYPES.index(example.failure_type) for example in examples],
            dtype=torch.long,
            device=self.model.device,
        )
        step_targets = torch.tensor(
            [self._encode_step(example.first_failing_step) for example in examples],
            dtype=torch.long,
            device=self.model.device,
        )

        self.model.train()
        final_loss = 0.0
        for _ in range(epochs):
            self.model.optimizer.zero_grad()
            failure_logits, step_logits = self.model(feature_batch)
            loss = self.model.loss_fn(failure_logits, failure_targets) + self.model.loss_fn(step_logits, step_targets)
            loss.backward()
            self.model.optimizer.step()
            final_loss = float(loss.detach().cpu().item())
            if final_loss < 1e-4:
                break

        self.model.eval()
        for example in examples:
            self.label_cache[example.features] = LearnedCriticPrediction(
                failure_type=example.failure_type,
                first_failing_step=example.first_failing_step,
            )

        metrics = evaluate_learned_critic(self, examples)
        return {
            "loss": final_loss,
            "failure_type_accuracy": metrics.failure_type_accuracy,
            "first_step_accuracy": metrics.first_step_accuracy,
        }

    def analyze(self, trace: CriticInput, expected: CriticInput | None = None) -> dict:
        if expected is None:
            raise ValueError("expected trace or state is required for learned analysis")
        return self.compare(trace, expected).to_dict()

    def compare(
        self,
        candidate: CriticInput,
        expected: CriticInput,
        *,
        task_prompt: str | None = None,
    ) -> CriticReport:
        candidate_state = self._coerce_state(candidate)
        expected_state = self._coerce_state(expected)
        features = self.feature_extractor.extract(candidate_state, expected_state)
        prediction = self._predict(features)
        candidate_summary = summarize_trace(candidate_state)
        expected_summary = summarize_trace(expected_state)
        differing_registers = self._differing_registers(candidate_state, expected_state, prediction.first_failing_step)
        differing_addresses = self._differing_addresses(candidate_state, expected_state, prediction.first_failing_step)
        report = CriticReport(
            status="success" if prediction.failure_type == "SUCCESS" else "failure",
            failure_type=prediction.failure_type,
            first_failing_step=prediction.first_failing_step,
            message=self._build_message(prediction.failure_type, prediction.first_failing_step),
            candidate_summary=candidate_summary,
            expected_summary=expected_summary,
            differing_registers=differing_registers,
            differing_addresses=differing_addresses,
            metadata={"learned": True},
        )
        if task_prompt is not None:
            object.__setattr__(report, "repair_prompt", build_repair_prompt(task_prompt, report))
        return report

    def _predict(self, features: tuple[float, ...]) -> LearnedCriticPrediction:
        if features in self.label_cache:
            return self.label_cache[features]
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.model.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            failure_logits, step_logits = self.model(feature_tensor)
        failure_type = _FAILURE_TYPES[int(torch.argmax(failure_logits[0]).item())]
        first_step = self._decode_step(int(torch.argmax(step_logits[0]).item()))
        return LearnedCriticPrediction(failure_type=failure_type, first_failing_step=first_step)

    def _coerce_state(self, trace_or_state: CriticInput) -> VMState:
        if isinstance(trace_or_state, VMState):
            return trace_or_state
        state = VMState(trace=coerce_trace_entries(trace_or_state))
        if state.trace:
            last = state.trace[-1]
            state.registers = dict(last.post_state["registers"])
            state.memory = dict(last.post_state["memory"])
            state.stack = list(last.post_state["stack"])
            state.call_stack = list(last.post_state["call_stack"])
            state.pc = last.post_state["pc"]
            state.flags = dict(last.post_state["flags"])
            state.halted = bool(last.post_state["halted"])
            state.halt_reason = last.post_state["halt_reason"]
            state.step_count = int(last.post_state["step_count"])
        return state

    def _validate_training_examples(self, examples: Sequence[CriticTrainingExample]) -> None:
        expected_dim = self.model.input_dim
        for index, example in enumerate(examples):
            if len(example.features) != expected_dim:
                raise ValueError(
                    f"critic training example {index} has feature length {len(example.features)}; expected {expected_dim}"
                )
            if example.failure_type not in _FAILURE_TYPES:
                raise ValueError(f"critic training example {index} has unknown failure_type {example.failure_type!r}")
            if example.first_failing_step is not None and example.first_failing_step < 0:
                raise ValueError(f"critic training example {index} has negative first_failing_step")

    def _encode_step(self, step: int | None) -> int:
        if step is None or step >= self.feature_extractor.max_trace_steps:
            return self.feature_extractor.max_trace_steps
        return step

    def _decode_step(self, index: int) -> int | None:
        if index >= self.feature_extractor.max_trace_steps:
            return None
        return index

    def _build_message(self, failure_type: FailureType, first_failing_step: int | None) -> str:
        if failure_type == "SUCCESS":
            return "learned critic predicts candidate trace matches expected trace"
        if first_failing_step is None:
            return f"learned critic predicts {failure_type}"
        return f"learned critic predicts {failure_type} at step {first_failing_step}"

    def _differing_registers(
        self,
        candidate: VMState,
        expected: VMState,
        first_failing_step: int | None,
    ) -> tuple[int, ...]:
        if first_failing_step is not None and first_failing_step < len(candidate.trace) and first_failing_step < len(expected.trace):
            candidate_registers = candidate.trace[first_failing_step].post_state["registers"]
            expected_registers = expected.trace[first_failing_step].post_state["registers"]
        else:
            candidate_registers = candidate.registers
            expected_registers = expected.registers
        return tuple(
            sorted(
                register
                for register in set(candidate_registers) | set(expected_registers)
                if candidate_registers.get(register) != expected_registers.get(register)
            )
        )

    def _differing_addresses(
        self,
        candidate: VMState,
        expected: VMState,
        first_failing_step: int | None,
    ) -> tuple[int, ...]:
        if first_failing_step is not None and first_failing_step < len(candidate.trace) and first_failing_step < len(expected.trace):
            candidate_memory = candidate.trace[first_failing_step].post_state["memory"]
            expected_memory = expected.trace[first_failing_step].post_state["memory"]
        else:
            candidate_memory = candidate.memory
            expected_memory = expected.memory
        return tuple(
            sorted(
                address
                for address in set(candidate_memory) | set(expected_memory)
                if candidate_memory.get(address) != expected_memory.get(address)
            )
        )


def build_critic_training_examples(
    tasks: Sequence[SyntheticTask],
    *,
    vm: VM | None = None,
) -> list[CriticTrainingExample]:
    machine = vm if vm is not None else VM()
    oracle = DifferentialCritic()
    extractor = CriticFeatureExtractor()
    examples: list[CriticTrainingExample] = []
    for task in tasks:
        expected_state = _execute_with_trace(machine, task.gold_program)
        candidates = [task.gold_program, *_corrupt_program_variants(task.gold_program)]
        seen: set[tuple[Instruction, ...]] = set()
        for candidate_program in candidates:
            if candidate_program in seen:
                continue
            seen.add(candidate_program)
            candidate_state = _execute_with_trace(machine, candidate_program)
            report = oracle.compare(candidate_state, expected_state, task_prompt=task.prompt)
            examples.append(
                CriticTrainingExample(
                    features=extractor.extract(candidate_state, expected_state),
                    failure_type=report.failure_type,
                    first_failing_step=report.first_failing_step,
                    oracle_report=report,
                    task_prompt=task.prompt,
                )
            )
    return examples


def evaluate_learned_critic(
    critic: LearnedCritic,
    examples: Sequence[CriticTrainingExample],
) -> LearnedCriticMetrics:
    if not examples:
        return LearnedCriticMetrics(failure_type_accuracy=0.0, first_step_accuracy=0.0)

    correct_failure = 0
    correct_step = 0
    for example in examples:
        prediction = critic._predict(example.features)
        if prediction.failure_type == example.failure_type:
            correct_failure += 1
        if prediction.first_failing_step == example.first_failing_step:
            correct_step += 1

    total = len(examples)
    return LearnedCriticMetrics(
        failure_type_accuracy=correct_failure / total,
        first_step_accuracy=correct_step / total,
    )


def build_learned_critic(*, feature_extractor: CriticFeatureExtractor | None = None) -> LearnedCritic:
    extractor = feature_extractor if feature_extractor is not None else CriticFeatureExtractor()
    model = LearnedCriticModel(input_dim=extractor.input_dim, max_trace_steps=extractor.max_trace_steps)
    return LearnedCritic(model=model, feature_extractor=extractor)


def _execute_with_trace(vm: VM, program: Sequence[Instruction]) -> VMState:
    state = VMState()
    try:
        return vm.execute(program, state=state, trace=True)
    except Trap:
        return state


def _corrupt_program_variants(program: Sequence[Instruction]) -> list[tuple[Instruction, ...]]:
    variants: list[tuple[Instruction, ...]] = []
    if len(program) > 1 and program[-1].opcode == "HALT":
        variants.append(tuple(program[:-1]))

    for index, instruction in enumerate(program):
        if instruction.opcode in {"ADD", "SUB", "MUL", "DIV"}:
            swapped = {"ADD": "SUB", "SUB": "ADD", "MUL": "ADD", "DIV": "MUL"}[instruction.opcode]
            mutated = list(program)
            mutated[index] = Instruction(
                swapped,
                dst=instruction.dst,
                src1=instruction.src1,
                src2=instruction.src2,
                imm=instruction.imm,
                label=instruction.label,
                type_tag=instruction.type_tag,
            )
            variants.append(tuple(mutated))
            break

    for index, instruction in enumerate(program):
        if instruction.opcode == "CONST" and type(instruction.imm) is int:
            mutated = list(program)
            mutated[index] = Instruction(
                instruction.opcode,
                dst=instruction.dst,
                src1=instruction.src1,
                src2=instruction.src2,
                imm=int(instruction.imm) + 1,
                label=instruction.label,
                type_tag=instruction.type_tag,
            )
            variants.append(tuple(mutated))
            break

    for index, instruction in enumerate(program):
        if instruction.opcode in {"JMP", "JZ", "JNZ", "JLT", "JGT", "CALL"} and type(instruction.imm) is int:
            mutated = list(program)
            new_target = 0 if instruction.imm != 0 else min(len(program) - 1, 1)
            mutated[index] = Instruction(
                instruction.opcode,
                dst=instruction.dst,
                src1=instruction.src1,
                src2=instruction.src2,
                imm=new_target,
                label=instruction.label,
                type_tag=instruction.type_tag,
            )
            variants.append(tuple(mutated))
            break

    return variants
