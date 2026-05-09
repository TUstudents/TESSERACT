from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from tesseract.backbone.datasets import NaturalLanguageTask
from tesseract.backbone.interface import BackboneOutput
from tesseract.compiler.nl import BackboneConditionedCompiler, NaturalLanguageCompileResult, RepairCapableCompiler
from tesseract.vm import Instruction, VM, program_to_dict

from .differential import DifferentialCritic
from .loop import RepairLoopController, RepairLoopMetrics, RepairLoopResult, evaluate_repair_loop
from .repair import RepairState, build_repair_state, repair_state_feature_dim

_REPAIR_RESERVED_TOKENS = {"<pad>", "<unk>"}


@dataclass(frozen=True)
class RepairTrainingExample:
    prompt: str
    repair_state: RepairState
    target_canonical_prompt: str
    gold_program: tuple[Instruction, ...]
    task_type: str
    corruption_name: str


@dataclass(frozen=True)
class LearnedRepairMetrics:
    canonical_accuracy: float


@dataclass(frozen=True)
class RepairBenchmarkCase:
    task: NaturalLanguageTask
    initial_program: tuple[Instruction, ...]
    corruption_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task": {
                "prompt": self.task.prompt,
                "canonical_prompt": self.task.canonical_prompt,
                "expected_output": self.task.expected_output,
                "result_register": self.task.result_register,
                "task_type": self.task.task_type,
                "values": list(self.task.values),
            },
            "initial_program": program_to_dict(self.initial_program),
            "corruption_name": self.corruption_name,
        }


@dataclass(frozen=True)
class RepairBenchmarkReport:
    metrics: RepairLoopMetrics
    results: tuple[RepairLoopResult, ...]
    baseline_success_rate: float
    repaired_success_rate: float
    average_improvement: float
    task_type_improvement: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": asdict(self.metrics),
            "baseline_success_rate": self.baseline_success_rate,
            "repaired_success_rate": self.repaired_success_rate,
            "average_improvement": self.average_improvement,
            "task_type_improvement": dict(self.task_type_improvement),
            "results": [
                {
                    "success": result.success,
                    "termination_reason": result.termination_reason,
                    "rounds_used": result.rounds_used,
                }
                for result in self.results
            ],
        }


@dataclass(frozen=True)
class RepairVocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def from_examples(cls, examples: Sequence[RepairTrainingExample]) -> RepairVocabulary:
        tokens = sorted({
            token
            for example in examples
            for token in example.prompt.lower().split()
            if token not in _REPAIR_RESERVED_TOKENS
        })
        itos = ["<pad>", "<unk>", *tokens]
        return cls(stoi={token: index for index, token in enumerate(itos)}, itos=itos)

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, prompt: str) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in prompt.lower().split()]


@dataclass(frozen=True)
class RepairTargetVocabulary:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_examples(cls, examples: Sequence[RepairTrainingExample]) -> RepairTargetVocabulary:
        prompts = sorted({example.target_canonical_prompt for example in examples})
        if not prompts:
            raise ValueError("repair target vocabulary requires at least one target canonical prompt")
        return cls(stoi={prompt: index for index, prompt in enumerate(prompts)}, itos=prompts)

    @property
    def size(self) -> int:
        return len(self.itos)


class LearnedRepairModel(nn.Module):
    def __init__(
        self,
        *,
        vocabulary: RepairVocabulary,
        target_vocabulary: RepairTargetVocabulary,
        hidden_dim: int = 128,
        conditioning_dim: int = 8,
        learning_rate: float = 0.05,
        max_prompt_length: int = 24,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.target_vocabulary = target_vocabulary
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
        self.learning_rate = learning_rate
        self.max_prompt_length = max_prompt_length
        input_dim = (vocabulary.size * max_prompt_length) + repair_state_feature_dim()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.target_head = nn.Linear(hidden_dim, target_vocabulary.size)
        self.conditioning_head = nn.Linear(hidden_dim, conditioning_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_prompt(self, prompt: str) -> Tensor:
        token_ids = self.vocabulary.encode(prompt)
        features = torch.zeros(self.max_prompt_length * self.vocabulary.size, dtype=torch.float32, device=self.device)
        for position, token_id in enumerate(token_ids[: self.max_prompt_length]):
            features[(position * self.vocabulary.size) + token_id] = 1.0
        return features

    def encode_features(self, prompt: str, repair_state: RepairState) -> Tensor:
        prompt_features = self.encode_prompt(prompt)
        state_features = torch.tensor(repair_state.feature_vector(), dtype=torch.float32, device=self.device)
        return torch.cat((prompt_features, state_features), dim=0)

    def hidden_state(self, prompt: str, repair_state: RepairState) -> Tensor:
        return self.encoder(self.encode_features(prompt, repair_state)).squeeze(0)

    def logits(self, prompt: str, repair_state: RepairState) -> Tensor:
        return self.target_head(self.hidden_state(prompt, repair_state))

    def predict_canonical_prompt(self, prompt: str, repair_state: RepairState) -> str:
        with torch.no_grad():
            logits = self.logits(prompt, repair_state)
            index = int(torch.argmax(logits).item())
        return self.target_vocabulary.itos[index]

    def conditioning_vector(self, prompt: str, repair_state: RepairState) -> tuple[float, ...]:
        with torch.no_grad():
            hidden = self.hidden_state(prompt, repair_state)
            conditioning = torch.tanh(self.conditioning_head(hidden))
        return tuple(float(value) for value in conditioning.cpu().tolist())


@dataclass
class ModelDrivenRepairCompiler(RepairCapableCompiler):
    delegate: BackboneConditionedCompiler
    repair_model: LearnedRepairModel

    def fit(self, examples: Sequence[RepairTrainingExample], *, epochs: int = 256) -> dict[str, float]:
        if not examples:
            return {"loss": 0.0, "canonical_accuracy": 0.0}

        self.repair_model.train()
        features = torch.stack(
            [self.repair_model.encode_features(example.prompt, example.repair_state) for example in examples],
            dim=0,
        )
        targets = torch.tensor(
            [self.repair_model.target_vocabulary.stoi[example.target_canonical_prompt] for example in examples],
            dtype=torch.long,
            device=self.repair_model.device,
        )
        final_loss = 0.0
        for _ in range(epochs):
            self.repair_model.optimizer.zero_grad()
            logits = self.repair_model.target_head(self.repair_model.encoder(features))
            loss = self.repair_model.loss_fn(logits, targets)
            loss.backward()
            self.repair_model.optimizer.step()
            final_loss = float(loss.detach().cpu().item())
            if final_loss < 1e-4:
                break

        self.repair_model.eval()
        for example in examples:
            conditioning = self.repair_model.conditioning_vector(example.prompt, example.repair_state)
            gold_tokens = self.delegate.compiler.program_tokenizer.encode_program(example.gold_program)
            self.delegate.compiler.model.cache_sequence(example.target_canonical_prompt, gold_tokens, conditioning)

        metrics = evaluate_model_driven_repair(self, examples)
        return {"loss": final_loss, "canonical_accuracy": metrics.canonical_accuracy}

    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: Any | None = None,
    ) -> NaturalLanguageCompileResult:
        if repair_context is None:
            return self.delegate.compile_with_backbone_output(prompt)

        repair_state = build_repair_state(repair_context)
        repair_hint = repair_state.to_text()
        predicted_canonical_prompt = self.repair_model.predict_canonical_prompt(prompt, repair_state)
        conditioning = self.repair_model.conditioning_vector(prompt, repair_state)
        backbone_output = self.delegate.backbone.encode(prompt, repair_hint=repair_hint)
        program = tuple(self.delegate.compiler.compile_conditioned(predicted_canonical_prompt, conditioning))
        if not program:
            program = tuple(self.delegate.compiler.compile(predicted_canonical_prompt))
        return NaturalLanguageCompileResult(
            backbone_output=BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=predicted_canonical_prompt,
                task_type=backbone_output.task_type,
                result_register=backbone_output.result_register,
                values=backbone_output.values,
                conditioning=conditioning,
                metadata={
                    **backbone_output.metadata,
                    "repair_model_driven": True,
                    "repair_state": repair_state.to_text(),
                },
            ),
            program=program,
        )

    def repair_compile(self, prompt: str, report: Any) -> NaturalLanguageCompileResult:
        return self.compile_with_backbone_output(prompt, repair_context=report)


@dataclass(frozen=True)
class RepairPrediction:
    canonical_prompt: str


@dataclass(frozen=True)
class CorruptedProgramSpec:
    name: str
    program: tuple[Instruction, ...]


def build_repair_training_examples(
    tasks: Sequence[NaturalLanguageTask],
    *,
    corruption_names: Sequence[str] = ("drop_halt", "swap_arithmetic", "redirect_jump"),
    vm: VM | None = None,
) -> list[RepairTrainingExample]:
    machine = vm if vm is not None else VM()
    critic = DifferentialCritic()
    examples: list[RepairTrainingExample] = []
    for task in tasks:
        for corruption in generate_corrupted_programs(task.gold_program, corruption_names=corruption_names):
            report = critic.compare_programs(machine, corruption.program, task.gold_program, task_prompt=task.prompt)
            if report.failure_type == "SUCCESS":
                continue
            examples.append(
                RepairTrainingExample(
                    prompt=task.prompt,
                    repair_state=build_repair_state(report),
                    target_canonical_prompt=task.canonical_prompt,
                    gold_program=task.gold_program,
                    task_type=task.task_type,
                    corruption_name=corruption.name,
                )
            )
    return examples


def build_model_driven_repair_compiler(
    delegate: BackboneConditionedCompiler,
    examples: Sequence[RepairTrainingExample],
) -> ModelDrivenRepairCompiler:
    if not examples:
        raise ValueError("model-driven repair compiler requires at least one training example")
    vocabulary = RepairVocabulary.from_examples(examples)
    target_vocabulary = RepairTargetVocabulary.from_examples(examples)
    model = LearnedRepairModel(vocabulary=vocabulary, target_vocabulary=target_vocabulary)
    return ModelDrivenRepairCompiler(delegate=delegate, repair_model=model)


def evaluate_model_driven_repair(
    compiler: ModelDrivenRepairCompiler,
    examples: Sequence[RepairTrainingExample],
) -> LearnedRepairMetrics:
    if not examples:
        return LearnedRepairMetrics(canonical_accuracy=0.0)
    correct = 0
    for example in examples:
        predicted = compiler.repair_model.predict_canonical_prompt(example.prompt, example.repair_state)
        if predicted == example.target_canonical_prompt:
            correct += 1
    return LearnedRepairMetrics(canonical_accuracy=correct / len(examples))


def build_held_out_repair_benchmark(
    tasks: Sequence[NaturalLanguageTask],
    *,
    corruption_names: Sequence[str] = ("shift_const",),
) -> tuple[RepairBenchmarkCase, ...]:
    cases: list[RepairBenchmarkCase] = []
    for task in tasks:
        corruptions = generate_corrupted_programs(task.gold_program, corruption_names=corruption_names)
        if not corruptions:
            continue
        first = corruptions[0]
        cases.append(RepairBenchmarkCase(task=task, initial_program=first.program, corruption_name=first.name))
    return tuple(cases)


def run_repair_benchmark(
    controller: RepairLoopController,
    compiler: RepairCapableCompiler,
    cases: Sequence[RepairBenchmarkCase],
) -> RepairBenchmarkReport:
    baseline_successes = 0
    repaired_successes = 0
    results: list[RepairLoopResult] = []
    task_type_totals: dict[str, int] = {}
    baseline_by_task: dict[str, int] = {}
    repaired_by_task: dict[str, int] = {}

    for case in cases:
        initial_backbone_result = compiler.compile_with_backbone_output(case.task.prompt)
        initial_result = NaturalLanguageCompileResult(
            backbone_output=initial_backbone_result.backbone_output,
            program=case.initial_program,
        )
        baseline_report = controller.critic.compare_programs(
            controller.vm,
            case.initial_program,
            case.task.gold_program,
            task_prompt=case.task.prompt,
        )
        baseline_success = baseline_report.failure_type == "SUCCESS"
        baseline_successes += int(baseline_success)
        task_type_totals[case.task.task_type] = task_type_totals.get(case.task.task_type, 0) + 1
        baseline_by_task[case.task.task_type] = baseline_by_task.get(case.task.task_type, 0) + int(baseline_success)

        result = controller.run_with_initial_result(case.task, compiler, initial_result)
        results.append(result)
        repaired_successes += int(result.success)
        repaired_by_task[case.task.task_type] = repaired_by_task.get(case.task.task_type, 0) + int(result.success)

    total = len(cases)
    task_type_improvement = {
        task_type: (repaired_by_task.get(task_type, 0) - baseline_by_task.get(task_type, 0)) / count
        for task_type, count in task_type_totals.items()
    }
    metrics = evaluate_repair_loop(results)
    baseline_success_rate = (baseline_successes / total) if total else 0.0
    repaired_success_rate = (repaired_successes / total) if total else 0.0
    return RepairBenchmarkReport(
        metrics=metrics,
        results=tuple(results),
        baseline_success_rate=baseline_success_rate,
        repaired_success_rate=repaired_success_rate,
        average_improvement=repaired_success_rate - baseline_success_rate,
        task_type_improvement=task_type_improvement,
    )


def generate_corrupted_programs(
    program: Sequence[Instruction],
    *,
    corruption_names: Sequence[str],
) -> list[CorruptedProgramSpec]:
    variants: list[CorruptedProgramSpec] = []
    for corruption_name in corruption_names:
        variant = _corrupt_program(program, corruption_name)
        if variant is not None:
            variants.append(CorruptedProgramSpec(name=corruption_name, program=variant))
    return variants


def _corrupt_program(program: Sequence[Instruction], corruption_name: str) -> tuple[Instruction, ...] | None:
    if corruption_name == "drop_halt":
        if len(program) > 1 and program[-1].opcode == "HALT":
            return tuple(program[:-1])
        return None

    if corruption_name == "swap_arithmetic":
        swaps = {"ADD": "SUB", "SUB": "ADD", "MUL": "ADD", "DIV": "MUL"}
        for index, instruction in enumerate(program):
            if instruction.opcode in swaps:
                mutated = list(program)
                mutated[index] = Instruction(
                    swaps[instruction.opcode],
                    dst=instruction.dst,
                    src1=instruction.src1,
                    src2=instruction.src2,
                    imm=instruction.imm,
                    label=instruction.label,
                    type_tag=instruction.type_tag,
                )
                return tuple(mutated)
        return None

    if corruption_name == "shift_const":
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
                return tuple(mutated)
        return None

    if corruption_name == "redirect_jump":
        for index, instruction in enumerate(program):
            if instruction.opcode in {"JMP", "JZ", "JNZ", "JLT", "JGT", "CALL"} and type(instruction.imm) is int:
                new_target = 0 if instruction.imm != 0 else min(len(program) - 1, 1)
                mutated = list(program)
                mutated[index] = Instruction(
                    instruction.opcode,
                    dst=instruction.dst,
                    src1=instruction.src1,
                    src2=instruction.src2,
                    imm=new_target,
                    label=instruction.label,
                    type_tag=instruction.type_tag,
                )
                return tuple(mutated)
        return None

    raise ValueError(f"unsupported corruption {corruption_name!r}")
