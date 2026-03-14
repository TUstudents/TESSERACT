# TESSERACT Implementation Plan

## Purpose

This document turns the architecture described in `docs/TESSERACT_Design_Document_v0_2.md` into a concrete, phased implementation roadmap for the current repository state.

The guiding principle is:

- keep exactness in the VM,
- keep learning in the compiler/backbone/critic,
- add features only when they are testable,
- test every feature as it lands.

---

## Current Repository Assessment

Reviewed files:

- `README.md`
- `docs/TESSERACT_Design_Document_v0_2.md`
- `pyproject.toml`
- `src/tesseract/vm/ir.py`
- `src/tesseract/vm/machine.py`
- `src/tesseract/vm/state.py`
- `src/tesseract/compiler/interface.py`
- `src/tesseract/critic/interface.py`
- `tests/test_vm.py`

### Current implementation status

The repository is a scaffold, not yet a functional prototype.

Implemented today:

- a minimal `Instruction` dataclass,
- a minimal `VMState`,
- a `VM` that supports only `CONST`, `ADD`, and `HALT`,
- protocol-only compiler and critic interfaces,
- one VM unit test.

### Current gaps relative to the design document

Missing major subsystems:

- typed IR semantics,
- full opcode set,
- control-flow execution,
- memory and stack semantics,
- call/return behavior,
- runtime traps and debug modes,
- static type checking,
- assembler and label resolution,
- serialization and replay,
- trace capture and differential analysis,
- critic implementation,
- repair loop,
- synthetic supervision datasets,
- backbone/compiler model implementation,
- natural-language compilation benchmarks,
- reproducible evaluation harness.

### Test findings

Current test behavior:

- `pytest -q` fails with `ModuleNotFoundError: tesseract`
- `uv run pytest -q` fails with the same import error
- `PYTHONPATH=src pytest -q` passes

This means repository bootstrap and packaging ergonomics must be addressed before major implementation work.

---

## Implementation Principles

1. **VM first**: the exact executor must be trustworthy before training learned components.
2. **Static checks before runtime when possible**: malformed IR should be rejected early.
3. **Deterministic by default**: traces, outputs, and failures must be reproducible.
4. **Feature-complete tests**: every feature needs positive, negative, and integration coverage where applicable.
5. **No hidden shortcutting**: later ML evaluation must verify that success comes from VM execution, not direct answer heuristics.
6. **Phase gates**: no phase starts before the previous one has green tests and documented acceptance criteria.

---

## Testing Strategy

Every implemented feature should come with the following categories of tests as applicable:

- **Unit tests**: individual opcodes, helper functions, validators, serializers.
- **Negative tests**: malformed instructions, invalid labels, type errors, traps.
- **Integration tests**: full programs spanning multiple instructions and subsystems.
- **Replay tests**: serialized programs/traces reproduce the same outputs.
- **Regression tests**: added whenever a bug is fixed.
- **Acceptance tests**: milestone-level tests per phase.

Each phase below includes explicit testing requirements.

---

## Phase 0 — Repository Bootstrap and Test Harness Hardening

### Goal

Make the repository runnable, installable, and testable from a clean checkout.

### Work items

- Fix packaging/import path so `pytest` works without setting `PYTHONPATH=src` manually.
- Add development dependencies for testing and linting.
- Add a standard local workflow for:
  - install,
  - test,
  - lint,
  - type-check.
- Add basic CI configuration.
- Add common test fixtures/utilities for VM programs and states.
- Document the developer workflow in `README.md` or a dedicated development guide.

### Deliverables

- root-level test execution works,
- repeatable dev commands,
- baseline CI,
- documented setup and test instructions.

### Tests required

- package import smoke test,
- editable install smoke test,
- `pytest` invocation from repo root,
- `uv run pytest` invocation from repo root,
- CI smoke test on a fresh environment.

### Exit criteria

- `pytest` passes from repo root,
- package import succeeds without manual path hacks,
- CI is green.

---

## Phase 1 — Typed IR and Exact VM Core

### Goal

Implement the symbolic execution core described in the design document.

### Work items

#### 1. Expand the IR

Implement a stronger IR model with:

- validated opcode definitions,
- explicit operands,
- labels,
- immediates,
- type tags,
- optional instruction metadata for debugging.

#### 2. Expand VM state

Add explicit support for:

- register file,
- RAM,
- call stack,
- flags,
- halt status,
- trap metadata,
- step counter.

#### 3. Implement the phase-1 opcode set

Initial target opcode set:

- data movement:
  - `MOV`
  - `CONST`
- arithmetic:
  - `ADD`
  - `SUB`
  - `MUL`
  - `DIV`
- logical:
  - `AND`
  - `OR`
  - `NOT`
  - `XOR`
- comparison:
  - `CMP_EQ`
  - `CMP_LT`
  - `CMP_GT`
- control flow:
  - `JMP`
  - `JZ`
  - `JNZ`
  - `JLT`
  - `JGT`
  - `HALT`
- memory:
  - `LOAD`
  - `STORE`
- stack/calls:
  - `PUSH`
  - `POP`
  - `CALL`
  - `RET`

#### 4. Implement runtime trap semantics

Required trap classes:

- invalid opcode,
- divide-by-zero,
- invalid address,
- type mismatch,
- timeout,
- optional checked overflow for debug mode.

#### 5. Deterministic execution tracing

Add optional trace capture with per-step records of:

- current instruction,
- pre-state snapshot or summary,
- post-state snapshot or summary,
- trap if raised.

### Deliverables

- complete reference VM in Python,
- deterministic execution semantics,
- trap model,
- core opcode coverage.

### Tests required

#### Unit tests

- one unit test per opcode,
- register read/write behavior,
- default register semantics,
- memory load/store correctness,
- stack push/pop correctness,
- call/return behavior,
- flags and comparison behavior.

#### Negative tests

- invalid opcode trap,
- divide-by-zero trap,
- invalid address trap,
- timeout trap,
- type trap,
- stack underflow behavior if modeled as trap.

#### Integration tests

- arithmetic expression evaluation,
- factorial,
- Fibonacci,
- loop-based accumulation,
- simple function call program,
- small array reduction via memory.

#### Determinism tests

- repeated execution produces identical final state,
- repeated execution produces identical trace,
- same program + same state => same trap kind and location.

### Exit criteria

- VM semantics are stable and documented,
- opcode coverage is comprehensive,
- all runtime failure modes are test-covered.

---

## Phase 2 — Static Analysis, Assembler, and Serialization

### Goal

Make programs validatable, inspectable, and serializable.

### Work items

#### 1. Static validation

Implement compile-time validation for:

- invalid register references,
- undefined labels,
- malformed instructions,
- invalid operand combinations,
- static type mismatches where checkable.

#### 2. Type checker

Implement a prototype type system for the IR with judgments sufficient for early-phase VM correctness.

#### 3. Assembler and label resolution

Support:

- symbolic labels,
- label-to-address resolution,
- assembly of human-readable program definitions,
- disassembly back to readable form.

#### 4. Serialization

Add canonical JSON serialization for:

- instructions,
- programs,
- traces,
- trap outcomes,
- optional state snapshots.

#### 5. Replay tooling

Allow deterministic replay of a stored program and trace artifact.

### Deliverables

- static checker,
- assembler/disassembler,
- serialization format,
- replay support.

### Tests required

- assembler round-trip tests,
- label resolution tests,
- program serialization round-trip tests,
- trace serialization round-trip tests,
- static rejection tests for malformed programs,
- type checker tests for valid and invalid instruction patterns,
- replay consistency tests.

### Exit criteria

- invalid programs are rejected before execution when possible,
- valid programs can be stored, loaded, and replayed exactly.

---

## Phase 3 — Synthetic Task Corpus and Compiler Baseline

### Goal

Train the first learned compiler using supervised synthetic data with gold IR.

### Work items

#### 1. Dataset generation

Build synthetic corpora for:

- arithmetic,
- branching,
- loops,
- memory manipulation,
- sorting small arrays,
- graph reachability,
- finite-state transitions,
- execution trace prediction tasks.

Each example should include, where possible:

- task input,
- expected output,
- gold IR,
- optional gold trace,
- metadata for evaluation.

#### 2. Compiler baseline

Implement the recommended first compiler:

- autoregressive IR decoder.

#### 3. Training loop

Add supervised training for:

- IR prediction,
- compile validity,
- execution correctness,
- optional type and length regularization.

#### 4. Metrics and evaluation

Track:

- exact output accuracy,
- exact IR match,
- execution success rate,
- compile validity rate,
- average program length,
- trap rate by category.

### Deliverables

- synthetic dataset generators,
- baseline compiler implementation,
- train/eval scripts,
- metrics dashboard or report output.

### Tests required

- dataset generator correctness tests,
- gold IR validity tests,
- small-batch training smoke test,
- overfit-a-tiny-dataset test,
- inference output shape/schema tests,
- decoded-program validity tests,
- execution agreement tests on a fixed mini benchmark.

### Exit criteria

- compiler can overfit a small synthetic set,
- generated programs execute with measurable accuracy,
- metrics are stable and reproducible.

---

## Phase 4 — Trace Capture, Critic Scaffolding, and Repair Diagnostics

### Goal

Build the execution-observability and failure-localization layer.

### Work items

#### 1. Trace schema

Define a formal trace object with:

- step index,
- instruction,
- pre-state summary,
- post-state summary,
- branch decisions,
- memory effects,
- trap metadata,
- optional invariant checks.

#### 2. Differential trace analysis

Implement utilities to compare:

- candidate trace vs gold trace,
- candidate final state vs gold final state,
- first divergent step,
- branch divergence,
- register divergence,
- memory divergence.

#### 3. Critic baseline

Move beyond the protocol stub and implement a structured critic interface that can emit:

- first failing step,
- failure type,
- localization details,
- repair hints.

#### 4. Invariant instrumentation

Support assertions such as:

- type invariants,
- bounds invariants,
- sortedness,
- conservation/count invariants,
- task-specific postconditions.

### Deliverables

- trace schema,
- diff utilities,
- critic baseline implementation,
- invariant instrumentation.

### Tests required

- trace emission tests,
- trace diff tests,
- first-divergence detection tests,
- invariant trigger tests,
- critic output schema tests,
- failure localization tests on synthetic corrupt programs.

### Exit criteria

- failed executions produce structured diagnostics,
- divergence can be localized automatically on benchmark cases.

---

## Phase 5 — Natural-Language Compilation Path

### Goal

Connect a semantic backbone to the compiler so natural-language tasks can compile to executable IR.

### Work items

#### 1. Backbone implementation

Add a real backbone module rather than just package placeholders.

#### 2. NL-to-IR task scope

Start with tightly scoped tasks such as:

- add two numbers,
- compare two numbers,
- compute factorial,
- sum a memory range,
- execute a simple branching procedure.

#### 3. Backbone-to-compiler interface

Define how semantic representations condition IR decoding.

#### 4. Sequence/string extensions if needed

Only add sequence-aware opcodes after the scalar pipeline works. Possible additions:

- `LEN`
- `SLICE`
- `SEQ_LOAD`
- `SEQ_STORE`
- `TOKEN_EQ`
- `CHAR_EQ`

### Deliverables

- minimal backbone implementation,
- NL-to-IR dataset format,
- end-to-end prompt → program → execution pipeline.

### Tests required

- dataset parsing tests,
- prompt-to-program smoke tests,
- backbone/compiler interface tests,
- end-to-end NL → IR → VM execution tests,
- anti-shortcut tests where VM execution is ablated or emitted IR is randomized.

### Exit criteria

- simple NL tasks compile into valid executable IR,
- exact answer quality depends on VM execution rather than direct heuristic answering.

---

## Phase 6 — Iterative Repair Loop

### Goal

Implement compile → execute → diagnose → repair → recompile.

### Work items

#### 1. Repair context design

Define a compact repair representation containing:

- original task,
- failing program summary,
- failure localization,
- critic hint,
- optional compressed trace digest.

#### 2. Repair-conditioned decoding

Allow the compiler to produce a revised program from repair context.

#### 3. Loop controller

Add deterministic loop semantics with:

- round cap,
- success/failure termination,
- oscillation detection,
- timeout handling.

#### 4. Metrics

Track:

- success after 1 round,
- success after 2 rounds,
- success after 3 rounds,
- non-convergence rate,
- oscillation rate,
- repair cost in extra steps.

### Deliverables

- repair loop controller,
- repair-conditioned compiler path,
- repair evaluation harness.

### Tests required

- loop state-machine tests,
- max-round termination tests,
- repair-context schema tests,
- synthetic broken-program repair tests,
- regression tests demonstrating improvement over one-shot compilation.

### Exit criteria

- repair measurably improves benchmark success,
- repair loop terminates predictably and logs failure reasons.

---

## Phase 7 — Performance, Reproducibility, and Evaluation Hardening

### Goal

Make the reference implementation robust enough for sustained research work.

### Work items

#### 1. Reproducibility

- fixed seeds,
- deterministic config capture,
- artifact versioning,
- benchmark freeze files.

#### 2. Performance

- profile VM throughput,
- optimize Python hotspots,
- add checkpointing/replay optimizations,
- consider a Rust/C++ VM only after Python semantics are locked.

#### 3. Evaluation hardening

Add stable benchmark suites for:

- exact execution tasks,
- critic localization,
- repair improvement,
- macro-step efficiency,
- anti-shortcut behavior.

#### 4. Reporting

Generate machine-readable and human-readable experiment summaries.

### Deliverables

- reproducible experiment harness,
- benchmark suites,
- performance reports,
- artifact logging strategy.

### Tests required

- fixed-seed reproducibility tests,
- benchmark sanity tests,
- replay consistency tests across saved artifacts,
- long-run timeout stability tests,
- evaluation report generation tests.

### Exit criteria

- experiments are repeatable,
- benchmarks are stable,
- the Python reference implementation is trustworthy as the semantic baseline.

---

## Recommended Build Order

To minimize rework, implementation should proceed in this order:

1. **Phase 0** — fix packaging and tests first
2. **Phase 1** — complete the exact VM
3. **Phase 2** — add validation, assembly, serialization, replay
4. **Phase 3** — build synthetic supervision and compiler baseline
5. **Phase 4** — add traces and critic scaffolding
6. **Phase 5** — connect natural language to compilation
7. **Phase 6** — implement iterative repair
8. **Phase 7** — optimize, benchmark, and harden reproducibility

This ordering preserves the architecture’s intended separation of concerns and avoids training against unstable executor semantics.

---

## Immediate Next Actions

1. Fix package/test execution so `pytest` works from the repository root.
2. Expand the VM from `CONST`/`ADD`/`HALT` into the full phase-1 reference executor.
3. Add exhaustive tests for each new opcode and each trap condition.
4. Only after the VM is stable, begin building static checking and the compiler dataset/tooling.

---

## Definition of Done for the Prototype

A meaningful prototype is complete when the repository can demonstrate:

- a deterministic exact VM with typed IR,
- full test coverage for core execution features,
- static validation and replayable traces,
- a supervised compiler that can emit valid programs for synthetic tasks,
- a critic that can localize failures,
- a repair loop that improves execution success,
- basic NL-to-IR examples that genuinely depend on VM execution.
