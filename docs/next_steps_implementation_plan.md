# TESSERACT Next Steps Implementation Plan

## Purpose

This document turns the current post-prototype review into a concrete, numbered implementation plan for the next development cycle.

It assumes the current repository already has:

- a tested Python reference VM,
- validation, assembly, serialization, and replay tooling,
- a synthetic compiler baseline,
- a rule-based natural-language path,
- a deterministic critic/repair scaffold,
- reproducibility and benchmark/reporting helpers.

The goal of this plan is to move the repository from a well-tested scaffolded prototype toward the fuller architecture described in `docs/TESSERACT_Design_Document_v0_2.md`, without destabilizing the exact-execution core.

---

## Guiding principles

1. **Keep exactness in the VM.** Do not move exact symbolic behavior into the backbone.
2. **Upgrade learned subsystems around a fixed executor.** Preserve VM, validator, tokenizer, and replay semantics while replacing placeholder models.
3. **Expand scope incrementally.** Add one major learned or semantic capability at a time.
4. **Preserve reproducibility.** Every new training or evaluation path must have seed control and deterministic smoke coverage.
5. **Protect anti-shortcut guarantees.** New models must still be evaluated by emitted-program execution, not direct answer heuristics.
6. **Require phase gates.** No phase is complete without tests, docs updates, and full-project green validation.

---

## Current gap summary

Relative to `docs/TESSERACT_Design_Document_v0_2.md`, the largest remaining gaps are:

1. the backbone is rule-based rather than learned,
2. the compiler is now a small neural autoregressive decoder, but it is still narrow and not yet a strong research-grade architecture,
3. the NL task family is still narrow,
4. the IR/value model is still mostly scalar `int | bool`,
5. sequence/string semantics are still absent,
6. the critic is deterministic rather than learned,
7. repair conditioning is wired but still weak,
8. evaluation is useful but not yet a full research harness.

---

## Numbered implementation plan

## 1. Replace the placeholder compiler with a real neural autoregressive decoder *(completed)*

### Goal

Introduce a PyTorch autoregressive decoder for IR generation while keeping the current tokenizer, validation layer, VM execution path, and public compiler workflow stable.

### Scope

- add a neural compiler model in `src/tesseract/compiler/`,
- preserve `ProgramTokenizer` and current vocabulary machinery unless a compatibility-preserving extension is required,
- keep compile-time validation and execution-based evaluation as the primary correctness path,
- keep the current count-based compiler as a baseline/reference implementation.

### Work items

1. Add a concrete neural decoder model, likely including:
   - prompt embedding,
   - program-token embedding,
   - autoregressive decoding,
   - teacher-forced next-token loss.
2. Add a training wrapper that mirrors current compiler training conventions.
3. Add inference-time decoding for `<bos> -> ... -> <eos>` generation.
4. Add checkpoint save/load helpers.
5. Add evaluation metrics for:
   - token accuracy,
   - exact program match,
   - compile validity rate,
   - execution accuracy.
6. Keep the current baseline code path available for comparison.

### Deliverables

- neural autoregressive compiler model,
- training/inference wrapper,
- checkpointing support,
- baseline-vs-neural evaluation path.

### Tests required

- model forward-pass shape tests,
- seeded training-step reproducibility smoke test,
- decode termination and malformed-output handling tests,
- checkpoint round-trip test,
- execution-backed integration test on synthetic tasks,
- regression test proving neural path still routes through validation and VM execution.

### Exit criteria

- neural compiler can be trained and decoded end to end,
- compile outputs are validated before execution,
- execution accuracy is measurable through the existing evaluation stack,
- current tests remain green.

---

## 2. Add a learned backbone and real conditioning path into the compiler *(completed)*

### Goal

Replace prompt regex normalization as the main semantic frontend with a learned backbone that conditions IR decoding.

### Scope

- keep `RuleBasedBackbone` as a baseline/debug implementation,
- add a learned backbone in `src/tesseract/backbone/`,
- make compiler conditioning depend on learned representations rather than only canonical prompt rewriting.

### Work items

1. Define a concrete backbone module interface beyond the current protocol.
2. Add a small PyTorch backbone model suitable for the current task scale.
3. Define the conditioning contract between backbone and compiler, for example:
   - pooled embedding,
   - token-level memory,
   - task/planner summary vector,
   - repair-summary embedding.
4. Update `BackboneConditionedCompiler` so conditioning can use learned representations directly.
5. Keep canonical prompts and gold IR supervision for bootstrap training.

### Deliverables

- learned backbone implementation,
- backbone-to-compiler conditioning path,
- baseline comparison against the rule-based backbone.

### Tests required

- backbone output schema/shape tests,
- deterministic seeded smoke tests,
- learned backbone + compiler integration test,
- regression test showing unsupported prompts fail cleanly where expected,
- anti-shortcut tests confirming answers still depend on emitted IR execution.

### Exit criteria

- learned conditioning is used in compilation,
- rule-based backbone remains available as a baseline,
- end-to-end NL -> IR -> VM path works with learned conditioning.

---

## 3. Expand task scope beyond arithmetic/max/sum-to-n *(completed)*

### Goal

Broaden the synthetic and natural-language task families so the system is tested on richer control flow, iteration, and memory use.

### Priority task additions

1. factorial,
2. Fibonacci,
3. simple branching procedures,
4. memory-range summation,
5. small array tasks after sequence/memory semantics are ready.

### Work items

1. Extend synthetic task generation and gold-program builders.
2. Add canonical prompts and NL templates for each new task family.
3. Add held-out task splits for training/evaluation.
4. Extend benchmark suite construction to report per-task-family metrics.
5. Preserve exact gold execution and validation for every task.

### Deliverables

- broader synthetic corpus,
- broader NL corpus,
- per-task-family benchmark coverage.

### Tests required

- gold-program correctness tests per task family,
- NL parsing/template tests,
- end-to-end compile/execute tests for each task family,
- anti-shortcut ablations per task family,
- benchmark split reproducibility tests.

### Exit criteria

- system supports multiple nontrivial algorithmic task families,
- held-out execution quality is measurable by family,
- benchmark outputs remain deterministic.

---

## 4. Strengthen the IR and value model toward the design document

### Goal

Incrementally expand the executor toward the richer machine model in the design doc without destabilizing the current semantics.

### Recommended order

1. `checked_i64`,
2. `f32`,
3. clearer pointer/address discipline,
4. sequence operations.

### Candidate sequence/string opcodes

- `LEN`,
- `SEQ_LOAD`,
- `SEQ_STORE`,
- `SLICE`,
- `TOKEN_EQ`,
- `CHAR_EQ`.

### Work items

1. Extend type tags and runtime coercion rules carefully.
2. Extend static validation/type propagation accordingly.
3. Add any new IR operand or metadata forms only when required.
4. Update serialization, replay, assembler/disassembler, and tests together.
5. Document exact semantics for every new value type or opcode.

### Deliverables

- expanded type/value coverage,
- extended validator,
- extended serializer/replay support,
- sequence-aware execution primitives where implemented.

### Tests required

- unit tests per new opcode/type,
- negative tests for type/address/trap behavior,
- serialization round-trip tests,
- replay consistency tests,
- mixed integration programs using new semantics.

### Exit criteria

- new types/opcodes are documented, validated, serializable, and replayable,
- VM behavior remains deterministic,
- no regressions in existing scalar semantics.

---

## 5. Upgrade the critic from deterministic scaffold to trainable subsystem

### Goal

Build a learned trace critic while keeping the current differential critic as the oracle/reference implementation.

### Scope

- do not remove `DifferentialCritic`,
- use current execution comparison and invariants to derive supervision,
- add a learned critic model in parallel.

### Work items

1. Build critic-training data from gold vs corrupted/candidate programs.
2. Generate labels for:
   - first failing step,
   - failure type,
   - differing register/address,
   - repair-hint class or patch category.
3. Add a PyTorch critic model.
4. Add evaluation utilities comparing learned outputs to oracle-derived labels.
5. Keep repair prompt generation aligned with existing `CriticReport` structure.

### Deliverables

- critic dataset generator,
- learned critic model,
- critic evaluation harness.

### Tests required

- critic-label correctness tests,
- dataset generation determinism tests,
- learned critic output-schema tests,
- agreement tests against oracle targets,
- regression tests preserving current `CriticReport` compatibility.

### Exit criteria

- learned critic can be trained on oracle-derived supervision,
- critic outputs remain compatible with repair infrastructure,
- oracle and learned critic can be compared under a shared evaluation API.

---

## 6. Make repair genuinely model-driven

### Goal

Move from scaffolded repair wiring to a real repair-conditioned compile loop.

### Scope

- keep deterministic controller semantics,
- make the compiler/backbone meaningfully condition on repair state,
- measure repair improvement on held-out failures.

### Work items

1. Define a compact repair-state representation for training/inference.
2. Add repair-conditioned decoding inputs to the learned backbone/compiler stack.
3. Build held-out broken-program or hard-task repair evaluation sets.
4. Measure:
   - success after 1/2/3 rounds,
   - oscillation rate,
   - non-convergence rate,
   - repair cost,
   - task-family-specific improvement.
5. Keep oscillation and max-round termination behavior deterministic.

### Deliverables

- repair-conditioned decoding path,
- held-out repair benchmark,
- repair-improvement reporting.

### Tests required

- repair-state serialization/schema tests if serialized,
- multi-round deterministic behavior tests,
- held-out repair improvement smoke tests,
- regression tests for oscillation and max-round termination,
- compatibility tests with current `RepairLoopController` result schema.

### Exit criteria

- repair measurably improves over one-shot compilation on held-out failures,
- termination remains predictable,
- repair metrics are reported automatically.

---

## 7. Harden evaluation into a research-grade experiment harness

### Goal

Extend the current evaluation helpers into a fuller experiment and benchmarking layer aligned with the design document.

### Work items

1. Add benchmark suites for:
   - exact execution,
   - critic localization,
   - repair improvement,
   - anti-shortcut behavior,
   - macro-step efficiency.
2. Add artifact capture for:
   - model config,
   - seed,
   - benchmark suite freeze payload,
   - checkpoint metadata,
   - code/version identifiers where practical.
3. Add richer reporting:
   - per-task-family metrics,
   - compile-vs-execute failure breakdown,
   - repair-round curves,
   - trace-length/program-length summaries.
4. Add basic performance instrumentation for the Python VM and evaluation stack.

### Deliverables

- broader benchmark suite set,
- experiment manifests/artifact capture,
- richer reporting outputs,
- basic performance summaries.

### Tests required

- benchmark freeze/load round-trip tests,
- report-generation tests for new metrics,
- reproducibility tests across seeds and saved artifacts,
- timeout-stability tests,
- macro-step metric sanity tests.

### Exit criteria

- experiments are reproducible from frozen artifacts,
- benchmarks cover output quality, failure localization, repair, and anti-shortcut behavior,
- reports support research iteration instead of only smoke validation.

---

## Recommended execution order

Implement the phases in this order:

1. neural autoregressive compiler,
2. learned backbone conditioning,
3. expanded task families,
4. richer IR/value semantics,
5. learned critic,
6. learned repair,
7. research-grade evaluation hardening.

This order preserves the current exact-execution baseline while upgrading the learned components around it.

---

## Immediate next action

Start with **Step 4: strengthen the IR and value model toward the design document**.

That is now the highest-leverage next change because the repository has a broader learned compiler/backbone path and expanded task coverage, but the executor is still much narrower than the design document’s richer machine model.
