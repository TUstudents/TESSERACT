# TESSERACT: A Compiler–Executor Language Model
## Complete Design Proposal and Implementation Plan

**Version:** 0.2  
**Date:** March 2026  
**Status:** Revised review draft

---

## Abstract

TESSERACT (**T**yped **E**xecutor with **S**emantic **S**tate, **E**xact **R**AM, **A**nd **C**ompiler-**T**) is a bimodal language-model architecture that separates continuous semantic reasoning from exact symbolic execution. Instead of forcing attention to simulate a computer directly, TESSERACT learns to compile tasks into an internal typed intermediate representation (IR), executes that IR on an exact virtual machine (VM) with persistent memory and explicit control flow, and uses a trace critic to diagnose failures and guide repair. The central design thesis is that exactness should live in the VM semantics, while learning should live in the compiler, semantic backbone, and critic. This document presents the revised full proposal, formal model, subsystem theory, implementation plan, training strategy, milestones, resource estimates, risks, and review criteria.

---

## Table of Contents

1. Motivation
2. Design thesis
3. System overview
4. Formal machine model
5. Semantic backbone
6. Latent compiler
7. Typed intermediate representation
8. Exact virtual machine
9. Trace critic and repair loop
10. Training theory and objectives
11. Complexity and scaling analysis
12. Verification strategy
13. Reference implementation plan
14. Roadmap, team, and compute estimates
15. Risks and open problems
16. Review checklist
17. Appendix: core math and formulas

---

## 1. Motivation

### 1.1 Problem statement

A practical Turing-complete LLM must solve three problems simultaneously:

1. **Exact state evolution**: symbolic state must update by formal rules, not approximate hidden-state drift.
2. **Trainable credit assignment**: the model must learn which computation to run and how to repair failures.
3. **Mutable executable memory**: the model needs read/write memory with overwrite semantics, not only retrieval over past tokens.

Most current approaches solve only part of the problem:
- attention constructions can show theoretical expressivity but not practical exact execution,
- tool-use systems externalize computation,
- memory-augmented networks offer writable memory but historically struggle with training and scale.

### 1.2 Clarifying the limitation of tool-use systems

Tool-use systems preserve meaningful **interface-level state visibility**: the model can see tool inputs, tool outputs, and sometimes serialized execution traces in the conversational context. What they do not generally preserve is:
- **gradient-level visibility** through execution,
- **latent-space continuity** across internal machine states,
- **direct access to intermediate executable state** during learning.

TESSERACT is aimed at an internal executable state substrate rather than external orchestration alone.

### 1.3 Thesis

The right question is not:

> Can a transformer itself be Turing-complete?

The right question is:

> Can a language model learn to compile tasks into an internal exact machine that is efficient, verifiable, and repairable?

TESSERACT answers that question architecturally.

---

## 2. Design thesis

### 2.1 Separation of concerns

TESSERACT explicitly separates:

- **semantic modeling**: approximate, neural, continuous,
- **execution semantics**: exact, discrete, deterministic.

This yields a clean division:

- the **backbone** understands the task,
- the **compiler** emits a typed latent program,
- the **VM** executes it exactly,
- the **critic** diagnoses failures and guides repair.

### 2.2 Why this is different from retrieval-based exactness

Retrieval-centric architectures treat computation as selecting the right previous state. That is insufficient because computation also requires:
- writable memory,
- explicit control flow,
- branching and loops,
- stack and call semantics,
- typed arithmetic,
- failure localization.

TESSERACT is therefore a **compiler–executor architecture**, not a better retrieval trick.

---

## 3. System overview

The system has four main modules.

### Module A — Semantic Backbone

A standard transformer or transformer–SSM hybrid reads the input tokens and builds a semantic latent state:

\[
h_{1:n}^{(L)} = \mathrm{Backbone}(x_{1:n}, \rho),
\]

where \(\rho\) is optional repair context.

Responsibilities:
- language understanding,
- task decomposition,
- algorithm selection,
- planning,
- semantic grounding.

### Module B — Latent Compiler

A decoder emits a latent typed program:

\[
\pi = (\iota_1, \iota_2, \dots, \iota_T), \qquad \iota_t \in \mathcal{I}.
\]

Responsibilities:
- choose opcodes,
- allocate registers,
- assign control-flow labels,
- generate memory-access patterns,
- construct typed executable traces.

### Module C — Exact Virtual Machine

The VM executes the latent program exactly:

\[
S_{t+1} = F_{\mathrm{VM}}(S_t, \pi[\mathrm{pc}_t]).
\]

Responsibilities:
- register operations,
- memory read/write,
- stack and call semantics,
- branch resolution,
- deterministic halting,
- timeout and trap signaling.

### Module D — Trace Critic

The critic observes execution traces and emits repair signals:

\[
c = \mathrm{Critic}(\mathcal{T}, y, y^\star, z),
\]

where \(z\) may include gold traces, assertions, or execution diagnostics.

Responsibilities:
- identify first failing step,
- diagnose branch or memory errors,
- propose recompile/patch actions,
- supervise iterative repair.

### 3.1 Repair-time data flow

During repair, the Backbone is re-invoked on a compressed repair context rather than necessarily on the full raw trace. Let round \(k\) repair context be

\[
\rho^{(k)} = \mathrm{Summarize}(x, \pi^{(k)}, c^{(k)}, d^{(k)}),
\]

where \(d^{(k)}\) is a trace digest. Then

\[
h^{(k)} = \mathrm{Backbone}(x, \rho^{(k)}),
\qquad
p_\theta(\pi^{(k+1)} \mid h^{(k)}).
\]

This makes the repair loop explicit.

---

## 4. Formal machine model

### 4.1 State space

Let the VM state be

\[
S_t = (R_t, M_t, \Sigma_t, \mathrm{pc}_t, C_t),
\]

where:

- \(R_t\): register file,
- \(M_t\): RAM,
- \(\Sigma_t\): call stack,
- \(\mathrm{pc}_t\): program counter,
- \(C_t\): control flags.

#### Registers

\[
R_t : \{1,\dots,N_R\} \to \mathcal{V}
\]

#### Memory

\[
M_t : \{0,\dots,N_M-1\} \to \mathcal{V}
\]

#### Stack

\[
\Sigma_t = [f_1, \dots, f_k], \qquad f_i \in \mathcal{F}
\]

#### Value domain

Use a tagged sum:

\[
\mathcal{V} = \mathbb{Z}_{32} \sqcup \mathbb{Z}_{64} \sqcup \mathbb{B} \sqcup \mathbb{F}_{32} \sqcup \mathbb{P} \sqcup \mathcal{E} \sqcup \mathcal{X}
\]

where:
- \(\mathbb{P}\): pointer space,
- \(\mathcal{E}\): optional embedding-valued semantic objects,
- \(\mathcal{X}\): sequence/string handles or array descriptors.

### 4.2 Exact transition semantics

The VM transition function is

\[
F_{\mathrm{VM}} : \mathcal{S} \times \mathcal{I} \to \mathcal{S} \sqcup \mathcal{T}_{\mathrm{trap}},
\]

where \(\mathcal{T}_{\mathrm{trap}}\) is the set of runtime trap states.

Example instructions:
- `MOV r_d, r_s`
- `ADD r_d, r_a, r_b`
- `LOAD r_d, [r_a + imm]`
- `STORE [r_a + imm], r_s`
- `CMP r_a, r_b`
- `JLT label`
- `CALL label`
- `RET`
- `HALT`

For example, `ADD`:

\[
F_{\mathrm{ADD}}(S, (ADD,d,a,b)) = (R', M, \Sigma, \mathrm{pc}+1, C')
\]

with

\[
R'(d) = R(a) + R(b).
\]

### 4.3 Runtime error semantics

The VM supports explicit trap states:
- \(\mathrm{TRAP}_{\mathrm{TYPE}}\)
- \(\mathrm{TRAP}_{\mathrm{ADDR}}\)
- \(\mathrm{TRAP}_{\mathrm{DIV0}}\)
- \(\mathrm{TRAP}_{\mathrm{OVERFLOW}}\) (checked mode only)
- \(\mathrm{TRAP}_{\mathrm{TIMEOUT}}\)
- \(\mathrm{TRAP}_{\mathrm{INVALID\_OP}}\)

Type mismatch example:

\[
F_{\mathrm{VM}}(S_t, \iota_t) = \mathrm{TRAP}_{\mathrm{TYPE}}
\]

when the runtime tag configuration does not match the instruction type judgment.

### 4.4 Integer and floating-point policy

Initial recommended semantics:
- `i32`, `i64`: modular wraparound by default,
- `checked_i32`, `checked_i64`: trap-on-overflow in debug mode,
- `f32`: IEEE-754 semantics,
- invalid pointer arithmetic: address trap.

This separates production semantics from diagnostic semantics.

### 4.5 Step budget and timeout

Execution must be bounded during training and evaluation. Let \(B\) be the step budget. If execution exceeds \(B\), then

\[
\mathrm{Exec}(\pi, x; B) = \mathrm{TRAP}_{\mathrm{TIMEOUT}}.
\]

### 4.6 Turing-completeness of the executor

If the VM supports:
- finite control,
- writable unbounded memory abstraction,
- conditional branching,
- address arithmetic,
- halting,

then it can simulate a Turing machine.

A tape is represented in RAM, the head position in a register, and the transition table in the latent program or dispatch table. Each Turing-machine step maps to a bounded instruction sequence.

**Scope note.** This is a statement about the **unbounded-memory abstraction**. Any concrete implementation with finite RAM is not literally Turing-complete, though it may approximate that regime for intended workloads.

---

## 5. Semantic backbone

### 5.1 Role

The semantic backbone is a standard LLM module that does not need to be exact. Its purpose is to map natural language and context into a latent task state suitable for compilation and repair.

### 5.2 Interface

Inputs:
- user prompt,
- optional retrieved context,
- prior repair summaries,
- optional symbolic state summaries.

Outputs:
- contextual hidden states \(h_{1:n}^{(L)}\),
- optional planner tokens,
- compiler conditioning vectors.

### 5.3 Recommended architecture

Initial implementation:
- 200M–700M parameter transformer,
- standard positional mechanism,
- grouped-query attention,
- 16–32 layers,
- d_model 768–1536.

These sizes are intended to be sufficient for research-scale algorithmic reasoning and natural-language-to-IR experiments without moving immediately into frontier-scale training.

Alternative:
- transformer front-end + SSM blocks for longer repair contexts and trace-summary histories.

### 5.4 Why the backbone should not execute directly

If exact execution is pushed into the backbone:
- gradients and exactness conflict,
- control flow remains implicit,
- memory mutation is hard to certify,
- errors are opaque.

Therefore the backbone should remain a semantic compiler frontend.

---

## 6. Latent compiler

### 6.1 Compiler objective

Given input \(x\), compile to a program \(\pi\):

\[
p_\theta(\pi \mid x).
\]

### 6.2 Latent program length

Let \(T\) be the number of emitted instructions. The compiler should minimize both execution error and unnecessary trace length.

Suggested regularizers:

\[
\mathcal{L}_{\mathrm{len}} = \beta \frac{T}{T^\star}
\quad \text{or} \quad
\mathcal{L}_{\mathrm{len}} = \beta \log(1+T),
\]

where \(T^\star\) is a target or reference program length when available.

### 6.3 Compiler architectures

**Option A: Autoregressive IR decoder**  
Simple, interpretable, easy to train.

**Option B: Segment-wise decoder**  
Emits basic blocks instead of individual instructions.

**Option C: Structured transducer**  
Separates control-flow skeleton from operand filling.

Recommended first prototype: **Option A**.

### 6.4 Compiler outputs

Each instruction includes:
- opcode,
- destination operand,
- source operands,
- immediate if needed,
- type tag,
- optional label metadata.

### 6.5 Compiler constraints

The compiler must satisfy:
- type consistency,
- defined labels,
- no invalid register indices,
- no unresolved control flow,
- bounded memory access when required.

Constraint handling policy:
- static violations become **compile-time errors** and may reject a program before execution,
- additionally, soft penalties may be added during training for near-miss structural violations.

---

## 7. Typed intermediate representation

### 7.1 Instruction format

Define the latent instruction space as:

\[
\mathcal{I} = \mathcal{O} \times \mathcal{A}_1 \times \mathcal{A}_2 \times \mathcal{A}_3 \times \mathcal{T} \times \mathcal{L}
\]

where:
- \(\mathcal{O}\): opcode vocabulary,
- \(\mathcal{A}_i\): operand spaces,
- \(\mathcal{T}\): type tags,
- \(\mathcal{L}\): labels and control annotations.

### 7.2 Minimal initial opcode set

Suggested phase-1 opcode set:
- `MOV`
- `CONST`
- `ADD`, `SUB`, `MUL`, `DIV`
- `AND`, `OR`, `NOT`, `XOR`
- `CMP_EQ`, `CMP_LT`, `CMP_GT`
- `JMP`, `JZ`, `JNZ`, `JLT`, `JGT`
- `LOAD`, `STORE`
- `PUSH`, `POP`
- `CALL`, `RET`
- `HALT`

### 7.3 Type discipline

Each instruction has a type judgment:

\[
\Gamma \vdash \iota_t : \mathrm{ok}
\]

For example,

\[
\frac{\Gamma(r_a)=i32 \quad \Gamma(r_b)=i32 \quad \Gamma(r_d)=i32}{\Gamma \vdash ADD\ r_d, r_a, r_b : \mathrm{ok}}
\]

Type validity can be checked before execution.

### 7.4 Macro-step semantics

One latent instruction can represent many token-level reasoning operations. If implicit chain-of-thought would require \(L\) token steps but the IR uses \(T\) executable instructions with \(T \ll L\), then execution compresses the reasoning trace.

### 7.5 Sequence and string handling

Natural-language-grounded tasks eventually require sequence-aware semantics. This is deferred but explicitly planned.

Minimal phase-3 extension:
- `SEQ_LOAD`
- `SEQ_STORE`
- `LEN`
- `SLICE`
- `TOKEN_EQ`
- `CHAR_EQ`

Represent strings or token arrays as RAM-backed sequences with explicit length metadata.

---

## 8. Exact virtual machine

### 8.1 Responsibilities

The VM is the exact symbolic core. It owns:
- arithmetic correctness,
- branching correctness,
- memory mutation,
- call/return,
- halting semantics.

### 8.2 Memory semantics

Reads:

\[
\mathrm{LOAD}(a) = M_t(a).
\]

Writes:

\[
M_{t+1}(x) =
\begin{cases}
v & \text{if } x = a_t \\
M_t(x) & \text{otherwise}
\end{cases}
\]

This is exact overwrite semantics.

### 8.3 Control flow

Program counter update:

\[
\mathrm{pc}_{t+1} =
\begin{cases}
\ell & \text{if branch condition is true} \\
\mathrm{pc}_t + 1 & \text{otherwise}
\end{cases}
\]

### 8.4 Stack semantics

Call frames contain:
- return address,
- optional local bindings,
- optional type environment.

Push:

\[
\Sigma_{t+1} = \Sigma_t \mathbin{\|} [f]
\]

Pop:

\[
\Sigma_{t+1} = \mathrm{pop}(\Sigma_t)
\]

### 8.5 Determinism

Given fixed program \(\pi\), initial state \(S_0\), exact arithmetic policy, and step budget \(B\), execution is deterministic up to either halting or a unique trap outcome.

---

## 9. Trace critic and repair loop

### 9.1 Trace object

Let execution trace be

\[
\mathcal{T} = (S_0, S_1, \dots, S_m).
\]

### 9.2 Critic outputs

The critic predicts:
- first failing step,
- wrong branch taken,
- wrong register,
- wrong memory address,
- invariant broken,
- suggested patch class.

### 9.3 Critic supervision sources

The critic can be trained from three supervision sources.

**Source A — Gold traces**  
When gold IR exists, gold traces exist as well. This provides supervision for:
- first failing step,
- first diverging state,
- branch divergence,
- operand or address mismatch.

**Source B — Differential execution**  
If both candidate and gold programs are available, define

\[
t_{\mathrm{fail}} = \min\{t : S_t \neq S_t^\star\}.
\]

This gives direct labels for failure localization.

**Source C — Assertions and invariants**  
Even without gold traces, instrument execution with checks:
- type invariants,
- bounds invariants,
- sortedness,
- conservation rules,
- task-specific loop invariants.

### 9.4 Critic loss decomposition

Let

\[
\mathcal{L}_{\mathrm{trace}} =
\mu_1 \mathcal{L}_{\mathrm{failstep}} +
\mu_2 \mathcal{L}_{\mathrm{failtype}} +
\mu_3 \mathcal{L}_{\mathrm{localization}} +
\mu_4 \mathcal{L}_{\mathrm{repairhint}}.
\]

This makes the critic a supervised subsystem rather than an informal add-on.

### 9.5 Repair loop

Loop:
1. compile program \(\pi^{(k)}\),
2. execute exactly,
3. analyze trace,
4. condition Backbone and compiler on critic summary,
5. emit \(\pi^{(k+1)}\).

### 9.6 Mathematical form

Let the repair state be

\[
r^{(k)} = C(\pi^{(k)}, \mathcal{T}^{(k)}, y^{(k)}, y^\star).
\]

Then the next program distribution is

\[
p_\theta(\pi^{(k+1)} \mid x, r^{(k)}).
\]

### 9.7 Why repair matters

A one-shot compiler is brittle. The repair loop converts execution into a closed-loop symbolic process rather than a single latent guess.

---

## 10. Training theory and objectives

### 10.1 Core losses

Use the joint objective

\[
\mathcal{L} =
\lambda_1 \mathcal{L}_{\mathrm{LM}} +
\lambda_2 \mathcal{L}_{\mathrm{IR}} +
\lambda_3 \mathcal{L}_{\mathrm{exec}} +
\lambda_4 \mathcal{L}_{\mathrm{trace}} +
\lambda_5 \mathcal{L}_{\mathrm{type}} +
\lambda_6 \mathcal{L}_{\mathrm{len}} +
\lambda_7 \mathcal{L}_{\mathrm{repair}}.
\]

### 10.2 Training without differentiating through the VM

The VM is exact and discrete. Instead of pretending it is fully differentiable:
- differentiate through the compiler and Backbone,
- supervise with execution outputs,
- use trace summaries for structured credit assignment,
- use search or policy optimization when teacher-forced IR imitation is insufficient.

### 10.3 Repair objective

For \(K\) repair rounds, define

\[
\mathcal{L}_{\mathrm{repair}} = \sum_{k=1}^{K} \gamma^{k-1} \, \ell(y^{(k)}, y^\star), \qquad 0 < \gamma \le 1.
\]

This rewards fast correction.

### 10.4 Recommended phased training schedule

To avoid fragile simultaneous optimization of all losses, use staged activation.

**Phase 1**

\[
\mathcal{L}^{(1)} = \mathcal{L}_{\mathrm{IR}} + \mathcal{L}_{\mathrm{type}} + \mathcal{L}_{\mathrm{exec}}
\]

**Phase 2**

\[
\mathcal{L}^{(2)} = \mathcal{L}^{(1)} + \mathcal{L}_{\mathrm{trace}}
\]

**Phase 3**

\[
\mathcal{L}^{(3)} = \mathcal{L}^{(2)} + \mathcal{L}_{\mathrm{repair}}
\]

**Phase 4**

Joint fine-tuning with language objectives where appropriate:

\[
\mathcal{L}^{(4)} = \mathcal{L}^{(3)} + \mathcal{L}_{\mathrm{LM}}
\]

More advanced loss balancing (e.g. uncertainty weighting or GradNorm) is deferred until after the staged baseline is stable.

### 10.5 Search and policy optimization

Teacher-forced IR imitation is expected to work for synthetic corpora with gold programs. For natural-language compilation, this may be insufficient. Therefore:
- supervised imitation is the **prototype regime**,
- search/distillation/policy optimization is the **scaling regime**.

---

## 11. Complexity and scaling analysis

Let:
- \(n\): input token length,
- \(T\): latent program length,
- \(m\): executed VM steps,
- \(N_M\): RAM size,
- \(d\): backbone hidden dimension.

### 11.1 Backbone cost

For a standard transformer backbone:

\[
O(n^2 d)
\]

or lower with linear-attention / SSM variants.

### 11.2 Compiler cost

Autoregressive IR decoding:

\[
O(T^2 d)
\]

or approximately \(O(Td)\) with efficient decoding.

### 11.3 VM execution cost

Register ops and RAM access are constant-time in the ideal RAM model:

\[
O(m).
\]

In a Python prototype, constant factors and interpreter overhead will be significant, so this should be understood as an abstract machine-model cost rather than a literal wall-clock claim.

### 11.4 Total cost

Approximate total cost:

\[
O(\mathrm{Backbone}) + O(\mathrm{Compiler}) + O(m).
\]

### 11.5 Macro-step advantage

If implicit token-level reasoning would require \(L\) steps but execution uses \(m\) VM steps, then the ideal compression factor is

\[
\text{compression} \approx \frac{L}{m}.
\]

This is the main scaling lever.

### 11.6 Checkpointing tradeoff

If trace checkpoints are stored every \(K\) steps, memory is approximately

\[
O\left(\frac{m}{K} |S|\right)
\]

with reconstruction overhead \(O(K)\).

Balancing \(Q\) random accesses gives objective

\[
\frac{m}{K}|S| + QK.
\]

Minimizing yields

\[
K^\star \approx \sqrt{\frac{m|S|}{Q}}.
\]

---

## 12. Verification strategy

### 12.1 What can be proved exactly

- VM transition semantics are exact by construction.
- Type safety can be checked statically.
- Determinism holds for fixed program and initial state.
- Turing-completeness follows from the machine model in the unbounded-memory abstraction.

### 12.2 What must be validated empirically

- compiler correctness on natural-language tasks,
- trace critic utility,
- repair convergence,
- backbone/VM interaction stability,
- macro-step efficiency gains.

### 12.3 Reference test suite

Phase-1 exact tasks:
- arithmetic expressions,
- factorial / Fibonacci,
- sorting small arrays,
- graph reachability,
- finite-state simulation,
- code trace prediction,
- spreadsheet-like dependency execution.

### 12.4 Success criteria

A prototype should demonstrate:
- exact task outputs,
- interpretable internal programs,
- explicit failure localization,
- successful repair in 1–3 rounds,
- shorter executable traces than token-level chain-of-thought.

### 12.5 Anti-shortcut verification

The system must show that it genuinely uses the VM.

**VM-ablation test**  
Disable execution and verify that exact-computation performance collapses on tasks requiring symbolic state.

**Program-randomization test**  
Corrupt emitted IR and verify that outputs degrade accordingly.

**Trace-faithfulness test**  
Check that final answers agree with VM-executed outputs from emitted programs; success with incorrect programs indicates shortcutting.

---

## 13. Reference implementation plan

### 13.1 Phase 0 — Core simulator

Deliverables:
- typed IR definition,
- exact Python VM,
- assembler / serializer,
- deterministic test harness,
- trap and timeout implementation.

### 13.2 Phase 1 — Synthetic compiler supervision

Deliverables:
- corpora of tasks with gold IR,
- autoregressive compiler model,
- exact execution training loop,
- type checker,
- compile-time rejection path.

### 13.3 Phase 2 — Trace critic

Deliverables:
- trace schema,
- failure classifiers,
- structured repair prompts,
- first-failing-step supervision,
- invariant instrumentation.

### 13.4 Phase 3 — Natural-language compilation

Deliverables:
- NL-to-IR dataset,
- backbone + compiler joint training,
- sequence/string opcode extension,
- benchmarks on algorithmic reasoning tasks.

### 13.5 Phase 4 — Iterative repair

Deliverables:
- compile–execute–repair loop,
- repair-conditioned decoding,
- evaluation on harder held-out tasks.

### 13.6 Suggested stack

- Python reference VM,
- PyTorch for backbone/compiler/critic,
- JSON or protobuf IR serialization,
- deterministic replay tooling,
- optional Rust or C++ VM for speed in later stages.

---

## 14. Roadmap, team, and compute estimates

### 14.1 Team assumption

Initial assumption:
- 1 research engineer + 1 research scientist, or
- 1 highly capable full-stack ML researcher with occasional systems support.

### 14.2 Rough time estimates

- **Phase 0:** 3–5 weeks
- **Phase 1:** 4–8 weeks
- **Phase 2:** 4–6 weeks
- **Phase 3:** 6–10 weeks
- **Phase 4:** 6–10 weeks

Total initial research cycle: approximately 5–9 months depending on data availability and whether the critic stabilizes quickly.

### 14.3 Compute estimates

**Phase 0–1**
- CPU-heavy VM work,
- 1–4 GPUs sufficient for synthetic compiler supervision.

**Phase 2**
- similar GPU scale, additional storage for trace corpora.

**Phase 3–4**
- larger GPU requirement due to natural-language backbone training,
- likely first serious compute bottleneck.

### 14.4 Milestones

**Milestone 1 — Exact VM and IR complete**
- deterministic VM,
- 32–64 opcodes,
- 32 registers,
- 64K RAM,
- stack + call support.

**Milestone 2 — Compiler learns gold IR**
- >99% exact execution on synthetic algorithm corpus,
- stable typed decoding,
- short latent traces.

**Milestone 3 — Critic localizes failures**
- identify first failing step accurately,
- classify wrong branch / wrong operand / wrong address.

**Milestone 4 — Natural language to executable IR**
- end-to-end exact performance on selected NL algorithm tasks,
- interpretable compiled traces.

**Milestone 5 — Repair loop works**
- measurable success increase from iterative repair,
- reduced brittle one-shot failures.

---

## 15. Risks and open problems

1. **Latent IR identifiability**  
   Many internal programs can solve the same task.

2. **Compiler search difficulty**  
   Good execution may require exploration beyond teacher-forced IR imitation.

3. **Neural shortcutting**  
   The backbone may try to solve tasks heuristically instead of using the VM.

4. **Repair instability**  
   Recompilation loops may oscillate rather than converge.

5. **Typed semantic objects**  
   Integrating symbolic and embedding-valued objects into one VM may complicate semantics.

6. **Benchmark mismatch**  
   Synthetic exactness may not transfer to realistic language tasks.

7. **Sparse and delayed compiler learning signal**  
   Because the VM is discrete and exact, compiler learning may depend on sparse, delayed, and non-smooth execution feedback outside the supervised-IR regime.

---

## 16. Review checklist

Use this checklist for architectural review.

### Theory
- Is exactness clearly localized in the VM?
- Is the compiler/VM separation mathematically coherent?
- Are claims about Turing-completeness confined to the executor abstraction?

### Systems
- Is the IR minimal but expressive enough?
- Are memory, stack, and control-flow semantics explicit?
- Are trap, overflow, and timeout semantics defined?
- Is determinism testable and reproducible?

### Training
- Are supervision sources available for IR, execution, and trace criticism?
- Is the non-differentiable VM handled honestly?
- Is repair training specified clearly?
- Is hyperparameter sensitivity of the multi-term loss acknowledged and mitigated?

### Evaluation
- Are exact tasks included?
- Are failures localizable?
- Is macro-step efficiency measured, not only output accuracy?
- Is anti-shortcut verification included?

### Scope discipline
- Does the document avoid claiming the backbone itself executes exactly?
- Are open problems stated plainly?

---

## 17. Appendix: core math and formulas

### A. Turing-machine simulation sketch

Encode tape cells in RAM, head location in register \(r_h\), current TM state in register \(r_q\). One transition does:
1. `LOAD` symbol under head,
2. branch on \((r_q, symbol)\),
3. `STORE` new symbol,
4. increment or decrement \(r_h\),
5. update \(r_q\),
6. continue or halt.

This simulates a Turing machine with constant-factor overhead per transition in the unbounded-memory abstraction.

### B. Joint loss

\[
\mathcal{L} =
\lambda_1 \mathcal{L}_{\mathrm{LM}} +
\lambda_2 \mathcal{L}_{\mathrm{IR}} +
\lambda_3 \mathcal{L}_{\mathrm{exec}} +
\lambda_4 \mathcal{L}_{\mathrm{trace}} +
\lambda_5 \mathcal{L}_{\mathrm{type}} +
\lambda_6 \mathcal{L}_{\mathrm{len}} +
\lambda_7 \mathcal{L}_{\mathrm{repair}}.
\]

### C. Critic loss decomposition

\[
\mathcal{L}_{\mathrm{trace}} =
\mu_1 \mathcal{L}_{\mathrm{failstep}} +
\mu_2 \mathcal{L}_{\mathrm{failtype}} +
\mu_3 \mathcal{L}_{\mathrm{localization}} +
\mu_4 \mathcal{L}_{\mathrm{repairhint}}.
\]

### D. Repair objective

\[
\mathcal{L}_{\mathrm{repair}} = \sum_{k=1}^{K} \gamma^{k-1} \, \ell(y^{(k)}, y^\star).
\]

### E. Checkpointing tradeoff

\[
\text{memory} \approx O\left(\frac{m}{K}|S|\right), \qquad \text{recompute cost} = O(K),
\]

with balancing rule

\[
K^\star \approx \sqrt{\frac{m|S|}{Q}}.
\]

---

## Closing summary

TESSERACT is not a proposal to make attention itself into an exact computer. It is a proposal to let a language model compile problems into a small exact computer, inspect execution traces, and repair its own symbolic programs. The core architectural bet is that this separation of semantic reasoning and exact execution is the missing step toward a practical Turing-complete LLM.
