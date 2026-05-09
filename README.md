# TESSERACT

A typed execution coprocessor for exact language-model computation.

TESSERACT separates probabilistic language understanding from deterministic execution. The language-model side compiles intent into a typed intermediate representation (IR); the VM validates and executes that IR exactly; the critic and repair loop turn traps, trace mismatches, and output failures into structured feedback for another dispatch attempt.

## Documentation

- `docs/TESSERACT_Design_Document_v0_2.md` — architecture, theory, and full design proposal
- `docs/implementation_plan.md` — phased implementation roadmap with testing requirements for every feature
- `docs/implementation_status.md` — current implementation status and remaining research gaps
- `docs/next_steps_implementation_plan.md` — concrete numbered follow-on plan for the next development cycle

## Repo layout

- `docs/` — design documents and implementation planning
- `src/tesseract/backbone/` — backbone interfaces, NL task data, and rule-based / learned backbone baselines
- `src/tesseract/compiler/` — typed dispatch compiler interfaces and NL-conditioned compiler path
- `src/tesseract/vm/` — deterministic execution coprocessor VM and IR
- `src/tesseract/critic/` — oracle/learned critics and model-driven repair loop tooling
- `src/tesseract/evaluation/` — reproducibility, benchmark, experiment, and reporting helpers
- `tests/` — unit and integration tests

## Development

### Install

```bash
uv sync --group dev
```

### Run tests

```bash
uv run pytest -q
```

### Lint and type-check

```bash
uv run ruff check .
uv run mypy
```

## VM semantics notes

Current reference-VM policy:

- unread registers are treated as zero-initialized
- unread memory addresses are treated as zero-initialized
- this is an explicit implementation choice for the current prototype and is covered by tests

## Status

The repository now includes a tested VM/IR core with expanded type coverage (`checked_i64`, `f32`, and prototype `addr` tagging), validation/assembly/serialization tooling, a synthetic compiler stack with a small neural autoregressive decoder plus a retained count-based baseline for comparison, both rule-based and learned NL backbone paths, broader task coverage spanning arithmetic/control-flow/loop/memory families, differential and learned critic paths, model-driven repair support with held-out repair benchmarks, and a richer research-oriented evaluation/reporting harness with experiment manifests, anti-shortcut checks, critic-localization benchmarking, and macro-step summaries.

The current prototype is strongest as a deterministic coprocessor substrate and evaluation scaffold. The next major milestones are deeper learned-model quality, broader task domains such as sequences and strings, and stronger evidence that the model-side compiler can reliably dispatch useful exact computations through the typed VM boundary.
