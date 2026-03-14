# TESSERACT

A compiler–executor language model architecture that separates continuous semantic reasoning from exact symbolic execution.

## Documentation

- `docs/TESSERACT_Design_Document_v0_2.md` — architecture, theory, and full design proposal
- `docs/implementation_plan.md` — phased implementation roadmap with testing requirements for every feature

## Repo layout

- `docs/` — design documents and implementation planning
- `src/tesseract/backbone/` — backbone interfaces, NL task data, and rule-based backbone baselines
- `src/tesseract/compiler/` — latent compiler interfaces and NL-conditioned compiler path
- `src/tesseract/vm/` — exact virtual machine and IR
- `src/tesseract/critic/` — trace critic and repair loop controller
- `src/tesseract/evaluation/` — reproducibility, benchmark, and reporting helpers
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

The repository now includes a tested VM/IR core, validation/assembly/serialization tooling, a synthetic compiler baseline, a rule-based NL backbone path, a deterministic repair loop scaffold, and reproducible benchmark/reporting helpers. The current compiler baseline is still a count-based autoregressive placeholder rather than a neural decoder, so the next major compiler milestone is upgrading that decoder while preserving the now-wired NL, repair, and evaluation plumbing.
