# TESSERACT

A compiler–executor language model architecture that separates continuous semantic reasoning from exact symbolic execution.

## Documentation

- `docs/TESSERACT_Design_Document_v0_2.md` — architecture, theory, and full design proposal
- `docs/implementation_plan.md` — phased implementation roadmap with testing requirements for every feature

## Repo layout

- `docs/` — design documents and implementation planning
- `src/tesseract/backbone/` — semantic backbone interfaces
- `src/tesseract/compiler/` — latent compiler interfaces
- `src/tesseract/vm/` — exact virtual machine and IR
- `src/tesseract/critic/` — trace critic and repair loop
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

The repository now includes a tested VM/IR core, validation/assembly/serialization tooling, a synthetic compiler baseline, and a trace critic scaffold. The current compiler baseline is a count-based autoregressive placeholder rather than a neural decoder, so the next major compiler milestone is improving that autoregressive path and connecting it to later repair and NL compilation phases.
