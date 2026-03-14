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

## Status

Early scaffold repository with a detailed implementation plan. Current code includes a minimal VM prototype and initial tests; the next milestone is Phase 0 bootstrap and test harness hardening, followed by the exact VM/IR core.
