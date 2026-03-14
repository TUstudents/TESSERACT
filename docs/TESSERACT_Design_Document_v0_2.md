# TESSERACT: A Compiler–Executor Language Model
## Complete Design Proposal and Implementation Plan

**Version:** 0.2  
**Date:** March 2026  
**Status:** Revised review draft

## Summary

TESSERACT separates continuous semantic reasoning from exact symbolic execution. A neural backbone compiles tasks into a typed latent IR, an exact VM executes that IR with persistent memory and explicit control flow, and a trace critic diagnoses failures and guides repair.

## Core modules

- Semantic Backbone
- Latent Compiler
- Exact Virtual Machine
- Trace Critic

## Core claims

- Exactness is localized in VM semantics.
- Learning is localized in the backbone, compiler, and critic.
- Repair is performed by a compile–execute–critique–repair loop.

## Full document

Use the separately provided full design document if needed; this repo copy is a concise version for initial scaffolding.
