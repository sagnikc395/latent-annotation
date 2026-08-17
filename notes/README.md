# Notes

Background and planning for agent-based annotation of SAE latents. Written to be
read in order, but each stands alone.

| | |
|---|---|
| [01 — Agent foundations](01-agent-foundations.md) | What an agent actually is, the three decisions that matter, and the failure modes specific to this project. Start here if you have not built one. |
| [02 — ToolUniverse](02-tooluniverse.md) | What to do with the 2,599 tools: which to expose, which to avoid, and the operational cost of live database calls. |
| [03 — SAE latents and mutation effects](03-sae-and-mutation-effects.md) | Why annotation is harder than "show the model the top activations", and three ways latents connect to variant effect prediction. |
| [04 — Roadmap](04-roadmap.md) | Concrete next steps, ordered by what unblocks the most. |
| [05 — Codebase map and config](05-codebase-and-config.md) | Where everything lives, what the central `config.py` controls, what the two bloat passes removed and why, what only looks dead, the repo's conventions, and where the remaining risk is. Start here before editing code. |

The code these notes describe is in `latent_annotation/`, with a runnable demo in
`examples/`. See the top-level [README](../README.md) for how to run it.
