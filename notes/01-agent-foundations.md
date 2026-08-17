# Building agents (for people who have not built one)

## An agent is a loop, not a model

The entire mechanism:

```
messages = [system, user_task]
while True:
    reply = model(messages, tools)
    if reply has no tool calls: return reply.text
    results = [run(call) for call in reply.tool_calls]
    messages += [reply, *results]
```

That is the whole thing. `latent_annotation/agent.py` is this loop with the
project's specific bookkeeping added. Everything else people call "agent
frameworks" is convenience around those seven lines. Writing it yourself is worth
it here because the failure modes are the research object — we want to know *when*
the model annotated a latent without reading its activations, not have a framework
paper over it.

The model does not execute anything. It emits a structured request ("call
`UniProt_get_function_by_accession` with `accession=P00533`"); your code runs it and
hands the text back. Every capability an agent has is a capability you wrote and
exposed.

## The three decisions that determine whether it works

**1. What goes in the tool surface.** Each tool costs context on every single
request, and models degrade when choosing between many similar options. Nine
curated tools beat ToolUniverse's 2,599. The description is the routing signal: say
*when* to call it, not just what it does. Compare the `get_latent_evidence`
description in `tools.py` ("Call this first for any latent... it is the only source
of ground-truth activation data") with a bare "returns latent evidence" — that
sentence is what makes the model call it before guessing.

**2. What the model gets to see.** Do not put all 10,000 latents in the prompt. Put
one task in the prompt and let the model *pull* evidence through a tool. This is why
`LatentEvidenceStore` is exposed as `get_latent_evidence` rather than serialized
into the system prompt: cost per latent stays flat as the SAE grows, and each
annotation gets a clean context uncontaminated by the previous latent's proteins.

**3. What the output has to look like.** A free-text label cannot be scored,
diffed across SAE runs, or checked for whether the agent did any lookups. The JSON
schema in `schema.py` is the actual contract, and it deliberately forces fields the
model would skip: per-claim source attribution, a falsification condition, rejected
hypotheses. Designing that schema is most of the work of designing the agent.

## Grounding is the point

An LLM asked "what does a latent firing on GxGxxG mean?" answers fluently from
parametric memory. That answer is unfalsifiable and unciteable. The same model with
a UniProt tool must produce an accession and a retrieved record. The difference is
not intelligence, it is auditability — and for an interpretability paper,
auditability is the deliverable.

Two guardrails in this repo enforce it:

- `require_evidence_tool` — if the model answers without ever calling
  `get_latent_evidence`, the loop pushes back once before accepting anything. An
  annotation produced without reading an activation is a hallucination regardless of
  how plausible it sounds.
- Every `evidence` entry carries `source_tool` and `source_id`, so a reviewer can
  re-run the lookup.

## Working with an open-weights model

Qwen-3 over the HuggingFace API behaves differently from a frontier model in ways
the loop has to absorb rather than assume away:

| Behaviour | Handling in this repo |
|---|---|
| Wraps JSON in a ```json fence, or adds prose | `extract_json` tries fenced, raw, and brace-slice candidates |
| Ignores `response_format` on some HF providers | Schema validated client-side by `validate_annotation` |
| Emits tool arguments as a JSON *string* | `_coerce_arguments` normalizes both shapes |
| Answers without calling tools | one nudge, then recorded as a failure |
| Returns malformed JSON | up to `max_repair_attempts` correction turns with the specific errors |

None of these are exotic; they are the standard tax on open tool-calling models.
Budget for a repair loop and treat schema validation as part of the agent, not as
downstream cleanup.

## Failure modes to watch for in this project

- **Sycophantic labelling.** The model finds *a* pattern in any set of sequences.
  Broad, high-firing-rate latents should come back `polysemantic` or
  `uninterpretable`; if they never do, the prompt is rewarding confident answers.
  The fixture in `examples/make_fixture.py` includes a deliberately meaningless
  latent for exactly this check.
- **Confirmation loops.** The model forms a hypothesis from example 1 and looks up
  only proteins that confirm it. Mitigation: the schema requires a disconfirming
  case, and the prompt asks for the site the label does *not* explain.
- **Context exhaustion.** A single UniProt entry can be tens of thousands of
  characters. `_stringify` truncates at 8k; without that, one lookup crowds out
  everything else.
- **Silent tool errors.** Tool failures return as `is_error` results the model can
  see and route around, not exceptions that abort the run.

## Where to go deeper

- Anthropic, *Building Effective Agents* (2024) — when an agent beats a fixed
  workflow. Short, and the honest answer is "less often than you'd think."
- The ReAct paper (Yao et al., 2022) — the interleaved reason/act pattern this loop
  implements.
- Toolformer (Schick et al., 2023) — background on tool use as a learned capability.
- ToolUniverse / *Democratizing Biomedical AI Agents* — the scientific-tool layer
  used here (see [02-tooluniverse.md](02-tooluniverse.md)).
