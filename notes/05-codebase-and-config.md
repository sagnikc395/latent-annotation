# Codebase map, the central config, and the bloat passes

This is the reference document for the repo as it actually is: what each file is
for, how the modules depend on each other, where every tunable lives, what two
code-bloat passes removed and why, and what looks removable but is not. Read it
before editing code.

## 1. Module map

```
latent_annotation/
  config.py     the single source of configuration — section 3
  evidence.py   SAE activations -> top-activating windows + firing stats (no torch/numpy)
  tools.py      ToolUniverse + local tools behind one dispatcher (ToolBox)
  llm.py        ChatClient protocol; HuggingFaceChatClient (Qwen-3) + MockChatClient
  agent.py      the per-latent tool-use loop (LatentAnnotationAgent)
  schema.py     annotation JSON schema, system prompt, JSON extraction/validation
  __init__.py   deliberately empty — section 6
examples/
  make_fixture.py       generates examples/fixtures/latents.json (synthetic activations)
  annotate_latents.py   end-to-end demo: evidence -> agent -> annotations.jsonl
main.py                 ToolUniverse discovery helper (search the full catalog)
notes/                  these documents
```

Roughly 1,000 lines of source across six modules plus two scripts. Every module
has one job, and the split is chosen so the expensive parts (a model, a network
round-trip, an SAE checkpoint) are each isolated behind one seam:

| Seam | Module | What it lets you do without the expensive thing |
|---|---|---|
| `ChatClient` protocol | `llm.py` | Run the whole loop offline with `MockChatClient` |
| `LatentEvidenceStore` | `evidence.py` | Build evidence from any activation dump, no SAE code in this repo |
| `ToolBox` | `tools.py` | Run with `--no-tooluniverse` when the databases are down or slow |

## 2. Data flow of one annotation

```
fixture (proteins + activations)
  -> evidence.LatentEvidenceStore.from_json        build windows + firing stats
  -> tools.evidence_tools(store)                   get_latent_evidence / list_latents
  -> tools.ToolBox                                 local tools + curated ToolUniverse tools
  -> agent.LatentAnnotationAgent.annotate(latent)  loop over llm.ChatClient
  -> schema.validate_annotation                    enforce ANNOTATION_SCHEMA client-side
  -> annotations.jsonl                             annotation + tool trace + usage
```

The loop inside `annotate` is a small state machine, and its three non-obvious
branches are all failure handling rather than happy path:

1. **Model wants tools** → run them, append results, continue.
2. **Model answers without ever calling `get_latent_evidence`** → nudge once
   (`NUDGE_MISSING_EVIDENCE`), because that annotation was invented, not observed.
3. **Reply is not parseable JSON, or fails the schema** → send the specific errors
   back and ask for a correction, up to `max_repair_attempts`.

Module dependency graph — `config` is a leaf that everything imports, and nothing
imports back into it:

```
config.py            (imports nothing from the package)
   ^-- evidence.py, tools.py, llm.py, agent.py, examples/, main.py
evidence.py  <- tools.py      (LatentEvidenceStore type hint + evidence_tools)
llm.py       <- agent.py      (ChatClient, ChatResponse)
schema.py    <- agent.py      (ANNOTATION_SCHEMA, SYSTEM_PROMPT, extract/validate)
tools.py     <- agent.py      (ToolBox)
```

There are no cycles and no module imports the package root.

## 3. The central config (`config.py`)

Every tunable in the project lives in `latent_annotation/config.py`, grouped by
the module that consumes it:

| Group | Constants | Consumed by |
|---|---|---|
| Paths | `REPO_ROOT`, `FIXTURE_PATH`, `OUTPUT_PATH`, `ENV_FILE` | both example scripts, `llm.load_client` |
| Env var names | `ENV_LLM_PROVIDER`, `ENV_LLM_MODEL`, `ENV_HF_TOKEN`, `ENV_HF_API_KEY` | `llm.load_client` |
| Model layer | `DEFAULT_MODEL`, `DEFAULT_HF_PROVIDER`, `TEMPERATURE`, `MAX_TOKENS`, `TIMEOUT_SECONDS`, `USE_RESPONSE_FORMAT`, `MOCK_LOOKUP_TOOL` | `HuggingFaceChatClient`, `MockChatClient` |
| Agent loop | `AgentConfig` (`max_turns`, `require_evidence_tool`, `max_repair_attempts`) | `LatentAnnotationAgent` |
| Evidence | `TOP_K_EXAMPLES`, `WINDOW_SIZE`, `HIGHLIGHT_SPAN`, `ACTIVE_THRESHOLD` | `build_latent_evidence` |
| Tool surface | `PROTEIN_CONTEXT_TOOLS`, `MUTATION_EFFECT_TOOLS`, `TOOL_RESULT_LIMIT` | `ToolBox` assembly, `_stringify`, example, `main.py` |

Two kinds of settings, two ways to change them:

- **Code settings** (tool lists, windowing, loop limits): edit the value in
  `config.py`. Modules import the constant and use it as a *default*, so a
  per-call override is still possible — `build_latent_evidence(..., top_k=4)`
  for a quick experiment — without forking the project default.
- **Environment settings** (provider, model id, token): set the variable in
  `.env` (see `.env.example`). The config constants are the *names* of those
  variables plus the fallback defaults; the runtime values are read by
  `llm.load_client`.

What is deliberately **not** in `config.py`:

- **Prompts.** `SYSTEM_PROMPT` lives in `schema.py` next to the schema it
  describes, because the two have to change together — the prompt's "Output
  format" paragraph is a restatement of `ANNOTATION_SCHEMA`. `NUDGE_MISSING_EVIDENCE`
  lives in `agent.py` next to the branch that sends it. These are content, not
  configuration; splitting them from their call site makes both harder to read.
- **The schema enums** (`CONCEPT_TYPES`, `CONFIDENCE_LEVELS`,
  `MUTATION_SENSITIVITY`). They are the output contract, not knobs.
- **CLI defaults** (`--limit 25` in `main.py`). Argparse already documents them.

`config.py` must stay a leaf: it imports nothing from the package, so anything
can import it without a cycle. Don't put logic in it.

## 4. What the bloat passes removed

Two passes. Each removal was checked by tracing every reference across code,
notes, and README; nothing removed was reachable from a call site.

### Pass 1 — consolidation

| Removed | From | Why |
|---|---|---|
| `ChatResponse.finish_reason` | `llm.py` | Written by both clients, read by nobody — the loop keys off `wants_tools` and `text`. |
| `LatentAnnotationAgent.annotate_many` | `agent.py` | No caller, and the example script already inlines the same sequential loop — dead *and* duplicated. |
| `LatentEvidenceStore.__contains__` | `evidence.py` | Magic method never invoked; lookups go through `get`. |
| Package-root re-exports (`from latent_annotation import ...`) | `main.py`, `examples/`, README | `__init__.py` is empty, so these raised `ImportError` — `main.py` and the demo were both broken. |
| Module-local `DEFAULT_MODEL` and friends | `llm.py`, `tools.py`, `evidence.py`, `agent.py` | Moved to `config.py`, so each setting lives in exactly one file. |

Pass 1 collected settings that had been scattered across four files: the model
defaults and env-var strings in `llm.py`; `AgentConfig`'s defaults in `agent.py`;
`top_k` / `window` / `highlight_span` / `active_threshold` in `evidence.py`; the
two tool tuples and the truncation limit in `tools.py`; and the fixture/output
paths in `examples/annotate_latents.py`.

### Pass 2 — dead state and residual duplication

| Removed | From | Why |
|---|---|---|
| `AnnotationResult.ok` | `agent.py` | Property with zero readers. The demo computes its own pass/fail from `annotation is None` and `schema_errors`, so `ok` was a second, unused definition of "did this work". |
| `AnnotationResult.transcript` | `agent.py` | Assigned at all three exit paths, read by nobody. It also duplicated the part that matters: `tool_calls` already records the trace, and that is what `annotations.jsonl` persists. Holding the full message list per result was pure memory overhead. |
| `MockChatClient.calls` | `llm.py` | Counter incremented every turn, never read. |
| `LatentExample.window_sequence` | `evidence.py` | `highlighted` is the same string with two brackets inserted. Both were serialized into the evidence bundle, so every top-activating window was sent to the model twice — roughly doubling the payload of the one tool the agent is required to call. |
| `AgentConfig` imported via `agent.py` | `examples/annotate_latents.py` | Re-export indirection: `AgentConfig` is defined in `config.py`, and importing it through `agent.py` contradicted the repo's own "name the defining module" convention. |
| Explicit `config=AgentConfig()` argument | `examples/annotate_latents.py` | Restated the constructor default. |
| Duplicate fixture path (`OUT = ...`) | `examples/make_fixture.py` | Recomputed `config.FIXTURE_PATH` by hand. The generator and the reader can now never disagree about where the fixture is. |
| Hardcoded `"UniProt_get_function_by_accession"` | `llm.py` | The mock's lookup tool is now `config.MOCK_LOOKUP_TOOL`, because it is only valid if it is also in `PROTEIN_CONTEXT_TOOLS` — two values that must agree belong in one file. |

Verification for pass 2: `ruff check --select F,ARG,ERA,SIM,B` is clean apart
from one intentional `ARG002` (below), and the full pipeline was re-run before
and after — `--show-evidence` with the live ToolUniverse surface (11 tools) and
the mock end-to-end run (3/3 latents clean, no schema errors).

## 5. Kept deliberately — things that look dead but are not

- **`MUTATION_EFFECT_TOOLS`** is read only by `main.py`, for display. It is the
  reserved tool surface for the planned mutation-effect agent (notes 03 and 04),
  and having it written down is the point.
- **`MockChatClient.complete`'s unused `response_schema` parameter** (the one
  remaining ruff `ARG002`). Required to satisfy the `ChatClient` protocol; the
  mock has no use for a schema it is hardcoded to match.
- **`LatentExample.peak_index` / `window_start` / `description`** are never read
  by our code, but they are serialized into the evidence bundle the model reads.
  Position in the chain is exactly what distinguishes a `positional_or_terminal`
  latent from a `sequence_motif` one.
- **`use_response_format`, `timeout`, `temperature`, `max_tokens` constructor
  params** are only ever exercised with their config defaults today. They are the
  override points for the benchmark sweeps the roadmap calls for, and
  `use_response_format` in particular is a kill switch for HF providers that
  mishandle `response_format`.
- **`ToolBox.names()`** has a single caller (`--show-evidence`), which is the
  only way to see the tool surface without spending a model call on it.
- **`list_latents`** is never called in the fixture runs because the demo passes
  latent ids explicitly, but it is the tool an agent needs when it is pointed at
  a store it did not choose.

## 6. Conventions

- **`__init__.py` stays empty.** No re-exports, no side effects. Import from the
  defining submodule: `from latent_annotation.evidence import LatentEvidenceStore`.
  This keeps the dependency graph visible at the import site and avoids a second,
  package-level API surface that has to be kept in sync. It also means there is
  exactly one place a symbol comes from — which is why pass 2 removed the
  `AgentConfig`-through-`agent.py` import even though it worked.
- **Config is a leaf.** See section 3.
- **Defaults come from config, overrides are per call.** A signature names its
  config default (`top_k: int = config.TOP_K_EXAMPLES`); callers who need
  something else pass it explicitly.
- **No heavy dependencies in the core.** `evidence.py` is plain lists and JSON,
  so the evidence layer runs and tests without torch or numpy. `tooluniverse`
  and `huggingface_hub` are imported lazily, so the rest of the package imports
  in an environment that has neither.
- **Tool errors are results, not exceptions.** `ToolBox.run` returns
  `(text, is_error)` so a failed accession lookup makes the agent try another
  accession instead of killing the run.
- **The agent loop is explicit, not a framework.** Tool traces are research
  artifacts, and the loop's failure modes — ungrounded answers, malformed JSON —
  are handled inline so they stay visible rather than being swallowed by a
  library.
- **Two values that must agree live in one file.** The rule behind moving
  `MOCK_LOOKUP_TOOL` and `FIXTURE_PATH` into config.

## 7. How to extend

- **Add a ToolUniverse tool to the agent surface:** add its name to
  `PROTEIN_CONTEXT_TOOLS` (or `MUTATION_EFFECT_TOOLS`) in `config.py`. Before you
  do, ask which annotation schema field it makes more accurate — if none, it is
  context you pay for on every request. Use `uv run python main.py <pattern>` to
  discover names and `--spec <name>` for the full spec. (The catalog is ~2,600
  tools and grows with each ToolUniverse release, so trust `main.py`'s printed
  count over any number written in these notes.)
- **Swap the model:** `LLM_PROVIDER` / `LLM_MODEL` in `.env`, or add a
  `ChatClient` implementation and wire it into `llm.load_client` — the agent
  never touches a vendor SDK.
- **Change evidence presentation:** window size, example count, highlight span,
  firing threshold are all in `config.py`, with no call sites to chase.
- **Add a second agent** (e.g. the mutation-effect agent): a tool tuple, a
  schema, and a system prompt, assembled the way `examples/annotate_latents.py`
  assembles the annotation agent. The loop in `agent.py` is reusable as-is if the
  new agent has the same shape — evidence first, then databases, then structured
  output.
- **Plug in your own SAE:** the only coupling point is
  `LatentEvidenceStore.from_activation_dump`, which takes
  `{latent_id: {protein_id: [per-residue activation, ...]}}` plus a list of
  `ProteinRecord`. Nothing downstream changes.

## 8. Where the remaining risk is

Not bloat, but worth knowing while editing:

- **`MockChatClient` recovers state by re-reading the message list** — including
  a regex over the user turn to find the latent id (`_requested_latent`). It is
  coupled to the exact wording of the task string in `agent.annotate`. Change
  that string and the mock silently annotates `"unknown"`.
- **`TOOL_RESULT_LIMIT` truncates mid-JSON.** An 8,000-character cut of a UniProt
  record is not valid JSON, and the model sees the fragment. That is deliberate
  (better than blowing the context window) but it means a tool that returns one
  huge blob is effectively unusable — prefer the narrow accessor tools.
- **Nothing is tested.** There is no test suite; the mock run is the smoke test.
  `evidence.py` is dependency-free specifically so it can be unit-tested first,
  and that is the highest-value place to start (see notes 04).
