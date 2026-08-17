# latent-annotation

Agent-based annotation of sparse autoencoder latents over protein language models.

An agent is given one SAE latent, pulls its top-activating sequence windows, looks
the proteins up in UniProt / InterPro / Pfam / Reactome through
[ToolUniverse](https://github.com/mims-harvard/ToolUniverse), and returns a
structured, cited annotation — including what would falsify it.

## Run it

```bash
uv sync
uv run python examples/annotate_latents.py                        # mock model, offline
uv run python examples/annotate_latents.py --show-evidence        # no model at all
```

For real annotations, copy `.env.example` to `.env`, set `HF_TOKEN`, and:

```bash
uv run python examples/annotate_latents.py --provider huggingface
uv run python examples/annotate_latents.py --provider huggingface --model Qwen/Qwen3-32B
```

The default model layer is `MockChatClient` — a scripted client that walks the same
tool trajectory a real run does (fetch evidence → look up the top accession → emit
a schema-shaped annotation) without a token or network access to the model. Live
ToolUniverse calls still happen, so a mock run exercises the whole pipeline except
the reasoning.

## Layout

```
latent_annotation/
  config.py     every tunable in one place: tool lists, model defaults, evidence windowing, agent loop limits
  evidence.py   SAE activations -> top-activating windows, firing stats (no torch/numpy)
  tools.py      ToolUniverse + local tools behind one dispatcher
  llm.py        ChatClient protocol; Qwen-3 via HuggingFace, plus the mock
  agent.py      the tool-use loop
  schema.py     annotation JSON schema, system prompt, JSON extraction/validation
examples/
  make_fixture.py       synthetic activations (real accessions, illustrative fragments)
  annotate_latents.py   end-to-end demo
notes/          background: agents, ToolUniverse, SAEs and mutation effects, roadmap, codebase map
```

`latent_annotation/__init__.py` is deliberately empty — import from the
submodules directly (e.g. `from latent_annotation.evidence import LatentEvidenceStore`)
so it is always clear which module a symbol comes from.

## Using your own SAE

The only coupling point is `LatentEvidenceStore`:

```python
from latent_annotation.evidence import LatentEvidenceStore, ProteinRecord

store = LatentEvidenceStore.from_activation_dump(
    proteins=[ProteinRecord("EGFR", sequence, accession="P00533")],
    activations={"L-0142": {"EGFR": [0.0, 0.0, 2.4, ...]}},  # per-residue, per latent
)
```

Everything downstream is unchanged.

## Notes

Design rationale and research planning live in [`notes/`](notes/README.md) — start
with [agent foundations](notes/01-agent-foundations.md) if you have not built an
agent before.
