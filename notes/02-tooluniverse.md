# What to do with ToolUniverse here

## What it is

ToolUniverse packages ~2,600 scientific database endpoints behind one uniform
interface: load once, get OpenAI-shaped tool specs, call by name with a dict. The
existing `main.py` in this repo already does the two operations that matter —
`tu.load_tools()` and `tu.tool_specification(name, format="openai")`. That is the
groundwork; the rest is deciding *which* tools and *what for*.

```python
tu = ToolUniverse(); tu.load_tools()
tu.find_tools_by_pattern("UniProt")          # discovery
tu.tool_specification(name, format="openai") # spec -> straight into the tools array
tu.run({"name": name, "arguments": {...}})   # execution
```

Because the specs are already OpenAI-shaped, they drop into a Qwen-3 `tools`
array with only a `{"type": "function", "function": ...}` wrapper —
`ToolBox._as_function` in `tools.py`.

## The load-everything trap

`load_tools()` gives you 2,599 tools. Passing all of them to a model is the single
most common mistake with ToolUniverse:

- Their schemas would be roughly a megabyte of context on *every* request.
- Selection accuracy collapses — dozens of tools have near-identical descriptions
  (there are 43 UniProt tools and 78 matching "Ensembl").
- Latency and cost scale with the tool list, per turn, forever.

So: load everything for *discovery* during development, expose a curated handful at
*runtime*. `PROTEIN_CONTEXT_TOOLS` in `config.py` is nine tools, chosen because each
answers a distinct question about a latent's top-activating proteins. If a latent
needs a tool outside that set, that is a signal to write a second agent
configuration, not to widen the default list.

For genuinely large tool sets, the escalation path is retrieval over tools —
ToolUniverse ships `tool_finder` and `find_tools_by_pattern` — where a first pass
selects ~10 relevant tools and only those are exposed. Worth it above roughly 30
tools; premature below that.

## Tool sets that map to this project's questions

**Interpreting a latent** (`PROTEIN_CONTEXT_TOOLS`) — given the accessions behind the
top-activating windows, what do the annotated records say?

| Tool | Answers |
|---|---|
| `UniProt_get_recommended_name_by_accession` | what protein is this |
| `UniProt_get_function_by_accession` | what does it do |
| `UniProt_get_features_by_accession` | do the peak positions fall in an annotated feature |
| `UniProt_get_subcellular_location_by_accession` | is the latent tracking localization |
| `UniProt_get_ptm_processing_by_accession` | is it a modification site |
| `InterPro_get_entries_for_protein` | which domains |
| `InterPro_get_protein_domain_architecture` | domain order and boundaries |
| `Pfam_get_protein_annotations` | family assignment |
| `Reactome_map_uniprot_to_pathways` | pathway context |

`UniProt_get_features_by_accession` is the load-bearing one. It is what turns "this
latent fires on kinases" into "this latent fires on residues inside the annotated
ATP-binding region of these kinases" — a positional claim that can be checked
against the peak index, rather than a guilt-by-association claim about the protein.

**Mutation effect** (`MUTATION_EFFECT_TOOLS`) — see
[03-sae-and-mutation-effects.md](03-sae-and-mutation-effects.md):
`UniProt_get_disease_variants_by_accession`, `MaveDB_search_score_sets`,
`MaveDB_get_variant_scores`, `ClinVar_search_variants`, `gnomad_get_constraint`.

**Discovered but unused so far**, worth knowing about: `InterPro_get_residue_annotations`
(residue-level, the natural next step for positional claims), `alphafold_get_annotations`
(structural context for the peak residue), `InterProScan_scan_sequence` (annotate a
sequence with no accession — relevant if the SAE corpus includes unannotated
proteins), `EBI_msa_align` (conservation at the firing site).

## Operational notes

- **Tool calls are live network requests.** Annotating 1,000 latents × ~6 lookups
  is 6,000 API calls to EBI/UniProt. Enable ToolUniverse's cache (`use_cache=True`
  on `run`, `dump_cache()` to persist) and expect to rate-limit.
- **Results are large.** One UniProt entry can exceed the useful context budget on
  its own; `_stringify` truncates at 8k characters. If truncation is cutting the
  informative part, prefer a narrower tool over a bigger limit.
- **Failures are data.** A dead accession or a 500 comes back as an `is_error` tool
  result so the model can try another accession. Silent retries would hide a real
  signal: which parts of the annotation corpus are unreachable.
- **Keep the tool list ordered and stable.** Reordering changes the prompt prefix
  and defeats any provider-side caching; it also makes runs non-comparable.

## Extending

Adding a tool is one line in the relevant tuple in `config.py`. Before adding one,
answer: which field of the annotation schema does this tool make more accurate? If
none, it is context you are paying for on every request and not using.
