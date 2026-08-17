# Roadmap — what to do next in this repo

Ordered by what unblocks the most. Everything here is speculative planning, not
committed work.

## Now: replace the fixture with real SAE output

The only coupling point is `LatentEvidenceStore`. Dump per-residue activations for
your SAE as `{latent_id: {protein_id: [act, ...]}}` and build with
`from_activation_dump`; nothing downstream changes.

Two things to decide while doing it:

- **Corpus bias.** Whatever the proteins over-represent, the annotations will
  over-represent. Record the composition of the corpus alongside the annotations.
- **Example selection.** Top-k by max activation (what `build_latent_evidence`
  does) biases toward the extreme tail. Stratified sampling across activation
  deciles gives the agent a better picture of what the latent does at typical
  firing strength — worth adding as an option.

## Next: close the evaluation loop

The annotations are currently unfalsified. The Bills et al. protocol adapted to
proteins:

1. Agent produces a label from the top-k windows.
2. A **second** call — label only, no activations — predicts activation on held-out
   sequences.
3. Correlate predicted against actual.

That correlation is an interpretability score per latent, and it is what turns this
from a labelling tool into a measurement. It also gives an honest baseline: score a
random-label control to see how much of the correlation is the protocol rather than
the annotation.

Cheaper checks worth having first:

- **Self-consistency.** Annotate each latent 3× at temperature > 0; disagreement
  rate is a confidence signal that costs nothing to compute.
- **Control latents.** Feed randomly-permuted activations. Anything not labelled
  `uninterpretable` is a calibration failure.
- **Human audit.** 20–30 latents graded by hand, as the ceiling estimate.

## Then: the mutation-effect agent

Build a second configuration on `MUTATION_EFFECT_TOOLS` with its own schema and
system prompt. Keep it separate from the annotation agent — mixing them lets the
agent look up MaveDB and call it a prediction.

Experimental order:
1. Annotate latents (no DMS access at annotation time).
2. Predict `mutation_effect.predicted_sensitivity` per latent.
3. Score against ProteinGym / MaveDB at the residues each latent fires on.

Direction C from [03-sae-and-mutation-effects.md](03-sae-and-mutation-effects.md) is
the one that produces a claim rather than a demo.

## Engineering, as it becomes the bottleneck

- **Caching.** ToolUniverse `use_cache=True` + `dump_cache()`. Annotating at scale
  hits the same accessions repeatedly; without a cache this is thousands of
  redundant EBI requests.
- **Concurrency.** The example script annotates sequentially in a plain loop
  (`for latent_id in ...: agent.annotate(...)`). Latents are independent, so a
  thread pool over `annotate` is the obvious parallelization — bounded by the
  slowest tool endpoint, not the model.
- **Checkpointing.** Write JSONL as you go (the example already does) and skip
  latents already present, so a 10k-latent run survives interruption.
- **Cost/latency ledger.** Record tokens and wall time per latent from day one; it
  determines whether the full sweep is affordable before you start it.

## Model-layer questions still open

- **Which Qwen-3 variant.** The default is the large instruct model. Smaller
  variants are worth benchmarking on the same latents — annotation may not need the
  big model, and if it does not, the full sweep gets much cheaper. `--model` makes
  this a one-flag comparison.
- **Tool-calling reliability.** Track how often the model skips
  `get_latent_evidence` (the loop already nudges and records it) and how often JSON
  needs repair. If either rate is high, the fix is usually the prompt or a smaller
  tool surface, not a bigger model.
- **Provider variance.** HF routes to whichever provider serves the model; support
  for `response_format` and tool calling differs between them. Pin `provider=` in
  `HuggingFaceChatClient` once you find one that behaves, so runs stay comparable.
- **A frontier-model reference run.** Even a few dozen latents annotated by a
  frontier model gives an upper bound to measure the open-weights setup against.
  The `ChatClient` protocol exists so that is a new class, not a rewrite.

## Open questions worth resolving before scaling up

- Does annotation quality depend more on the tool surface or on the model? Ablate
  by running with `--no-tooluniverse` (local evidence only) and comparing.
- How stable are annotations across two SAEs trained on the same activations? If
  unstable, individual latent claims are not publishable and the unit of analysis
  has to be something coarser.
- What fraction of latents are honestly uninterpretable? A pipeline that never says
  so is miscalibrated, and that fraction is itself a result.
