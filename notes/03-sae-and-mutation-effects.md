# SAE latents, annotation, and mutation effect prediction

## The setup

A protein language model (ESM-2, ESM-C, ProtT5) produces a residue-level
representation whose dimensions are dense and polysemantic — no single dimension
means one thing. A sparse autoencoder trained on those activations,

```
z = ReLU(W_enc (h - b_dec) + b_enc)     # sparse, overcomplete: dim(z) >> dim(h)
ĥ = W_dec z + b_dec
```

gives a much larger set of latents where each fires on a small fraction of
residues. The bet is that this sparse basis is closer to the model's actual
features than the raw dimensions are.

That bet is only cashable if you can say what each latent means. With 10k–100k
latents, hand-inspection does not scale — which is the entire reason for an
annotation agent.

## Why "annotation" is harder than it sounds

The naive pipeline — show a model the top-activating sequences, ask for a label — is
the one that produces impressive demos and unusable results. Three specific problems:

1. **The model will always find a pattern.** Given any set of protein fragments, an
   LLM produces a fluent explanation. A label with no falsification condition is not
   a finding. Hence `disconfirming_evidence` as a required schema field.

2. **Top-k activations are a biased sample.** They over-represent whatever the
   corpus over-represents. A latent that "fires on kinases" may be tracking the most
   common fold in the dataset rather than kinase-ness. Hence `rejected_hypotheses`,
   and the prompt's instruction to look for the site the label does not explain.

3. **Protein-level facts do not license residue-level claims.** "These are all
   kinases" is guilt by association. "These peaks fall inside the annotated
   ATP-binding region at positions X, Y, Z" is a positional claim, checkable against
   the peak index. `UniProt_get_features_by_accession` and
   `InterPro_get_residue_annotations` are what let the agent make the second kind.

The literature precedent is Bills et al. (2023), *Language models can explain
neurons in language models* — explain, then **simulate** the explanation and score
it against real activations. The simulation step is what makes an explanation
falsifiable, and it is the main thing missing from this repo today (see
[04-roadmap.md](04-roadmap.md)).

## Mutation effect prediction: why latents are interesting here

Zero-shot variant effect prediction from a PLM is usually the masked-marginal
score,

```
score(wt -> mut at position i) = log p(mut | context) - log p(wt | context)
```

evaluated against deep mutational scanning data (ProteinGym, MaveDB). It works, and
it is a black box: you get a number, not a reason.

SAE latents offer a decomposition. If you can attribute the score to a handful of
interpretable latents, you get a mechanistic account of *why* a substitution is
predicted damaging — "it ablates the latent that marks the catalytic-site
configuration" rather than "the log-odds went down."

Three research directions this repo could support:

**A. Latents as an explanation layer.** Compute the variant score as usual, then
attribute it across latents that change between wild-type and mutant. The annotation
agent supplies the human-readable name for each contributing latent. This is the
lowest-risk direction: the prediction is unchanged, the interpretation is new.

**B. Latents as features.** Fit a lightweight model on latent activations at the
mutated position, benchmarked against the masked-marginal baseline on ProteinGym.
Interesting if it wins; interesting in a different way if a sparse, interpretable
feature set matches a dense baseline.

**C. Annotation-guided priors.** If a latent is annotated `active_or_binding_site`
with high confidence, substitutions that ablate it should be predicted more
damaging. This is the direction the `mutation_effect` field in the annotation schema
is built for — it forces the agent to state a testable expectation
(`predicted_sensitivity`) at annotation time, which can then be scored against DMS
data it never saw.

Direction C is where the agent stops being a labelling convenience and starts
producing hypotheses. It is also the one that can be evaluated cheaply: correlate
`predicted_sensitivity` against measured DMS effect at the residues each latent
fires on.

## Agents for mutation effect specifically

`MUTATION_EFFECT_TOOLS` in `config.py` is the second tool set, for an agent whose
question is "is this latent's territory mutation-sensitive?":

| Tool | Role |
|---|---|
| `UniProt_get_disease_variants_by_accession` | curated pathogenic variants — do they land on the firing sites |
| `MaveDB_search_score_sets` / `MaveDB_get_variant_scores` | experimental DMS measurements for these proteins |
| `MaveDB_get_effect_matrix` | one-shot DMS loader — parses HGVS to (ref, position, alt) and filters to single substitutions, which is exactly the shape a per-residue evaluation needs |
| `ClinVar_search_variants` | clinical significance of substitutions at the site |
| `gnomad_get_constraint` | population-level depletion of variation |

The clean experimental design: withhold the DMS data at annotation time, let the
agent predict sensitivity from structure and function alone, then score it. If the
agent can look up MaveDB while annotating, you have measured retrieval, not
prediction. Two tool sets, two agents, evaluated separately.

## Caveats worth writing down before running experiments

- **SAE latents are not stable across training runs.** Two SAEs on the same
  activations give different bases. Any claim about "the catalytic site latent"
  needs a cross-run reproducibility check, and annotations should be diffable —
  another argument for the structured schema.
- **Feature splitting.** Wider SAEs split one concept into several narrower ones.
  The same annotation may legitimately apply to multiple latents at higher width.
- **Reconstruction ≠ interpretability.** A better L0/reconstruction tradeoff does not
  imply more interpretable latents; measure them separately.
- **Annotation is an LLM output, not ground truth.** Report inter-run agreement and
  a human-audited subset before treating annotations as labels in a downstream claim.

## References

- Bills et al. (2023), *Language models can explain neurons in language models*
- Cunningham et al. (2023), *Sparse autoencoders find highly interpretable features*
- Bricken et al. (2023), *Towards monosemanticity* (Anthropic)
- Simon & Zou (2024), *InterPLM: discovering interpretable features in PLMs via SAEs*
- Meier et al. (2021), *Language models enable zero-shot prediction of mutation effects*
- Notin et al. (2023), *ProteinGym*
