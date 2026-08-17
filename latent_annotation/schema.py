"""The output contract for a latent annotation.

The schema is the interesting part of this file. An annotation that is just a
free-text label is unusable downstream: you cannot score it, diff it across SAE
training runs, or check whether the agent actually looked anything up. So the
schema forces three things the model would otherwise skip:

* every claim carries the tool and record id it came from (`evidence`),
* the agent must state what would falsify the label (`disconfirming_evidence`),
* the agent must name at least one rival hypothesis it rejected.

Open-weights models honour a JSON schema unevenly — some HuggingFace providers
support ``response_format``, some ignore it, and Qwen likes to wrap JSON in a code
fence. So the schema is enforced on our side too: ``extract_json`` recovers the
object from whatever the model actually emitted, and ``validate_annotation``
checks it before anything downstream sees it.
"""

from __future__ import annotations

import json
import re
from typing import Any

CONCEPT_TYPES = [
    "structural_motif",
    "sequence_motif",
    "domain_family",
    "active_or_binding_site",
    "post_translational_modification",
    "subcellular_localization",
    "taxonomic_or_phylogenetic",
    "biophysical_property",
    "disease_or_variant_associated",
    "positional_or_terminal",
    "polysemantic",
    "uninterpretable",
]

CONFIDENCE_LEVELS = ["high", "medium", "low"]

MUTATION_SENSITIVITY = ["high", "moderate", "low", "unknown"]

ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "latent_id": {
            "type": "string",
            "description": "The latent this annotation describes.",
        },
        "label": {
            "type": "string",
            "description": "Short human-readable name, under ~8 words, e.g. 'zinc-finger C2H2 coordinating residues'.",
        },
        "description": {
            "type": "string",
            "description": "Two to four sentences stating what the latent fires on and why the evidence supports that.",
        },
        "concept_type": {"type": "string", "enum": CONCEPT_TYPES},
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "evidence": {
            "type": "array",
            "description": "Each claim tied to the tool call and record that supports it. At least one entry must come from a database tool, not just the activation windows.",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source_tool": {
                        "type": "string",
                        "description": "Name of the tool that produced this evidence.",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Accession, entry id, or latent example id the evidence came from.",
                    },
                },
                "required": ["claim", "source_tool", "source_id"],
                "additionalProperties": False,
            },
        },
        "rejected_hypotheses": {
            "type": "array",
            "description": "Alternative readings of the same activations, and why each was rejected.",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "why_rejected": {"type": "string"},
                },
                "required": ["hypothesis", "why_rejected"],
                "additionalProperties": False,
            },
        },
        "disconfirming_evidence": {
            "type": "string",
            "description": "What observation would falsify this label — a sequence that should fire but does not, or a firing site the label cannot explain.",
        },
        "mutation_effect": {
            "type": "object",
            "description": "Whether this latent is expected to carry signal for variant effect prediction.",
            "properties": {
                "is_relevant": {"type": "boolean"},
                "predicted_sensitivity": {
                    "type": "string",
                    "enum": MUTATION_SENSITIVITY,
                    "description": "Expected fitness impact of substitutions at the residues this latent marks.",
                },
                "rationale": {"type": "string"},
            },
            "required": ["is_relevant", "predicted_sensitivity", "rationale"],
            "additionalProperties": False,
        },
    },
    "required": [
        "latent_id",
        "label",
        "description",
        "concept_type",
        "confidence",
        "evidence",
        "rejected_hypotheses",
        "disconfirming_evidence",
        "mutation_effect",
    ],
    "additionalProperties": False,
}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Recover a JSON object from a model reply. Returns ``(object, error)``.

    Handles the three things open models actually do: clean JSON, JSON inside a
    ```json fence, and JSON with prose wrapped around it.
    """
    text = (text or "").strip()
    if not text:
        return None, "model returned an empty message"

    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "no JSON object found in the reply"


def validate_annotation(annotation: dict[str, Any]) -> list[str]:
    """Check an annotation against ANNOTATION_SCHEMA. Returns human-readable issues.

    Falls back to a required-keys check if `jsonschema` isn't installed, so the
    validation path never becomes the reason a run fails.
    """
    try:
        import jsonschema
    except ImportError:
        return [
            f"missing required field: {key}"
            for key in ANNOTATION_SCHEMA["required"]
            if key not in annotation
        ]

    validator = jsonschema.Draft202012Validator(ANNOTATION_SCHEMA)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(annotation), key=lambda e: list(e.path))
    ]


SYSTEM_PROMPT = """\
You annotate latents from a sparse autoencoder trained on protein language model \
activations. Each latent is a direction that fires on some subset of residues; \
your job is to name what that subset has in common, grounded in external biology \
rather than in your own priors about what an SAE "should" learn.

Method for each latent:
1. Call `get_latent_evidence` to see the top-activating windows and the accessions \
they come from.
2. Read the windows before looking anything up. Note what the bracketed peak \
residues share — residue identity, local motif, position in the chain, or nothing \
obvious.
3. Look the proteins up. Use the UniProt, InterPro, Pfam, and Reactome tools on the \
accessions in the evidence to test whether your reading survives contact with the \
annotated record. Check whether the peak positions fall inside annotated features \
(domains, sites, PTMs) rather than assuming they do.
4. Try to break your own label. A latent that fires on five kinases may be tracking \
kinase activity, or ATP binding, or simply the most common fold in the corpus. Look \
for the site that your label does not explain.

Calibration matters more than coverage. `low` confidence with an honest \
disconfirming case is more useful to this project than a confident label that the \
next SAE training run contradicts. If the activations have no coherent explanation, \
say so with concept_type `uninterpretable`; if they have two, use `polysemantic` and \
describe both.

Cite the tool and record id for every claim. Do not report a database fact you did \
not retrieve in this session.

Output format: when you have finished investigating, reply with a single JSON \
object matching the annotation schema — no prose before or after it, no code \
fence. Every field is required.\
"""
