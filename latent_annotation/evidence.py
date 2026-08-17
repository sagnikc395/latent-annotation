"""Turn raw SAE activations into the evidence an annotator agent can reason about.

An SAE latent is only interpretable through the inputs that fire it. This module
takes per-residue activations for one latent over a corpus of proteins and
produces a compact, quotable evidence bundle: top-activating windows, the
accessions behind them, and simple firing statistics.

Deliberately dependency-free (plain lists, no torch/numpy) so the evidence layer
can be unit-tested without a model checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import config


@dataclass(frozen=True)
class ProteinRecord:
    """One sequence in the corpus the SAE was run over."""

    protein_id: str
    sequence: str
    accession: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class LatentExample:
    """A single top-activating site for a latent."""

    protein_id: str
    accession: str | None
    max_activation: float
    peak_index: int
    window_start: int
    highlighted: str  # the window, with the peak residue bracketed
    description: str | None = None


@dataclass
class LatentEvidence:
    """Everything the agent gets to see about one latent before it calls tools."""

    latent_id: str
    examples: list[LatentExample] = field(default_factory=list)
    firing_rate: float = 0.0
    mean_active_activation: float = 0.0
    max_activation: float = 0.0
    n_proteins_scanned: int = 0

    @property
    def accessions(self) -> list[str]:
        seen: list[str] = []
        for ex in self.examples:
            if ex.accession and ex.accession not in seen:
                seen.append(ex.accession)
        return seen

    def to_dict(self) -> dict:
        return {
            "latent_id": self.latent_id,
            "firing_rate": round(self.firing_rate, 6),
            "mean_active_activation": round(self.mean_active_activation, 4),
            "max_activation": round(self.max_activation, 4),
            "n_proteins_scanned": self.n_proteins_scanned,
            "accessions": self.accessions,
            "examples": [asdict(ex) for ex in self.examples],
        }


def _highlight(sequence: str, start: int, peak: int, span: int) -> str:
    """Bracket the peak residue inside a window so the model can see where it fired."""
    local = peak - start
    if not 0 <= local < len(sequence):
        return sequence
    lo = max(0, local - span)
    hi = min(len(sequence), local + span + 1)
    return f"{sequence[:lo]}[{sequence[lo:hi]}]{sequence[hi:]}"


def build_latent_evidence(
    latent_id: str,
    proteins: Sequence[ProteinRecord],
    activations: Mapping[str, Sequence[float]],
    *,
    top_k: int = config.TOP_K_EXAMPLES,
    window: int = config.WINDOW_SIZE,
    highlight_span: int = config.HIGHLIGHT_SPAN,
    active_threshold: float = config.ACTIVE_THRESHOLD,
) -> LatentEvidence:
    """Summarize one latent from per-residue activations.

    ``activations`` maps ``protein_id`` to a per-residue activation vector aligned
    with that protein's sequence. Proteins with no entry are treated as unscanned.
    """
    peaks: list[tuple[float, int, ProteinRecord]] = []
    active_values: list[float] = []
    n_residues = 0
    n_active = 0
    scanned = 0

    for protein in proteins:
        acts = activations.get(protein.protein_id)
        if acts is None:
            continue
        scanned += 1
        n_residues += len(acts)
        best_value = float("-inf")
        best_index = 0
        for index, value in enumerate(acts):
            if value > active_threshold:
                n_active += 1
                active_values.append(value)
            if value > best_value:
                best_value, best_index = value, index
        if best_value > active_threshold:
            peaks.append((best_value, best_index, protein))

    peaks.sort(key=lambda item: item[0], reverse=True)

    examples: list[LatentExample] = []
    for value, index, protein in peaks[:top_k]:
        start = max(0, index - window)
        end = min(len(protein.sequence), index + window + 1)
        window_sequence = protein.sequence[start:end]
        examples.append(
            LatentExample(
                protein_id=protein.protein_id,
                accession=protein.accession,
                max_activation=round(float(value), 4),
                peak_index=index,
                window_start=start,
                highlighted=_highlight(window_sequence, start, index, highlight_span),
                description=protein.description,
            )
        )

    return LatentEvidence(
        latent_id=latent_id,
        examples=examples,
        firing_rate=(n_active / n_residues) if n_residues else 0.0,
        mean_active_activation=(sum(active_values) / len(active_values)) if active_values else 0.0,
        max_activation=max((p[0] for p in peaks), default=0.0),
        n_proteins_scanned=scanned,
    )


class LatentEvidenceStore:
    """Lookup of precomputed evidence, keyed by latent id.

    The agent reads from this through a tool rather than getting every latent
    stuffed into its prompt, so annotating 10,000 latents costs the same context
    per latent as annotating one.
    """

    def __init__(self, evidence: Iterable[LatentEvidence]):
        self._evidence = {ev.latent_id: ev for ev in evidence}

    def __len__(self) -> int:
        return len(self._evidence)

    @property
    def latent_ids(self) -> list[str]:
        return list(self._evidence)

    def get(self, latent_id: str) -> LatentEvidence:
        try:
            return self._evidence[latent_id]
        except KeyError:
            raise KeyError(f"unknown latent {latent_id!r}") from None

    @classmethod
    def from_activation_dump(
        cls,
        proteins: Sequence[ProteinRecord],
        activations: Mapping[str, Mapping[str, Sequence[float]]],
        **kwargs,
    ) -> "LatentEvidenceStore":
        """Build from ``{latent_id: {protein_id: [act, ...]}}``."""
        return cls(
            build_latent_evidence(latent_id, proteins, per_protein, **kwargs)
            for latent_id, per_protein in activations.items()
        )

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "LatentEvidenceStore":
        """Load a fixture with ``proteins`` and ``activations`` keys."""
        payload = json.loads(Path(path).read_text())
        proteins = [ProteinRecord(**record) for record in payload["proteins"]]
        return cls.from_activation_dump(proteins, payload["activations"], **kwargs)
