"""Generate a small activation fixture so the agent can be run without an SAE.

The accessions are real (so the database tools hit real records); the sequence
fragments are synthetic stand-ins carrying the motif each latent is supposed to
detect. Replace this whole file with a dump from your own SAE — the agent only
ever sees `proteins` + `activations`.

    uv run python examples/make_fixture.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_annotation.config import FIXTURE_PATH as OUT  # noqa: E402

# accession, protein_id, illustrative fragment, note
PROTEINS = [
    (
        "P00533",
        "EGFR_frag",
        "LGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTS",
        "receptor tyrosine kinase, catalytic domain fragment",
    ),
    (
        "P04637",
        "TP53_frag",
        "SSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGST",
        "tumour suppressor, DNA-binding domain fragment",
    ),
    (
        "P68871",
        "HBB_frag",
        "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKK",
        "haemoglobin subunit beta, N-terminal fragment",
    ),
    (
        "P05067",
        "APP_frag",
        "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVE",
        "amyloid-beta precursor, Abeta region",
    ),
    (
        "P0DP23",
        "CALM_frag",
        "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPE",
        "calmodulin, EF-hand pair",
    ),
    (
        "P62826",
        "RAN_frag",
        "MAAQGEPQVQFKLVLVGDGGTGKTTFVKRHLTGEFEKKYVATLGVEVHPLVFHTNRGPIKFNVWDTAG",
        "Ran GTPase, P-loop region",
    ),
    (
        "P06213",
        "INSR_frag",
        "LGQGSFGMVYEGNARDIIKGEAETRVAVKTVNESASLRERIEFLNEASVMKGFTCHHVVRLLGVVSKG",
        "insulin receptor, catalytic domain fragment",
    ),
    (
        "P00519",
        "ABL1_frag",
        "LGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIIT",
        "ABL1 tyrosine kinase, catalytic domain fragment",
    ),
]

# Three latents chosen to span the range the annotator has to handle: one sharp
# and specific, one broad-but-real, one that should come back as noise. If every
# fixture latent is cleanly interpretable you never see whether the agent is
# calibrated.
LATENTS = {
    # Glycine-rich nucleotide-binding loop — sharp, structurally meaningful.
    "L-0142": re.compile(r"G.G..G"),
    # Acidic pairs — real chemistry, but fires nearly everywhere; a good test of
    # whether the agent reaches for `polysemantic` instead of over-labelling.
    "L-0731": re.compile(r"[DE][DE]"),
    # Any aromatic residue — the honest answer here is `uninterpretable`.
    "L-1503": re.compile(r"[FWY]"),
}


def activations_for(pattern: re.Pattern[str], sequence: str) -> list[float]:
    acts = [0.0] * len(sequence)
    for match in pattern.finditer(sequence):
        start, end = match.span()
        peak = (start + end) // 2
        for i in range(start, end):
            # triangular profile around the motif centre
            acts[i] = round(2.4 - 0.25 * abs(i - peak), 3)
    return acts


def main() -> None:
    proteins = [
        {
            "protein_id": protein_id,
            "accession": accession,
            "sequence": sequence,
            "description": f"{note} (illustrative fragment, not the full record)",
        }
        for accession, protein_id, sequence, note in PROTEINS
    ]

    activations = {
        latent_id: {
            protein["protein_id"]: activations_for(pattern, protein["sequence"])
            for protein in proteins
        }
        for latent_id, pattern in LATENTS.items()
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"proteins": proteins, "activations": activations}, indent=2) + "\n"
    )
    for latent_id, per_protein in activations.items():
        firing = sum(1 for acts in per_protein.values() if any(a > 0 for a in acts))
        print(f"{latent_id}: fires in {firing}/{len(proteins)} proteins")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
