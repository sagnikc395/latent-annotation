"""End-to-end demo: annotate SAE latents with a tool-using agent.

    uv run python examples/annotate_latents.py                       # mock model, offline
    uv run python examples/annotate_latents.py --provider huggingface  # Qwen-3, needs HF_TOKEN
    uv run python examples/annotate_latents.py L-0142                # one latent
    uv run python examples/annotate_latents.py --show-evidence       # no model at all

Provider defaults to `huggingface` when HF_TOKEN is set in `.env`, otherwise `mock`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_annotation.agent import LatentAnnotationAgent  # noqa: E402
from latent_annotation.config import FIXTURE_PATH, OUTPUT_PATH, PROTEIN_CONTEXT_TOOLS
from latent_annotation.evidence import LatentEvidenceStore
from latent_annotation.llm import load_client
from latent_annotation.tools import ToolBox, evidence_tools, load_tooluniverse


def build_toolbox(store: LatentEvidenceStore, *, with_tooluniverse: bool = True) -> ToolBox:
    if not with_tooluniverse:
        return ToolBox(local_tools=evidence_tools(store))
    tu = load_tooluniverse(PROTEIN_CONTEXT_TOOLS)
    return ToolBox(
        tooluniverse=tu,
        tooluniverse_tools=PROTEIN_CONTEXT_TOOLS,
        local_tools=evidence_tools(store),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latents", nargs="*", help="latent ids (default: all in the fixture)")
    parser.add_argument("--provider", choices=["mock", "huggingface"], default=None)
    parser.add_argument("--model", default=None, help="HF model id (default: Qwen-3)")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--no-tooluniverse", action="store_true", help="local tools only")
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="print evidence bundles and the tool surface, then exit without calling a model",
    )
    args = parser.parse_args()

    store = LatentEvidenceStore.from_json(args.fixture)
    latent_ids = args.latents or store.latent_ids
    toolbox = build_toolbox(store, with_tooluniverse=not args.no_tooluniverse)

    if args.show_evidence:
        print(f"{len(toolbox.names())} tools available:")
        for name in toolbox.names():
            print(f"  - {name}")
        for latent_id in latent_ids:
            print(f"\n=== {latent_id} ===")
            print(json.dumps(store.get(latent_id).to_dict(), indent=2))
        return 0

    client = load_client(args.provider, model=args.model)
    print(f"model layer: {type(client).__name__}")
    agent = LatentAnnotationAgent(client=client, toolbox=toolbox)

    failures = 0
    with args.output.open("w") as handle:
        for latent_id in latent_ids:
            print(f"\n=== {latent_id} ===", flush=True)
            result = agent.annotate(latent_id)
            if result.annotation is None:
                failures += 1
                print(f"  failed after {result.turns} turns: {result.error}")
                continue
            annotation = result.annotation
            print(f"  label      : {annotation.get('label')}")
            print(
                f"  type       : {annotation.get('concept_type')} "
                f"({annotation.get('confidence')} confidence)"
            )
            print(f"  tools used : {', '.join(result.tools_used) or 'none'}")
            print(f"  falsifiable: {annotation.get('disconfirming_evidence')}")
            if result.schema_errors:
                failures += 1
                print(f"  schema     : {len(result.schema_errors)} issue(s)")
                for issue in result.schema_errors[:5]:
                    print(f"               - {issue}")
            handle.write(
                json.dumps(
                    {
                        "annotation": annotation,
                        "tool_calls": result.tool_calls,
                        "turns": result.turns,
                        "usage": result.usage,
                        "schema_errors": result.schema_errors,
                    }
                )
                + "\n"
            )

    print(f"\nwrote {args.output} ({len(latent_ids) - failures}/{len(latent_ids)} clean)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
