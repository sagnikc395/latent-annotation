"""ToolUniverse discovery helper — find tools worth adding to the agent's surface.

The agent itself exposes a curated handful of tools (see latent_annotation/tools.py
and notes/02-tooluniverse.md). This script is for the other half of the job:
searching the full catalog (~2,600 tools, and it grows with each ToolUniverse
release) when you need a capability the curated set does not cover.

    uv run python main.py                  # summarize the catalog
    uv run python main.py UniProt InterPro # search by pattern
    uv run python main.py --spec UniProt_get_features_by_accession
"""

from __future__ import annotations

import argparse
import json

from tooluniverse import ToolUniverse

from latent_annotation.config import MUTATION_EFFECT_TOOLS, PROTEIN_CONTEXT_TOOLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="*", help="substrings to search tool names for")
    parser.add_argument("--spec", help="print the full OpenAI spec for one tool")
    parser.add_argument("--limit", type=int, default=25, help="max hits per pattern")
    args = parser.parse_args()

    tu = ToolUniverse()
    tu.load_tools()

    if args.spec:
        print(json.dumps(tu.tool_specification(args.spec, format="openai"), indent=2))
        return

    if not args.patterns:
        print(f"{len(tu.all_tools)} tools loaded.\n")
        print("Currently exposed to the annotation agent:")
        for name in PROTEIN_CONTEXT_TOOLS:
            print(f"  {name}")
        print("\nReserved for the mutation-effect agent:")
        for name in MUTATION_EFFECT_TOOLS:
            print(f"  {name}")
        print("\nPass patterns to search, e.g. `uv run python main.py MaveDB alphafold`.")
        return

    names = list(tu.all_tool_dict)
    for pattern in args.patterns:
        hits = [name for name in names if pattern.lower() in name.lower()]
        print(f"\n=== {pattern} ({len(hits)} hits) ===")
        for name in hits[: args.limit]:
            spec = tu.tool_specification(name, format="openai") or {}
            description = str(spec.get("description", "")).split("\n")[0]
            print(f"  {name}\n      {description[:110]}")
        if len(hits) > args.limit:
            print(f"  ... {len(hits) - args.limit} more")


if __name__ == "__main__":
    main()
