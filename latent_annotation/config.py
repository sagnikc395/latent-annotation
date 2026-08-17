"""Central configuration for the project.

Every tunable lives here so a change is visible in one file instead of buried in
the module that happens to use it. Modules import the constants they need and
fall back to them as defaults, so callers can still override per call (e.g. a
different model for a smoke test) without touching config.

Two kinds of configuration:

* **Code settings** — tool lists, evidence windowing, agent loop limits,
  result truncation. Change the value here.
* **Environment settings** — provider, model id, API token. These are read from
  the environment / `.env` (see `.env.example`); the constants below are the
  variable *names* and the fallback defaults.

Nothing in this module imports the rest of the package, so any module can import
from it without creating cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "examples" / "fixtures" / "latents.json"
OUTPUT_PATH = REPO_ROOT / "examples" / "annotations.jsonl"
ENV_FILE = ".env"

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
# Documented in .env.example. LLM_PROVIDER selects the model layer (mock or
# huggingface); HF_TOKEN is the API token, HUGGINGFACE_API_KEY accepted as an
# alias; LLM_MODEL overrides the default model id.

ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_API_KEY = "HUGGINGFACE_API_KEY"

# ---------------------------------------------------------------------------
# Model layer (latent_annotation/llm.py)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_HF_PROVIDER = "auto"  # HF routing: let the Inference API pick the provider
TEMPERATURE = 0.2
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 180.0
# Only request `response_format` once tool calling is done; some HF providers
# emit the schema instead of the tool call when both are active.
USE_RESPONSE_FORMAT = True

# The database tool MockChatClient calls after fetching evidence, so an offline
# run exercises a real ToolUniverse round-trip. Must be one of the tools the
# toolbox actually exposes — i.e. a member of PROTEIN_CONTEXT_TOOLS below.
MOCK_LOOKUP_TOOL = "UniProt_get_function_by_accession"

# ---------------------------------------------------------------------------
# Agent loop (latent_annotation/agent.py)
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Knobs for the per-latent tool-use loop."""

    max_turns: int = 12
    require_evidence_tool: bool = True  # nudge once if the model answers ungrounded
    max_repair_attempts: int = 2  # retries when the final message is not valid JSON


# ---------------------------------------------------------------------------
# Evidence extraction (latent_annotation/evidence.py)
# ---------------------------------------------------------------------------

TOP_K_EXAMPLES = 8  # how many top-activating windows the agent gets to see
WINDOW_SIZE = 24  # residues on each side of the peak in a window
HIGHLIGHT_SPAN = 2  # residues bracketed on each side of the peak
ACTIVE_THRESHOLD = 0.0  # activation above this counts as "firing"

# ---------------------------------------------------------------------------
# Tool surface (latent_annotation/tools.py)
# ---------------------------------------------------------------------------
# Curated starting sets. Full catalog: `ToolUniverse().load_tools()` then
# `find_tools_by_pattern("UniProt")`. Keep these lists short and ordered — every
# entry costs context on every request, and reordering invalidates the cache.

PROTEIN_CONTEXT_TOOLS: tuple[str, ...] = (
    "UniProt_get_recommended_name_by_accession",
    "UniProt_get_function_by_accession",
    "UniProt_get_features_by_accession",
    "UniProt_get_subcellular_location_by_accession",
    "UniProt_get_ptm_processing_by_accession",
    "InterPro_get_entries_for_protein",
    "InterPro_get_protein_domain_architecture",
    "Pfam_get_protein_annotations",
    "Reactome_map_uniprot_to_pathways",
)

MUTATION_EFFECT_TOOLS: tuple[str, ...] = (
    "UniProt_get_disease_variants_by_accession",
    "MaveDB_search_score_sets",
    "MaveDB_get_variant_scores",
    "MaveDB_get_effect_matrix",
    "ClinVar_search_variants",
    "gnomad_get_constraint",
)

# Tool results are serialized and truncated to this many characters — one
# UniProt entry can otherwise eat the context window before the agent has
# looked at anything else.
TOOL_RESULT_LIMIT = 8_000
