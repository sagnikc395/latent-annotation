"""The annotation agent: a tool-use loop over an OpenAI-shaped chat interface.

The loop is deliberately explicit rather than delegated to a framework, for two
reasons specific to this project: we want the per-latent tool trace as a research
artifact (which tools an annotation actually consulted is part of the result), and
open-weights tool calling is inconsistent enough that the failure modes need to be
visible — a model that answers without calling `get_latent_evidence` has
hallucinated its annotation, and the loop should be able to say so.

Shape of one annotation:

    user: "annotate L-0142"
      -> tool_calls: get_latent_evidence          -> tool results
      -> tool_calls: UniProt_get_function_by_...  -> tool results
      -> ... until the model answers without tool calls
      -> final text, parsed and validated against ANNOTATION_SCHEMA
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig
from .llm import ChatClient, ChatResponse
from .schema import ANNOTATION_SCHEMA, SYSTEM_PROMPT, extract_json, validate_annotation
from .tools import ToolBox

# Open-weights models skip tools more often than frontier ones. If the model tries
# to answer before reading the evidence, we push back once instead of accepting an
# annotation that was invented rather than observed.
NUDGE_MISSING_EVIDENCE = (
    "You have not called `get_latent_evidence` yet, so you have not seen a single "
    "activation. Call it for this latent before proposing any label."
)


@dataclass
class AnnotationResult:
    latent_id: str
    annotation: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    schema_errors: list[str] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [call["tool"] for call in self.tool_calls]


class LatentAnnotationAgent:
    """Annotates SAE latents one at a time, with a fresh context per latent."""

    def __init__(
        self, client: ChatClient, toolbox: ToolBox, config: AgentConfig | None = None
    ):
        self.client = client
        self.toolbox = toolbox
        self.config = config or AgentConfig()

    # -- public API --------------------------------------------------------

    def annotate(
        self, latent_id: str, *, extra_instructions: str = ""
    ) -> AnnotationResult:
        task = (
            f"Annotate latent {latent_id}.\n\n"
            "Start by calling get_latent_evidence, then verify your reading against "
            "the database tools before committing to a label. When you are done, "
            "reply with the annotation as a single JSON object and nothing else."
        )
        if extra_instructions:
            task += f"\n\nAdditional context for this run:\n{extra_instructions}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        result = AnnotationResult(latent_id=latent_id, annotation=None)
        nudged = False
        repairs = 0

        for turn in range(1, self.config.max_turns + 1):
            result.turns = turn
            response = self._complete(messages, result)

            if response.wants_tools:
                messages.append(_assistant_tool_message(response))
                messages.extend(self._run_tools(response, result))
                continue

            # No tool calls: the model is trying to answer.
            if (
                self.config.require_evidence_tool
                and not nudged
                and "get_latent_evidence" not in result.tools_used
            ):
                nudged = True
                messages.append({"role": "assistant", "content": response.text})
                messages.append({"role": "user", "content": NUDGE_MISSING_EVIDENCE})
                continue

            annotation, parse_error = extract_json(response.text)
            if annotation is None:
                if repairs < self.config.max_repair_attempts:
                    repairs += 1
                    messages.append({"role": "assistant", "content": response.text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"That reply could not be parsed as JSON ({parse_error}). "
                                "Reply with the annotation object only — no prose, no code fence."
                            ),
                        }
                    )
                    continue
                result.error = parse_error
                return result

            result.annotation = annotation
            result.schema_errors = validate_annotation(annotation)
            if result.schema_errors and repairs < self.config.max_repair_attempts:
                repairs += 1
                messages.append({"role": "assistant", "content": response.text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The annotation does not match the required schema:\n"
                            + "\n".join(f"- {issue}" for issue in result.schema_errors)
                            + "\nReturn a corrected JSON object."
                        ),
                    }
                )
                continue

            return result

        result.error = f"hit max_turns ({self.config.max_turns}) without a final answer"
        return result

    # -- internals ---------------------------------------------------------

    def _complete(
        self, messages: list[dict[str, Any]], result: AnnotationResult
    ) -> ChatResponse:
        response = self.client.complete(
            messages,
            self.toolbox.definitions(),
            response_schema=ANNOTATION_SCHEMA,
        )
        for key, value in response.usage.items():
            result.usage[key] = result.usage.get(key, 0) + value
        return response

    def _run_tools(
        self, response: ChatResponse, result: AnnotationResult
    ) -> list[dict[str, Any]]:
        """Execute every tool call from one assistant turn."""
        messages = []
        for call in response.tool_calls:
            text, is_error = self.toolbox.run(call.name, call.arguments)
            result.tool_calls.append(
                {
                    "turn": result.turns,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "is_error": is_error,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": text,
                }
            )
        return messages


def _assistant_tool_message(response: ChatResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.text or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in response.tool_calls
        ],
    }
