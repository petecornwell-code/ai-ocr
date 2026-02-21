import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)

AGENT_NAMES = ("agent_a", "agent_b", "agent_c")


def run_ocr_analysis(
    extracted_text: str,
    extraction_schema: dict,
) -> dict:
    """Run a 3-agent crew to extract fields, then tally votes deterministically.

    Each of the three OCR extraction agents uses a different LLM and
    independently extracts the fields defined in ``extraction_schema``.
    Their raw outputs are parsed into dicts, then fed to ``tally_votes``
    which compares results field-by-field:

      - 2/3 or 3/3 agreement  -> consensus value (success)
      - all three disagree     -> field flagged for intervention

    Returns a dict of the form::

        {
            "fields": {
                "<field_name>": {
                    "value": <consensus_value or null>,
                    "status": "consensus" | "intervention_required",
                    "votes": {"agent_a": ..., "agent_b": ..., "agent_c": ...}
                },
                ...
            },
            "summary": {
                "total_fields": int,
                "consensus_count": int,
                "intervention_count": int,
            }
        }
    """
    from crewai import Agent, Crew, Task

    field_names = list(extraction_schema.get("fields", {}).keys())
    fields_description = json.dumps(extraction_schema["fields"], indent=2)

    task_prompt = (
        "Extract the following fields from the OCR text below.\n\n"
        f"Fields to extract:\n{fields_description}\n\n"
        f"OCR text:\n{extracted_text}\n\n"
        "Return ONLY a JSON object mapping each field name to its extracted "
        "value. Use null for any field you cannot find."
    )
    expected = (
        "A JSON object with keys: "
        + ", ".join(field_names)
        + ". Each value is the extracted text or null."
    )

    agents_config = [
        {
            "role": "OCR Extraction Agent A",
            "backstory": (
                "You are a meticulous data-entry specialist who reads raw OCR "
                "output and extracts exact field values without hallucinating."
            ),
            "llm": "model_a",
        },
        {
            "role": "OCR Extraction Agent B",
            "backstory": (
                "You are a detail-oriented document processor who carefully "
                "identifies and extracts field values from noisy OCR text."
            ),
            "llm": "model_b",
        },
        {
            "role": "OCR Extraction Agent C",
            "backstory": (
                "You are an experienced forms analyst who can reliably parse "
                "OCR output and return accurate field values."
            ),
            "llm": "model_c",
        },
    ]

    agents = []
    tasks = []
    for cfg in agents_config:
        agent = Agent(
            role=cfg["role"],
            goal=(
                "Extract structured field values from raw OCR text. "
                "Return ONLY valid JSON with the requested field names as keys."
            ),
            backstory=cfg["backstory"],
            llm=cfg["llm"],
            verbose=False,
        )
        task = Task(
            description=task_prompt,
            expected_output=expected,
            agent=agent,
        )
        agents.append(agent)
        tasks.append(task)

    crew = Crew(agents=agents, tasks=tasks, verbose=False)
    crew.kickoff()

    # ── Parse each agent's output and tally deterministically ──────────
    agent_results = {}
    for name, task in zip(AGENT_NAMES, tasks):
        agent_results[name] = _parse_task_output(task, field_names)

    return tally_votes(agent_results, field_names)


def _parse_task_output(task, field_names: list[str]) -> dict[str, str | None]:
    """Extract a dict from a CrewAI task's output, with fallback to empty."""
    raw = str(task.output) if task.output else ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        # Try to find a JSON object embedded in the output text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse agent output as JSON: %.200s", raw)
    return {f: None for f in field_names}


# ── Deterministic vote tallying ────────────────────────────────────────


def tally_votes(
    agent_results: dict[str, dict[str, str | None]],
    field_names: list[str],
) -> dict:
    """Pure-logic vote tallying independent of CrewAI.

    Parameters
    ----------
    agent_results:
        ``{"agent_a": {field: value, ...}, "agent_b": ..., "agent_c": ...}``
    field_names:
        The fields to tally across.

    Returns the same ``{"fields": ..., "summary": ...}`` structure.
    """
    fields: dict = {}
    consensus_count = 0
    intervention_count = 0

    for field in field_names:
        votes = {
            agent: results.get(field)
            for agent, results in agent_results.items()
        }
        values = list(votes.values())
        counts = Counter(values)
        most_common_value, most_common_count = counts.most_common(1)[0]

        if most_common_count >= 2:
            fields[field] = {
                "value": most_common_value,
                "status": "consensus",
                "votes": votes,
            }
            consensus_count += 1
        else:
            fields[field] = {
                "value": None,
                "status": "intervention_required",
                "votes": votes,
            }
            intervention_count += 1

    return {
        "fields": fields,
        "summary": {
            "total_fields": len(field_names),
            "consensus_count": consensus_count,
            "intervention_count": intervention_count,
        },
    }
