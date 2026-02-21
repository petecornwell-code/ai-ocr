import json
from collections import Counter


def run_ocr_analysis(
    extracted_text: str,
    extraction_schema: dict,
) -> dict:
    """Run a 3-agent voting crew to extract and validate fields from OCR text.

    Each of the three OCR extraction agents uses a different LLM and
    independently extracts the fields defined in ``extraction_schema``.
    The vote coordinator then compares results field-by-field:
      - 2/3 or 3/3 agreement  → consensus value (success)
      - all three disagree     → field flagged for intervention

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

    # ── Three OCR-extraction agents, each on a different LLM ───────────
    agent_a = Agent(
        role="OCR Extraction Agent A",
        goal=(
            "Extract structured field values from raw OCR text. "
            "Return ONLY valid JSON with the requested field names as keys."
        ),
        backstory=(
            "You are a meticulous data-entry specialist who reads raw OCR "
            "output and extracts exact field values without hallucinating."
        ),
        llm="model_a",
        verbose=False,
    )

    agent_b = Agent(
        role="OCR Extraction Agent B",
        goal=(
            "Extract structured field values from raw OCR text. "
            "Return ONLY valid JSON with the requested field names as keys."
        ),
        backstory=(
            "You are a detail-oriented document processor who carefully "
            "identifies and extracts field values from noisy OCR text."
        ),
        llm="model_b",
        verbose=False,
    )

    agent_c = Agent(
        role="OCR Extraction Agent C",
        goal=(
            "Extract structured field values from raw OCR text. "
            "Return ONLY valid JSON with the requested field names as keys."
        ),
        backstory=(
            "You are an experienced forms analyst who can reliably parse "
            "OCR output and return accurate field values."
        ),
        llm="model_c",
        verbose=False,
    )

    # ── Extraction tasks (one per agent) ───────────────────────────────
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

    task_a = Task(
        description=task_prompt,
        expected_output=expected,
        agent=agent_a,
    )
    task_b = Task(
        description=task_prompt,
        expected_output=expected,
        agent=agent_b,
    )
    task_c = Task(
        description=task_prompt,
        expected_output=expected,
        agent=agent_c,
    )

    # ── Vote coordinator agent ─────────────────────────────────────────
    coordinator = Agent(
        role="Vote Coordinator",
        goal=(
            "Compare the three extraction results field-by-field. "
            "For each field, determine if at least two agents agree. "
            "Return a JSON report."
        ),
        backstory=(
            "You are a quality-assurance coordinator responsible for "
            "reconciling outputs from multiple OCR extraction agents. "
            "You tally votes per field and flag disagreements."
        ),
        verbose=False,
    )

    coordination_task = Task(
        description=(
            "You will receive extraction results from three agents as context. "
            "For each field, compare the three values:\n"
            "- If 2 or 3 agents agree, mark the field as 'consensus' with the "
            "  agreed-upon value.\n"
            "- If all 3 values are different, mark the field as "
            "  'intervention_required' with a null value.\n\n"
            "Return a JSON object with the structure:\n"
            '{"fields": {"<name>": {"value": ..., "status": "consensus"|'
            '"intervention_required", "votes": {"agent_a": ..., "agent_b": '
            '..., "agent_c": ...}}}, "summary": {"total_fields": N, '
            '"consensus_count": N, "intervention_count": N}}'
        ),
        expected_output=(
            "A JSON object with 'fields' and 'summary' keys following the "
            "schema described above."
        ),
        agent=coordinator,
        context=[task_a, task_b, task_c],
    )

    # ── Assemble and run the crew ──────────────────────────────────────
    crew = Crew(
        agents=[agent_a, agent_b, agent_c, coordinator],
        tasks=[task_a, task_b, task_c, coordination_task],
        verbose=False,
    )

    result = crew.kickoff()

    # Try to parse a structured result; fall back to the raw string.
    try:
        return json.loads(str(result))
    except json.JSONDecodeError:
        return _build_result_from_raw(str(result), field_names)


# ── Deterministic vote tallying (used by tests and as fallback) ────────


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


def _build_result_from_raw(raw: str, field_names: list[str]) -> dict:
    """Last-resort wrapper when the crew output isn't valid JSON."""
    fields = {}
    for f in field_names:
        fields[f] = {
            "value": None,
            "status": "intervention_required",
            "votes": {"agent_a": None, "agent_b": None, "agent_c": None},
        }
    return {
        "raw_output": raw,
        "fields": fields,
        "summary": {
            "total_fields": len(field_names),
            "consensus_count": 0,
            "intervention_count": len(field_names),
        },
    }
