"""Live DP canary source (U6): GeneratedCanary objects from the DP-HONEY model.

numpy-only — no sklearn/matplotlib — so the FastAPI app injects DP canaries
without the opt-in ``dphoney`` group. The values come from the same seeded
default-ε model the lab measures as indistinguishable. If DP sampling fails for
any reason, falls back to the template generator so a live request can never 500.
"""

from __future__ import annotations

from app.canaries.generator import GeneratedCanary, canary_hash, generate_canary
from app.db.repository import short_id


def generate_dp_canary(format_name: str, source_label: str) -> GeneratedCanary:
    try:
        from dphoney.generator import sample_default_canary

        value = sample_default_canary(format_name)
    except Exception:
        # Degenerate model, missing format, import error — fall back to template.
        return generate_canary(format_name, source_label)
    return GeneratedCanary(
        id=short_id("can"),
        value=value,
        value_hash=canary_hash(value),
        format=format_name,
        source_label=source_label,
    )


def generate_dp_canaries(
    *,
    formats: list[str] | tuple[str, ...],
    source_labels: list[str] | tuple[str, ...],
) -> list[GeneratedCanary]:
    canaries: list[GeneratedCanary] = []
    for index, format_name in enumerate(formats):
        source_label = source_labels[index % len(source_labels)]
        canaries.append(generate_dp_canary(format_name, source_label))
    return canaries
