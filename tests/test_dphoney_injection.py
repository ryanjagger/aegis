"""U6: DP canaries wired into the live injection path behind the source toggle."""

from __future__ import annotations

import pathlib

from sqlalchemy import select

import app.canaries.dp_source as dp_source
from app.canaries.injector import inject_canaries
from app.db.database import SessionLocal
from app.db.models import CanaryRecord, EventRecord
from app.scanners.canary_scanner import CanaryScanner
from dphoney.corpus import FORMATS


def _inject(source: str, request_id: str):
    with SessionLocal() as db:
        items, context, canaries = inject_canaries(
            db,
            input_items=[{"role": "user", "content": "hi"}],
            request_id=request_id,
            session_id="sess_dp",
            source=source,
        )
        db.commit()
        rows = list(
            db.execute(select(CanaryRecord).where(CanaryRecord.request_id == request_id)).scalars()
        )
        events = list(
            db.execute(select(EventRecord).where(EventRecord.request_id == request_id)).scalars()
        )
    return items, context, canaries, rows, events


def test_dp_canaries_are_format_valid_and_detected():  # AE3
    _, _, canaries, _, _ = _inject("dp", "req_dp_valid")
    assert len(canaries) == 5
    for canary in canaries:
        assert FORMATS[canary.format].validate(canary.value)
    # The text scanner still detects a leaked DP canary verbatim.
    leaked = canaries[0]
    scanner = CanaryScanner([{"id": leaked.id, "value": leaked.value}])
    assert scanner.scan_text(f"the secret is {leaked.value} ok", "test")


def test_template_default_unchanged():
    items, context, canaries, rows, _ = _inject("template", "req_tmpl")
    assert len(canaries) == 5 and len(rows) == 5
    assert "Internal diagnostic appendix" in context
    assert items[-1]["content"] == context
    github = next(c for c in canaries if c.format == "github_pat")
    assert github.value.startswith("ghp_") and len(github.value) == 40


def test_dp_uses_same_format_set_as_template():
    _, _, dp, _, _ = _inject("dp", "req_dp_fmt")
    _, _, template, _, _ = _inject("template", "req_tmpl_fmt")
    assert [c.format for c in dp] == [c.format for c in template]


def test_dp_sampling_failure_falls_back_to_template(monkeypatch):
    import dphoney.generator as gen

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(gen, "sample_default_canary", boom)
    canary = dp_source.generate_dp_canary("github_pat", "system_context")
    # Falls back to a format-valid template canary; no exception escapes.
    assert FORMATS["github_pat"].validate(canary.value)


def test_dp_path_stores_and_emits_like_template():
    _, _, _, rows, events = _inject("dp", "req_dp_evt")
    assert len(rows) == 5
    assert len(events) == 5
    assert all(event.event_type == "canary.injected" for event in events)


def test_dp_source_module_stays_light():
    # The live path must not pull torch/sklearn at module import.
    src = pathlib.Path(dp_source.__file__).read_text()
    imports = "\n".join(
        line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
    )
    assert "torch" not in imports and "sklearn" not in imports
