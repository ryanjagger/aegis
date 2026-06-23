from pydantic import BaseModel


class DetectorHit(BaseModel):
    detector: str
    surface: str
    severity: str
    matched_canary_id: str | None = None
    evidence_preview: str
    policy_recommendation: str
    matched_value: str | None = None
