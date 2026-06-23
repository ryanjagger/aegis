import base64
from urllib.parse import quote_plus

from app.scanners.canary_scanner import CanaryScanner

CANARY = {
    "id": "can_test",
    "value": "ghp_a8F2kLm9QzR4tYp7VxN3cB6uHsW0eD1jK5mT",
}


def test_exact_canary_scanner_detects_registered_canary() -> None:
    hits = CanaryScanner([CANARY]).scan_text(f"secret={CANARY['value']}", "surface")
    assert any(hit.detector == "exact_canary_match" for hit in hits)


def test_base64_scanner_detects_encoded_canary() -> None:
    encoded = base64.b64encode(CANARY["value"].encode()).decode()
    hits = CanaryScanner([CANARY]).scan_text(f"value={encoded}", "surface")
    assert any(hit.detector == "base64_decoded_canary_match" for hit in hits)


def test_hex_scanner_detects_encoded_canary() -> None:
    encoded = CANARY["value"].encode().hex()
    hits = CanaryScanner([CANARY]).scan_text(f"value={encoded}", "surface")
    assert any(hit.detector == "hex_decoded_canary_match" for hit in hits)


def test_url_decoder_detects_url_encoded_canary() -> None:
    encoded = quote_plus(CANARY["value"])
    hits = CanaryScanner([CANARY]).scan_text(f"https://example.test/?token={encoded}", "surface")
    assert any(hit.detector == "url_decoded_canary_match" for hit in hits)
