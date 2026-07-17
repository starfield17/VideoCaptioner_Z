from __future__ import annotations

from captioner.infrastructure.redaction import REDACTED, redact, redact_headers, redact_json


def test_redaction_covers_text_headers_and_nested_json() -> None:
    secret = "unit-test-key"
    assert redact("Bearer unit-test-key", (secret,)) == f"Bearer {REDACTED}"
    headers = redact_headers({"Authorization": f"Bearer {secret}", "X-Trace": secret}, (secret,))
    assert headers == {"Authorization": REDACTED, "X-Trace": REDACTED}
    value = redact_json({"api_key": secret, "nested": [f"value:{secret}"], "number": 2}, (secret,))
    assert value == {"api_key": REDACTED, "nested": [f"value:{REDACTED}"], "number": 2}
    assert redact({"authorization": secret}, (secret,)) == {"authorization": REDACTED}
