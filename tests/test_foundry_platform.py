"""Direct unit tests for the Foundry hosted-agent call-context contract.

These exercise ``util.foundry_platform`` in isolation (no HTTP layer), so
that the validation semantics are proven independent of how the header
happens to be extracted. See ``tests/test_hosted_responses.py`` for
integration-level coverage through the ``/invocations`` endpoint.
"""

import pytest

from util import foundry_platform
from util.foundry_platform import (
    FOUNDRY_CALL_ID_HEADER,
    MissingFoundryCallContextError,
    extract_foundry_call_id,
    require_foundry_call_id,
)


class TestExtractFoundryCallId:
    def test_returns_none_when_header_absent(self):
        assert extract_foundry_call_id({}) is None

    def test_returns_none_for_whitespace_only_value(self):
        assert extract_foundry_call_id({FOUNDRY_CALL_ID_HEADER: "   "}) is None

    def test_strips_surrounding_whitespace(self):
        headers = {FOUNDRY_CALL_ID_HEADER: "  call-abc-123  "}
        assert extract_foundry_call_id(headers) == "call-abc-123"


class TestRequireFoundryCallId:
    def test_returns_validated_call_id(self):
        headers = {FOUNDRY_CALL_ID_HEADER: "call-abc-123"}
        assert require_foundry_call_id(headers) == "call-abc-123"

    def test_raises_when_header_missing(self):
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id({})

    def test_raises_when_header_empty(self):
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id({FOUNDRY_CALL_ID_HEADER: ""})

    def test_raises_when_header_whitespace_only(self):
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id({FOUNDRY_CALL_ID_HEADER: "   "})

    def test_raises_when_too_long(self):
        headers = {FOUNDRY_CALL_ID_HEADER: "x" * 257}
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id(headers)

    def test_accepts_max_length(self):
        headers = {FOUNDRY_CALL_ID_HEADER: "x" * 256}
        assert require_foundry_call_id(headers) == "x" * 256

    @pytest.mark.parametrize(
        "bad_value",
        [
            "has spaces",
            "embedded\nnewline",
            "embedded\rcarriage",
            "embedded\ttab",
        ],
    )
    def test_raises_for_embedded_whitespace(self, bad_value):
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id({FOUNDRY_CALL_ID_HEADER: bad_value})

    def test_error_message_never_includes_the_call_id_value(self):
        """The failure message documents the header name, not any value --
        even a rejected one must not be echoed into logs/errors."""
        secret_looking_value = "super-secret-token-should-not-leak"
        with pytest.raises(MissingFoundryCallContextError) as exc_info:
            require_foundry_call_id(
                {FOUNDRY_CALL_ID_HEADER: f"{secret_looking_value} has spaces"}
            )
        assert secret_looking_value not in str(exc_info.value)

    def test_rejects_trailing_newline_even_when_extraction_does_not_strip(
        self, monkeypatch
    ):
        """Regression test.

        ``_VALID_CALL_ID`` is anchored with ``^...$`` and no
        ``re.MULTILINE``. Used with ``.match()``, Python's ``$`` matches
        just before a single trailing "\\n", so a raw value of
        "call-id\\n" would incorrectly pass validation. Today
        ``extract_foundry_call_id`` strips such values before they reach
        the regex, masking the defect end-to-end -- but validation must
        not depend on that as its only safeguard, since a future direct
        call to ``require_foundry_call_id`` (or a refactor of
        extraction) could bypass the strip. This test forces exactly
        that scenario by making extraction hand back an unstripped,
        trailing-newline value, proving the regex check itself
        (``fullmatch``) rejects it.
        """
        monkeypatch.setattr(
            foundry_platform, "extract_foundry_call_id", lambda headers: "call-id\n"
        )
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id({})

    def test_rejects_trailing_carriage_return_even_when_extraction_does_not_strip(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            foundry_platform, "extract_foundry_call_id", lambda headers: "call-id\r"
        )
        with pytest.raises(MissingFoundryCallContextError):
            require_foundry_call_id({})

    def test_validation_regex_uses_fullmatch_not_match_semantics(self):
        """Direct regression test on the compiled pattern itself: proves
        the historical ``.match()`` leniency (trailing "\\n" accepted)
        is no longer how validation behaves, by asserting ``fullmatch``
        -- which ``require_foundry_call_id`` now uses -- rejects it even
        though ``.match()`` on the very same pattern object would not.
        """
        pattern = foundry_platform._VALID_CALL_ID
        assert pattern.match("call-id\n") is not None  # historical leniency
        assert pattern.fullmatch("call-id\n") is None  # fixed validation path
