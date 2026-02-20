from __future__ import annotations

import base64
import json
from typing import Any, Optional, Dict


def extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extracts the raw token from a header value like 'Bearer <token>'.

    Returns None when header_value is None/empty or not a Bearer token.
    """
    if not header_value:
        return None

    value = header_value.strip()
    if not value:
        return None

    parts = value.split(None, 1)
    if len(parts) != 2:
        return None

    scheme, token = parts[0], parts[1]
    if scheme.lower() != "bearer":
        return None

    token = token.strip()

    # Some proxies/frameworks may wrap header values in quotes.
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()

    return token or None


def _b64url_decode(segment: str) -> bytes:
    # Base64url without padding per JWT spec.
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def decode_jwt_claims_unverified(token: str) -> Dict[str, Any]:
    """Decodes JWT claims WITHOUT verifying signature/issuer/audience.

    Use only to enrich context/logging. Do not use for authorization decisions.
    """
    # JWT: header.payload.signature
    parts = token.split(".")
    if len(parts) < 2:
        return {}

    payload_b64 = parts[1]
    try:
        payload = _b64url_decode(payload_b64)
        claims = json.loads(payload.decode("utf-8"))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def extract_user_fields_from_claims(claims: Dict[str, Any]) -> Dict[str, str]:
    """Extracts a small, stable user identity subset from Entra ID-style claims."""
    user_id = claims.get("oid") or claims.get("sub")
    tenant_id = claims.get("tid")
    user_name = (
        claims.get("preferred_username")
        or claims.get("upn")
        or claims.get("name")
        or claims.get("email")
    )

    out: Dict[str, str] = {}
    if isinstance(user_id, str) and user_id.strip():
        out["user_id"] = user_id.strip()
    if isinstance(tenant_id, str) and tenant_id.strip():
        out["tenant_id"] = tenant_id.strip()
    if isinstance(user_name, str) and user_name.strip():
        out["user_name"] = user_name.strip()

    return out
