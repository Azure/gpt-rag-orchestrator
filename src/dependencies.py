"""
Provides dependencies for API calls.
"""
import logging
import os
import json
import httpx
import hmac
import hashlib
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from connectors.appconfig import AppConfigClient
import jwt

__config: AppConfigClient = None
__cached_public_keys = {}  # {cache_key: {"keys": {...}, "expires_at": datetime}}


def _log_app_config_state(cfg: AppConfigClient, keys_to_check: Optional[List[str]] = None, prefix: str = "[Auth]") -> None:
    """Logs a safe, high-signal snapshot of App Configuration state.

    Intended for troubleshooting missing config keys in production without leaking secrets.
    """
    keys_to_check = keys_to_check or []

    endpoint = os.getenv("APP_CONFIG_ENDPOINT")
    endpoint_set = bool(endpoint and str(endpoint).strip())
    try:
        endpoint_host = endpoint.replace("https://", "").replace("http://", "").split("/")[0] if endpoint else None
    except Exception:
        endpoint_host = None

    disabled = bool(getattr(cfg, "disabled", False))
    auth_failed = bool(getattr(cfg, "auth_failed", False))
    allow_env_vars = bool(getattr(cfg, "allow_env_vars", False))

    client_dict = getattr(cfg, "client", None) or {}
    keys_loaded = len(client_dict) if isinstance(client_dict, dict) else 0

    # Presence only (no values)
    presence = {}
    if isinstance(client_dict, dict):
        for k in keys_to_check:
            presence[k] = "present" if k in client_dict else "missing"

    labels = "orchestrator,gpt-rag-orchestrator,gpt-rag,<no-label>"
    logging.warning(
        "%s AppConfig state: endpoint_set=%s endpoint_host=%s disabled=%s auth_failed=%s allow_env_vars=%s keys_loaded=%d labels=%s key_presence=%s",
        prefix,
        endpoint_set,
        endpoint_host,
        disabled,
        auth_failed,
        allow_env_vars,
        keys_loaded,
        labels,
        presence if presence else "{}",
    )


def _truncate(value: Optional[object], max_len: int = 160) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _safe_claims_snapshot(claims: Dict, include_pii: bool = True) -> Dict[str, object]:
    """Return a small allowlist of JWT claims suitable for logs.

    NOTE: This is still user data. Keep it at DEBUG unless explicitly needed.
    """
    if not isinstance(claims, dict):
        return {}

    out: Dict[str, object] = {}
    # Tenant / user identity
    for k in ["tid", "oid", "sub"]:
        if k in claims:
            out[k] = _truncate(claims.get(k))

    # App identity
    for k in ["appid", "azp", "azpacr"]:
        if k in claims:
            out[k] = _truncate(claims.get(k))

    # Permissions
    if "scp" in claims:
        out["scp"] = _truncate(claims.get("scp"))
    if "roles" in claims:
        roles = claims.get("roles")
        if isinstance(roles, list):
            out["roles"] = [_truncate(r, 80) for r in roles][:20]
        else:
            out["roles"] = _truncate(roles)

    # Groups (do not log group IDs)
    groups = claims.get("groups")
    if isinstance(groups, list):
        out["groups"] = f"<count:{len(groups)}>"
    elif groups is not None:
        out["groups"] = "<present>"

    # Group overage indicators (names only)
    if "hasgroups" in claims:
        out["hasgroups"] = claims.get("hasgroups")
    claim_names = claims.get("_claim_names")
    if isinstance(claim_names, dict):
        # Values can include sources/ids; keep keys only.
        keys = list(claim_names.keys())
        out["_claim_names"] = keys[:20]

    # Issuer / audience (helpful for troubleshooting)
    for k in ["iss", "aud"]:
        if k in claims:
            out[k] = _truncate(claims.get(k))

    if include_pii:
        for k in ["preferred_username", "upn", "email", "name"]:
            if k in claims:
                out[k] = _truncate(claims.get(k))

    return out

def _parse_cache_control_ttl(cache_control_header: str) -> int:
    """Parse Cache-Control header to extract max-age in seconds."""
    if not cache_control_header:
        return 3600  # Default to 1 hour if no header
    
    for part in cache_control_header.split(","):
        part = part.strip()
        if part.startswith("max-age="):
            try:
                return int(part.split("=")[1])
            except (ValueError, IndexError):
                return 3600
    
    return 3600  # Default fallback

def get_config(action: str = None) -> AppConfigClient:
    global __config

    if action == "refresh":
        __config = AppConfigClient()
    elif __config is None:
        __config = AppConfigClient()

    return __config


def _normalize_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v or None

async def validate_auth(
    dapr_api_token: str = Header(None, alias="dapr-api-token"),
    x_api_key: str = Header(None, alias="X-API-KEY")
):
    """
    Authentication dependency (no authorization here):
    1) Prefer dapr-api-token if present; otherwise use X-API-KEY.
    2) Missing or invalid credentials => 401 Unauthorized.
    3) 403 Forbidden should be used only by downstream authorization checks (not here).
    """

    # 1) Check dapr-api-token first if provided
    provided_dapr = _normalize_token(dapr_api_token)
    if provided_dapr is not None:
        candidates = [
            _normalize_token(os.getenv("APP_API_TOKEN")),
            _normalize_token(os.getenv("DAPR_API_TOKEN")),
            "dev-token",
        ]
        candidates = [c for c in candidates if c]

        matched_source = None
        if candidates:
            if os.getenv("APP_API_TOKEN") and hmac.compare_digest(provided_dapr, _normalize_token(os.getenv("APP_API_TOKEN")) or ""):
                matched_source = "APP_API_TOKEN"
            elif os.getenv("DAPR_API_TOKEN") and hmac.compare_digest(provided_dapr, _normalize_token(os.getenv("DAPR_API_TOKEN")) or ""):
                matched_source = "DAPR_API_TOKEN"
            elif hmac.compare_digest(provided_dapr, "dev-token"):
                matched_source = "dev-token"

        logging.debug(
            "[Auth] Dapr token provided (len=%d) -> %s",
            len(provided_dapr),
            f"validated_via={matched_source}" if matched_source else "invalid",
        )

        if not matched_source:
            logging.warning("Invalid Dapr token")
            raise HTTPException(status_code=401, detail="Invalid Dapr token")
        return True

    # 2) Fallback to API key if no dapr token
    try:
        expected_api_key = get_config().get("ORCHESTRATOR_APP_APIKEY", default=os.getenv("ORCHESTRATOR_APP_APIKEY"))
    except Exception:
        expected_api_key = os.getenv("ORCHESTRATOR_APP_APIKEY")

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "[Auth] Using API key auth (X-API-KEY). expected_key=%s",
            "set" if expected_api_key else "missing",
        )

    if not x_api_key:
        # Missing credentials -> 401
        raise HTTPException(status_code=401, detail="Missing credentials. Provide dapr-api-token or X-API-KEY")

    if not expected_api_key or x_api_key != expected_api_key:
        logging.error("Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True

def _jwks_urls_for_tenant(tenant_id: str) -> Dict[str, str]:
    """Return both v1 and v2 JWKS endpoints for an Entra tenant."""
    return {
        "v1": f"https://login.microsoftonline.com/{tenant_id}/discovery/keys",
        "v2": f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
    }


async def _get_cached_public_keys(tenant_id: str, jwks_url: Optional[str] = None) -> Dict:
    """Fetch Azure AD public signing keys with smart caching based on Cache-Control header.

    Note: Some tokens use the AAD v1 issuer (sts.windows.net) and may require the v1 keyset endpoint.
    We cache per (tenant_id, jwks_url) to avoid mixing keysets.
    """
    global __cached_public_keys

    jwks_urls = _jwks_urls_for_tenant(tenant_id)
    effective_url = jwks_url or jwks_urls["v2"]
    cache_key = f"{tenant_id}|{effective_url}"
    
    # Check if we have valid cached keys
    if cache_key in __cached_public_keys:
        cached_data = __cached_public_keys[cache_key]
        if cached_data.get("expires_at") > datetime.now():
            logging.debug(
                "[Auth] Using cached JWKS (expires_at=%s url=%s)",
                cached_data.get("expires_at"),
                effective_url,
            )
            return cached_data["keys"]
        else:
            del __cached_public_keys[cache_key]
    
    # Fetch fresh keys from Azure AD
    async with httpx.AsyncClient() as client:
        response = await client.get(effective_url, timeout=10)
        response.raise_for_status()
        keys_response = response.json()
        
        # Extract TTL from Cache-Control header
        cache_control = response.headers.get("cache-control", "")
        ttl_seconds = _parse_cache_control_ttl(cache_control)
    
    # Cache with Azure's specified expiration time
    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    __cached_public_keys[cache_key] = {
        "keys": keys_response,
        "expires_at": expires_at
    }
    
    logging.debug("[Auth] Cached JWKS (TTL: %d seconds url=%s)", ttl_seconds, effective_url)
    return keys_response


def _force_refresh_jwks_cache(tenant_id: str, jwks_url: Optional[str] = None) -> None:
    try:
        jwks_urls = _jwks_urls_for_tenant(tenant_id)
        urls_to_clear = [jwks_url] if jwks_url else list(jwks_urls.values())
        cleared_any = False
        for url in urls_to_clear:
            if not url:
                continue
            cache_key = f"{tenant_id}|{url}"
            if cache_key in __cached_public_keys:
                del __cached_public_keys[cache_key]
                cleared_any = True
        if cleared_any:
            logging.debug("[Auth] Forced JWKS cache refresh for tenant")
    except Exception:
        # Never fail auth due to cache cleanup
        logging.debug("[Auth] Failed to clear JWKS cache", exc_info=True)

async def validate_access_token(token: str) -> Dict:
    """Validate access token and extract user info (oid, preferred_username, name)."""
    cfg = get_config()

    # Basic integrity / correlation without logging the token.
    token = (token or "").strip()
    token_len = len(token)
    token_parts = token.split(".") if token else []
    token_segments = len(token_parts)
    token_fp = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12] if token else "<empty>"
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "[Auth] Token received: len=%d segments=%d fp=%s",
            token_len,
            token_segments,
            token_fp,
        )

        try:
            seg_lens = [len(p) for p in token_parts]
            logging.debug("[Auth] Token segment lengths: %s", seg_lens)
        except Exception:
            logging.debug("[Auth] Failed to compute token segment lengths", exc_info=True)

    # We expect JWS: header.payload.signature (3 segments).
    # If this is a JWE (5 segments) or malformed, fail with a clearer message.
    if token_segments != 3:
        logging.warning(
            "[Auth] Unsupported token format (segments=%d). Expected a signed JWT (JWS) with 3 segments.",
            token_segments,
        )
        raise HTTPException(status_code=401, detail="Invalid token")

    # Additional integrity evidence: ensure each segment is base64url-like and decodable.
    # This helps catch cases where some proxy/client code URL-encodes or mutates the token.
    try:
        import base64
        import re

        b64url_re = re.compile(r"^[A-Za-z0-9_-]+$")

        def _b64url_decode_len(segment: str) -> int:
            padded = segment + "=" * (-len(segment) % 4)
            return len(base64.urlsafe_b64decode(padded.encode("utf-8")))

        header_seg, payload_seg, sig_seg = token_parts[0], token_parts[1], token_parts[2]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "[Auth] Token segments base64url chars: header=%s payload=%s signature=%s",
                bool(b64url_re.match(header_seg)),
                bool(b64url_re.match(payload_seg)),
                bool(b64url_re.match(sig_seg)),
            )

        # Decode header/payload for evidence. Signature decode length is helpful when troubleshooting corruption.
        try:
            _hdr_len = _b64url_decode_len(header_seg)
            _pl_len = _b64url_decode_len(payload_seg)
            _sig_len = _b64url_decode_len(sig_seg)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "[Auth] Token segment decoded byte lengths: header=%d payload=%d signature=%d",
                    _hdr_len,
                    _pl_len,
                    _sig_len,
                )
        except Exception:
            # If signature segment isn't decodable, that's a strong indicator the token was mutated.
            logging.warning(
                "[Auth] Token base64url decode failed for at least one segment (fp=%s). "
                "This often indicates the token was modified (URL-encoded, truncated, or had characters replaced) before reaching the orchestrator.",
                token_fp,
            )
    except Exception:
        # Never fail auth due to extra diagnostics.
        pass

    # NOTE:
    # AppConfigClient.get() raises when key is missing. For auth configuration we want:
    # - a deterministic 500 (server misconfiguration) when required settings are absent
    # - a clear log message for troubleshooting
    tenant_id = None
    try:
        tenant_id = cfg.get_value("OAUTH_AZURE_AD_TENANT_ID", default=None, allow_none=True)
    except Exception:
        tenant_id = None

    client_id = None
    try:
        client_id = cfg.get_value("OAUTH_AZURE_AD_CLIENT_ID", default=None, allow_none=True)
    except Exception:
        client_id = None

    if not tenant_id:
        _log_app_config_state(cfg, keys_to_check=["OAUTH_AZURE_AD_TENANT_ID"], prefix="[Auth]")
        logging.error(
            "[Auth] Missing tenant configuration in Azure App Configuration. "
            "Confirm: (1) APP_CONFIG_ENDPOINT is set for the orchestrator app, "
            "(2) Managed Identity has App Configuration Data Reader, "
            "(3) key exists under an included label (e.g. 'gpt-rag' / 'gpt-rag-orchestrator' / 'orchestrator')."
        )
        raise HTTPException(
            status_code=500,
            detail="Authentication not configured (missing OAUTH_AZURE_AD_TENANT_ID)",
        )
    
    # High-signal debug: which tenant/URLs are used to validate
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        jwks_url = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
        oidc_config_url = f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration"
        issuer_candidates = [
            f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            f"https://sts.windows.net/{tenant_id}/",
        ]
        expected_audiences = None
        if client_id:
            expected_audiences = [client_id, f"api://{client_id}"]
        logging.debug(
            "[Auth] Validating token: tenant_id=%s jwks_url=%s oidc_config_url=%s expected_audiences=%s",
            tenant_id,
            jwks_url,
            oidc_config_url,
            expected_audiences or "<not-configured>",
        )
        logging.debug("[Auth] Issuer candidates: %s", issuer_candidates)

    try:
        # Step 1: Get the token header to extract kid
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        alg = header.get("alg")
        x5t = header.get("x5t")

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "[Auth] Token header: alg=%s kid=%s x5t=%s",
                alg,
                kid,
                x5t,
            )

        if alg and alg != "RS256":
            logging.warning("[Auth] Unsupported token alg=%s (expected RS256)", alg)
            raise HTTPException(status_code=401, detail="Invalid token")
        
        if not kid and not x5t:
            logging.warning("Token missing kid in header")
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        logging.debug("[Auth] Token key id: %s", kid or x5t)
        
        async def _find_key_obj(jwks_url: Optional[str]) -> Dict | None:
            # Step 2: Fetch public keys from Azure AD
            keys_response = await _get_cached_public_keys(tenant_id, jwks_url=jwks_url)
            # Step 3: Find the key with matching kid/x5t
            keys = keys_response.get("keys", []) or []
            matches: list[Dict] = []
            for key in keys:
                if kid and key.get("kid") == kid:
                    matches.append(key)
                elif x5t and key.get("x5t") == x5t:
                    matches.append(key)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "[Auth] JWKS keys=%d matches=%d (kid=%s x5t=%s)",
                    len(keys),
                    len(matches),
                    kid,
                    x5t,
                )

            if not matches:
                return None

            # Prefer signing keys when metadata is present.
            sig_matches = [m for m in matches if (m.get("use") == "sig" or "verify" in (m.get("key_ops") or []))]
            if sig_matches:
                return sig_matches[0]

            # If no metadata, pick first match (best-effort).
            return matches[0]

        # Step 4: Inspect unverified claims (for diagnostics + to pick the correct JWKS endpoint).
        # We do NOT trust these claims for authorization decisions.
        unverified = jwt.decode(token, options={"verify_signature": False})
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            try:
                _tid = unverified.get("tid")
                _iss = unverified.get("iss")
                _aud = unverified.get("aud")
                _aud_type = type(_aud).__name__
                logging.debug(
                    "[Auth] Token claims (unverified): tid=%s iss=%s aud_type=%s aud=%s",
                    _tid,
                    _iss,
                    _aud_type,
                    _truncate(_aud, 180),
                )
            except Exception:
                logging.debug("[Auth] Failed to read unverified token claims for diagnostics", exc_info=True)

        # High-signal client misconfiguration hint.
        # Even though the claim is unverified, it's consistently useful to explain 401s caused by requesting Graph tokens.
        try:
            _aud_uv = unverified.get("aud")
            graph_aud = "00000003-0000-0000-c000-000000000000"
            aud_is_graph = _aud_uv == graph_aud or (isinstance(_aud_uv, list) and graph_aud in _aud_uv)
            if aud_is_graph:
                if client_id:
                    logging.warning(
                        "[Auth] Incoming token audience indicates Microsoft Graph (aud=%s fp=%s). "
                        "Frontend must request an access token for this API scope: api://%s/user_impersonation (not Graph scopes like User.Read).",
                        graph_aud,
                        token_fp,
                        client_id,
                    )
                else:
                    logging.warning(
                        "[Auth] Incoming token audience indicates Microsoft Graph (aud=%s fp=%s). "
                        "Frontend must request an access token for the orchestrator API (Expose an API scope), not Graph scopes like User.Read.",
                        graph_aud,
                        token_fp,
                    )
        except Exception:
            pass

        # Extra guardrail: log tid mismatch clearly (usually indicates wrong tenant config).
        token_tid = unverified.get("tid")
        if token_tid and token_tid != tenant_id:
            logging.warning(
                "[Auth] Token tenant mismatch: token_tid=%s configured_tenant_id=%s",
                token_tid,
                tenant_id,
            )
            raise HTTPException(status_code=401, detail="Invalid token")

        jwks_urls = _jwks_urls_for_tenant(tenant_id)
        issuer_candidates = [
            f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            f"https://sts.windows.net/{tenant_id}/",
        ]

        expected_audiences: list[str] | None = None
        if client_id:
            expected_audiences = [client_id, f"api://{client_id}"]

        # Decide which JWKS endpoint should be used based on the token issuer.
        token_iss = unverified.get("iss")
        primary_jwks_url = jwks_urls["v2"]
        if isinstance(token_iss, str) and token_iss.startswith("https://sts.windows.net/"):
            primary_jwks_url = jwks_urls["v1"]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("[Auth] Token issuer is v1; using JWKS url=%s", primary_jwks_url)

        async def _validate_with_jwks_url(jwks_url_to_use: str) -> tuple[dict | None, Exception | None]:
            key_obj_local = await _find_key_obj(jwks_url_to_use)
            if not key_obj_local:
                return None, KeyError("key-not-found")

            public_key_local = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_obj_local))

            last_error: Exception | None = None
            for issuer in issuer_candidates:
                try:
                    return (
                        jwt.decode(
                            token,
                            public_key_local,
                            algorithms=["RS256"],
                            options={
                                "verify_aud": bool(expected_audiences),
                                "verify_iss": True,
                            },
                            issuer=issuer,
                            audience=expected_audiences,
                        ),
                        None,
                    )
                except jwt.exceptions.InvalidIssuerError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    break
            return None, last_error

        decoded, last_err = await _validate_with_jwks_url(primary_jwks_url)

        # If signature fails, it might be a rotated key + stale cache: force refresh and retry once.
        if decoded is None and isinstance(last_err, jwt.exceptions.InvalidSignatureError):
            logging.warning("[Auth] Token signature validation failed; forcing JWKS refresh and retrying once")
            _force_refresh_jwks_cache(tenant_id)
            decoded, last_err = await _validate_with_jwks_url(primary_jwks_url)

        # If still failing, try the alternate JWKS endpoint (v1 vs v2) one last time.
        if decoded is None and isinstance(last_err, jwt.exceptions.InvalidSignatureError):
            alternate_jwks_url = jwks_urls["v1"] if primary_jwks_url == jwks_urls["v2"] else jwks_urls["v2"]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("[Auth] Retrying validation with alternate JWKS url=%s", alternate_jwks_url)
            _force_refresh_jwks_cache(tenant_id)
            decoded, last_err = await _validate_with_jwks_url(alternate_jwks_url)

        if decoded is None:
            if isinstance(last_err, KeyError):
                logging.warning("[Auth] Token key id not found in JWKS (kid=%s x5t=%s)", kid, x5t)
            elif isinstance(last_err, jwt.exceptions.InvalidAudienceError):
                logging.warning(
                    "[Auth] Token audience mismatch: expected=%s got=%s",
                    expected_audiences,
                    _truncate(unverified.get("aud"), 180),
                )
            elif isinstance(last_err, jwt.exceptions.InvalidIssuerError):
                logging.warning(
                    "[Auth] Token issuer mismatch. expected=%s got=%s",
                    issuer_candidates,
                    unverified.get("iss"),
                )
            elif isinstance(last_err, jwt.exceptions.InvalidSignatureError):
                logging.warning(
                    "[Auth] Token signature validation failed (kid=%s x5t=%s fp=%s)",
                    kid,
                    x5t,
                    token_fp,
                )

                # Helpful hint: when clients send a Graph token to this API, the audience is Graph.
                # This claim is unverified here (signature failed), but it's still useful for troubleshooting.
                try:
                    _unverified_aud = unverified.get("aud")
                    if _unverified_aud == "00000003-0000-0000-c000-000000000000":
                        if client_id:
                            logging.warning(
                                "[Auth] Token audience indicates Microsoft Graph (aud=%s). "
                                "Frontend must request an API access token scope like api://%s/user_impersonation instead of Graph scopes (e.g., User.Read).",
                                _unverified_aud,
                                client_id,
                            )
                        else:
                            logging.warning(
                                "[Auth] Token audience indicates Microsoft Graph (aud=%s). "
                                "Frontend must request an API access token (Expose an API scope) instead of Graph scopes (e.g., User.Read).",
                                _unverified_aud,
                            )
                except Exception:
                    pass
            else:
                logging.warning("[Auth] Token validation error: %s", type(last_err).__name__ if last_err else "Unknown")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_oid = decoded.get("oid")
        user_username = decoded.get("preferred_username")
        user_name = decoded.get("name")

        if not user_username:
            logging.debug(
                "[Auth] Token validated but preferred_username is missing (claims_present: upn=%s email=%s name=%s)",
                "set" if decoded.get("upn") else "missing",
                "set" if decoded.get("email") else "missing",
                "set" if user_name else "missing",
            )

        logging.info(
            "[Auth] User token validated (oid=%s preferred_username=%s)",
            user_oid,
            user_username or "<missing>",
        )

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            # Log a compact allowlist snapshot of claims for troubleshooting.
            # This includes user display fields (PII) at DEBUG level only.
            logging.debug(
                "[Auth] Token claims (verified allowlist): %s",
                _safe_claims_snapshot(decoded, include_pii=True),
            )
        
        return {
            "oid": user_oid,
            "preferred_username": user_username,
            "name": user_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error("[Auth] Token validation failed: %s: %s", type(e).__name__, str(e))
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_user_groups_from_graph(user_oid: str) -> List[str]:
    """Fetch user's group memberships from Microsoft Graph using app credentials."""
    cfg = get_config()

    # These settings are optional: if not configured, we skip group enrichment.
    # Use allow_none to avoid raising when keys are missing from App Configuration.
    client_id = None
    client_secret = None
    tenant_id = None
    try:
        client_id = cfg.get_value("OAUTH_AZURE_AD_CLIENT_ID", default=None, allow_none=True)
    except Exception:
        client_id = None
    if not client_id:
        try:
            client_id = cfg.get_value("CLIENT_ID", default=None, allow_none=True)
        except Exception:
            client_id = None

    try:
        client_secret = cfg.get_value("OAUTH_AZURE_AD_CLIENT_SECRET", default=None, allow_none=True)
    except Exception:
        client_secret = None

    try:
        tenant_id = cfg.get_value("OAUTH_AZURE_AD_TENANT_ID", default=None, allow_none=True)
    except Exception:
        tenant_id = None

    if not all([client_id, client_secret, tenant_id]):
        logging.warning("Graph API credentials not fully configured; skipping group lookup")
        return []
    
    try:
        # Get app token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": "https://graph.microsoft.com/.default",
                    "grant_type": "client_credentials"
                },
                timeout=10
            )
            token_response.raise_for_status()
            app_token = token_response.json().get("access_token")
        
        if not app_token:
            logging.warning("Failed to obtain app token for Graph API")
            return []
        
        # Get user groups
        headers = {"Authorization": f"Bearer {app_token}"}
        async with httpx.AsyncClient() as client:
            groups_response = await client.get(
                f"https://graph.microsoft.com/v1.0/users/{user_oid}/memberOf",
                headers=headers,
                timeout=10
            )
            groups_response.raise_for_status()
            groups_data = groups_response.json()
        
        groups = [g.get("displayName", "unknown-group") for g in groups_data.get("value", [])]
        logging.debug("[Auth] User groups: %s", groups)
        return groups
    except Exception as e:
        logging.warning("[Auth] Failed to retrieve groups from Graph API: %s", type(e).__name__)
        return []

def handle_exception(exception: Exception, status_code: int = 500):
    logging.error(exception, stack_info=True, exc_info=True)
    raise HTTPException(
        status_code=status_code,
        detail=str(exception)
    ) from exception
