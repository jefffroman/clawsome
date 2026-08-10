"""Spec-compliance fixes for matrix-nio's SAS verification.

matrix-nio's in-room SAS never completed against Element. Two independent
defects, both in what nio puts on the wire:

1. **The commitment is hex-encoded.** ``Sas.from_key_verification_start``
   builds it with ``sha256(...).hexdigest()``. The spec requires the hash
   *unpadded base64* — a 43-character string, not 64 hex characters. The peer
   stores our commitment and, on receiving our ephemeral key, checks it against
   its own recomputation; a hex string can never match, so verification dies
   right after the key exchange, before any MAC is exchanged. This one is not
   new in 0.26 — it explains why these bots were never verifiable by anyone.

2. **The MAC method is pinned to the legacy one.** ``accept_verification``
   hardcodes ``chosen_mac_method = "hkdf-hmac-sha256"``, but computes MACs with
   vodozemac's *corrected* base64 (``EstablishedSas.calculate_mac``). Legacy
   ``hkdf-hmac-sha256`` is defined as libolm did it — with libolm's incorrect
   base64, which is why vodozemac still ships
   ``calculate_mac_invalid_base64``. So nio agrees to v1 and sends v2 bytes.
   The spec is explicit: "if both parties support ``hkdf-hmac-sha256.v2``,
   then ``hkdf-hmac-sha256`` MUST not be used."

The fixes mirror that split:

* :func:`spec_commitment` / :func:`fix_accept_commitment` re-encode the
  commitment correctly.
* :func:`prefer_mac_v2` upgrades the negotiated method to v2 whenever the peer
  offers it, which makes nio's own computation correct by construction.
* :func:`apply_legacy_mac_compat` remains the fallback for a peer that only
  offers v1: it routes MAC computation through the libolm-compatible encoding
  so v1 means what v1 means. It deliberately no-ops once v2 is negotiated.

Only the accepting side is patched here, which is all claw ever is — it never
initiates a verification. nio's ``_check_commitment`` (used by the *initiator*)
has the same hex bug and is left alone.
"""

from __future__ import annotations

import base64
import hashlib
import logging

from nio.api import Api

__all__ = [
    "LEGACY_MAC_METHOD",
    "MAC_METHOD_V2",
    "apply_legacy_mac_compat",
    "fix_accept_commitment",
    "prefer_mac_v2",
    "spec_commitment",
]

log = logging.getLogger("claw.channel.sas_compat")

#: Legacy MAC method; mandates libolm's incorrect base64.
LEGACY_MAC_METHOD = "hkdf-hmac-sha256"
#: Corrected MAC method. Preferred whenever the peer offers it.
MAC_METHOD_V2 = "hkdf-hmac-sha256.v2"


def _unpadded_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii").rstrip("=")


def spec_commitment(pubkey: str, start_content: dict) -> str:
    """The commitment as the spec defines it, in unpadded base64.

    SHA-256 over our ephemeral public key concatenated with the canonical JSON
    of the peer's ``m.key.verification.start`` content — the same inputs nio
    uses, differing only in the final encoding.
    """
    digest = hashlib.sha256(
        pubkey.encode() + Api.to_canonical_json(start_content).encode()
    ).digest()
    return _unpadded_b64(digest)


def fix_accept_commitment(sas: object, start_content: dict, accept_content: dict) -> bool:
    """Rewrite a hex commitment in ``accept_content`` to unpadded base64.

    Returns ``True`` if it was rewritten. A no-op if the commitment already
    looks correctly encoded, so a future matrix-nio that fixes this upstream
    silently takes over.
    """
    current = accept_content.get("commitment")
    if not isinstance(current, str) or not current:
        return False

    # A hex digest is 64 chars from [0-9a-f]; the correct form is 43 chars of
    # base64. Anything already non-hex is presumed correct — don't touch it.
    is_hex = len(current) == 64 and all(c in "0123456789abcdef" for c in current)
    if not is_hex:
        return False

    pubkey = getattr(sas, "pubkey", None)
    if not pubkey:
        log.warning("SAS: no ephemeral pubkey on Sas; cannot fix commitment")
        return False

    fixed = spec_commitment(pubkey, start_content)
    accept_content["commitment"] = fixed
    # Keep the Sas object consistent with what we actually sent.
    try:
        sas.commitment = fixed
    except Exception:  # pragma: no cover - plain attribute on nio's Sas
        pass
    log.info("SAS: re-encoded commitment hex -> unpadded base64 (%s…)", fixed[:12])
    return True


def prefer_mac_v2(sas: object, offered: list, accept_content: dict) -> bool:
    """Negotiate ``hkdf-hmac-sha256.v2`` when the peer supports it.

    nio pins the legacy method; the spec forbids using it when both sides can
    do v2. Returns ``True`` if the accept was upgraded to v2.
    """
    if MAC_METHOD_V2 not in (offered or []):
        return False
    if accept_content.get("message_authentication_code") == MAC_METHOD_V2:
        return True

    accept_content["message_authentication_code"] = MAC_METHOD_V2
    try:
        sas.chosen_mac_method = MAC_METHOD_V2
    except Exception:  # pragma: no cover
        return False
    log.info("SAS: negotiated %s (peer supports it; legacy v1 MUST NOT be used)",
             MAC_METHOD_V2)
    return True


class _LegacyBase64Mac:
    """Wraps a vodozemac ``EstablishedSas``, forcing libolm's MAC encoding.

    Only ``calculate_mac`` is redirected; everything else passes through. The
    wrapped object is a Rust extension type with read-only attributes, so
    wrapping is the only way to intercept the call.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def calculate_mac(self, input: str, info: str) -> str:
        """MAC in libolm's (spec-mandated for v1) base64 encoding."""
        return self._inner.calculate_mac_invalid_base64(input, info)

    def calculate_mac_invalid_base64(self, input: str, info: str) -> str:
        return self._inner.calculate_mac_invalid_base64(input, info)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def apply_legacy_mac_compat(sas: object) -> bool:
    """Force ``sas`` to compute MACs the way its negotiated method requires.

    Idempotent, and safe to call before the MAC exchange on any Sas object.
    Returns ``True`` if the shim is in place (or already was), ``False`` if it
    was deliberately skipped — no established SAS yet, or a MAC method other
    than the legacy one was negotiated (notably v2, which vodozemac's default
    ``calculate_mac`` already implements correctly).
    """
    established = getattr(sas, "established_sas", None)
    if established is None:
        return False

    if isinstance(established, _LegacyBase64Mac):
        return True

    method = getattr(sas, "chosen_mac_method", "") or ""
    if method != LEGACY_MAC_METHOD:
        log.debug(
            "SAS mac method %r is not %r — leaving nio's MAC computation alone",
            method, LEGACY_MAC_METHOD,
        )
        return False

    if not hasattr(established, "calculate_mac_invalid_base64"):
        log.warning(
            "vodozemac EstablishedSas has no calculate_mac_invalid_base64; "
            "cannot apply legacy MAC compatibility (SAS will likely fail)"
        )
        return False

    sas.established_sas = _LegacyBase64Mac(established)
    log.info("SAS: applied libolm-compatible MAC encoding for method %r", method)
    return True
