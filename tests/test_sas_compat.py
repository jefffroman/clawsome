"""Tests for the SAS spec-compliance fixes.

Two independent nio defects kept in-room verification from ever completing
against Element:

* the commitment was hex-encoded where the spec requires unpadded base64, so
  the peer's check against our ephemeral key could never match — verification
  died right after the key exchange, before any MAC;
* the MAC method was pinned to legacy ``hkdf-hmac-sha256`` while MACs were
  computed with vodozemac's corrected base64. Preferring ``.v2`` (which the
  spec mandates when both sides support it) makes that correct by
  construction; ``apply_legacy_mac_compat`` is the fallback for v1-only peers.

All of these are bugs about *bytes on the wire*, so the tests assert against
real vodozemac output and against the actual hex commitment captured from the
failing 2026-08-07 exchange, rather than against mocks.
"""

from __future__ import annotations

import pytest

from claw.channel.sas_compat import (
    LEGACY_MAC_METHOD,
    MAC_METHOD_V2,
    _LegacyBase64Mac,
    apply_legacy_mac_compat,
    fix_accept_commitment,
    prefer_mac_v2,
    spec_commitment,
)

vodozemac = pytest.importorskip("vodozemac", reason="E2E extra not installed")


class FakeSas:
    """Stand-in for nio's Sas: a plain object with the two attributes used."""

    def __init__(self, established, method=LEGACY_MAC_METHOD):
        self.established_sas = established
        self.chosen_mac_method = method


@pytest.fixture
def established():
    """A real established vodozemac SAS (needs a peer to complete the DH)."""
    ours, theirs = vodozemac.Sas(), vodozemac.Sas()
    return ours.diffie_hellman(theirs.public_key)


def test_shim_produces_libolm_encoding_not_vodozemac_default(established):
    """The whole point: the wire bytes must change to libolm's encoding."""
    sas = FakeSas(established)
    proper = established.calculate_mac("KEY", "INFO")
    legacy = established.calculate_mac_invalid_base64("KEY", "INFO")
    # Guard the premise — if these ever coincide the shim is pointless.
    assert proper != legacy

    assert apply_legacy_mac_compat(sas) is True
    assert sas.established_sas.calculate_mac("KEY", "INFO") == legacy
    assert sas.established_sas.calculate_mac("KEY", "INFO") != proper


def test_applies_to_every_nio_code_path(established):
    """nio calls calculate_mac for its own key, the peer's, and KEY_IDS."""
    sas = FakeSas(established)
    apply_legacy_mac_compat(sas)
    for inp, info in (
        ("ed25519:DEVICE", "MATRIX_KEY_VERIFICATION_MAC…ed25519:DEVICE"),
        ("ed25519:DEVICE", "MATRIX_KEY_VERIFICATION_MAC…KEY_IDS"),
        ("someMasterPubKey", "MATRIX_KEY_VERIFICATION_MAC…master"),
    ):
        assert sas.established_sas.calculate_mac(inp, info) == \
            established.calculate_mac_invalid_base64(inp, info)


def test_idempotent_does_not_double_wrap(established):
    sas = FakeSas(established)
    assert apply_legacy_mac_compat(sas) is True
    first = sas.established_sas
    assert apply_legacy_mac_compat(sas) is True
    assert sas.established_sas is first
    assert not isinstance(first._inner, _LegacyBase64Mac)


def test_skipped_when_a_corrected_method_is_negotiated(established):
    """If nio ever learns v2, the shim must step aside, not corrupt it."""
    sas = FakeSas(established, method="hkdf-hmac-sha256.v2")
    assert apply_legacy_mac_compat(sas) is False
    assert sas.established_sas is established


def test_skipped_before_key_exchange():
    """No established SAS yet -> nothing to wrap, and no crash."""
    sas = FakeSas(None)
    assert apply_legacy_mac_compat(sas) is False
    assert sas.established_sas is None


def test_skipped_when_method_not_yet_negotiated(established):
    sas = FakeSas(established, method="")
    assert apply_legacy_mac_compat(sas) is False
    assert sas.established_sas is established


def test_non_mac_attributes_pass_through(established):
    """verify_mac/bytes must keep working — only calculate_mac is redirected."""
    sas = FakeSas(established)
    apply_legacy_mac_compat(sas)
    assert sas.established_sas.bytes is not None
    assert callable(sas.established_sas.verify_mac)


def test_degrades_gracefully_without_the_legacy_primitive():
    """A vodozemac lacking the primitive must be left alone, not crash."""

    class NoLegacy:
        chosen_mac_method = LEGACY_MAC_METHOD

        def calculate_mac(self, input, info):
            return "x"

    inner = NoLegacy()
    sas = FakeSas(inner)
    assert apply_legacy_mac_compat(sas) is False
    assert sas.established_sas is inner


# --- commitment encoding ---------------------------------------------------

# A real m.key.verification.start content as Element X sends it in-room.
ELEMENT_START = {
    "from_device": "CJHVIJWPVJ",
    "hashes": ["sha256"],
    "key_agreement_protocols": ["curve25519-hkdf-sha256"],
    "m.relates_to": {"event_id": "$abc", "rel_type": "m.reference"},
    "message_authentication_codes": [
        "hkdf-hmac-sha256", "hkdf-hmac-sha256.v2",
        "org.matrix.msc3783.hkdf-hmac-sha256",
    ],
    "method": "m.sas.v1",
    "short_authentication_string": ["decimal", "emoji"],
}

# The exact hex commitment claw put on the wire on 2026-08-07, which Element
# could never match.
OBSERVED_HEX = "9150499278a80db3c892a539770e6c276fe14131e0292385916fbfd99b8b4f72"


class CommitSas:
    def __init__(self, pubkey="somePubKey", commitment=None):
        self.pubkey = pubkey
        self.commitment = commitment


def test_commitment_is_unpadded_base64_not_hex():
    c = spec_commitment("somePubKey", ELEMENT_START)
    # 32 raw bytes -> 43 unpadded base64 chars (a hex digest would be 64).
    assert len(c) == 43
    assert "=" not in c
    assert not all(ch in "0123456789abcdef" for ch in c)


def test_commitment_is_deterministic_and_input_sensitive():
    a = spec_commitment("keyA", ELEMENT_START)
    assert a == spec_commitment("keyA", ELEMENT_START)
    assert a != spec_commitment("keyB", ELEMENT_START)
    other = dict(ELEMENT_START, method="m.sas.v2")
    assert a != spec_commitment("keyA", other)


def test_fix_rewrites_the_observed_hex_commitment():
    sas = CommitSas(commitment=OBSERVED_HEX)
    content = {"commitment": OBSERVED_HEX}
    assert fix_accept_commitment(sas, ELEMENT_START, content) is True
    assert content["commitment"] != OBSERVED_HEX
    assert content["commitment"] == spec_commitment(sas.pubkey, ELEMENT_START)
    # The Sas object must agree with what we actually sent.
    assert sas.commitment == content["commitment"]


def test_fix_is_noop_once_upstream_encodes_correctly():
    """A future matrix-nio that emits base64 must be left alone."""
    good = spec_commitment("somePubKey", ELEMENT_START)
    sas = CommitSas(commitment=good)
    content = {"commitment": good}
    assert fix_accept_commitment(sas, ELEMENT_START, content) is False
    assert content["commitment"] == good


def test_fix_handles_missing_commitment():
    assert fix_accept_commitment(CommitSas(), ELEMENT_START, {}) is False


# --- MAC method negotiation ------------------------------------------------

class MethodSas:
    def __init__(self):
        self.chosen_mac_method = LEGACY_MAC_METHOD


def test_prefers_v2_when_peer_offers_it():
    sas = MethodSas()
    content = {"message_authentication_code": LEGACY_MAC_METHOD}
    assert prefer_mac_v2(sas, ELEMENT_START["message_authentication_codes"], content) is True
    assert content["message_authentication_code"] == MAC_METHOD_V2
    assert sas.chosen_mac_method == MAC_METHOD_V2


def test_keeps_v1_when_peer_only_offers_v1():
    sas = MethodSas()
    content = {"message_authentication_code": LEGACY_MAC_METHOD}
    assert prefer_mac_v2(sas, ["hkdf-hmac-sha256"], content) is False
    assert content["message_authentication_code"] == LEGACY_MAC_METHOD
    assert sas.chosen_mac_method == LEGACY_MAC_METHOD


def test_v2_negotiation_disengages_the_legacy_shim(established):
    """The two fixes must not both apply — v2 needs vodozemac's default."""
    sas = MethodSas()
    sas.established_sas = established
    content = {"message_authentication_code": LEGACY_MAC_METHOD}
    prefer_mac_v2(sas, ELEMENT_START["message_authentication_codes"], content)
    assert apply_legacy_mac_compat(sas) is False
    assert sas.established_sas is established
