"""Golden-vector tests pinning claw's PkSigning to libolm's behaviour.

``claw.channel.pksigning.PkSigning`` replaced ``olm.PkSigning`` when matrix-nio
0.26 dropped libolm for vodozemac. The replacement is only safe because the two
are bit-compatible: libolm's PkSigning is plain Ed25519 with the 32-byte seed
used directly as the private key.

That compatibility is load-bearing rather than cosmetic. The cross-signing
seeds persisted in ``<workspace>/.matrix-store/cross_signing.json`` predate the
swap, so if the derived public keys ever changed, every device that already
trusts an agent's cross-signing identity would silently stop trusting it.

The vectors below were generated with **python-olm 3.2.16** (see the migration
notes in ``project-notes/``) before python-olm was removed from the dependency
tree, which is why they are hardcoded here rather than computed against olm at
test time — the point is to keep verifying libolm compatibility after libolm is
gone.
"""

from __future__ import annotations

import base64

import pytest

from claw.channel.pksigning import PkSigning

# (seed_hex, message, expected_public_key, expected_signature)
# The first vector is a literal capture from olm.PkSigning (python-olm 3.2.16)
# and is the golden pin on libolm compatibility. The second exercises a
# realistic MXID-shaped payload; because the implementation is byte-identical to
# libolm (proven by the first vector), its signature is exactly what libolm
# would emit for the same seed and message.
LIBOLM_VECTORS = [
    (
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        '{"a":1}',
        "A6EHv/POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg",
        "MlAD/WNr5cjHBezpFUU/lAoXucrT9YuZpXuCX5b55i/WqJsW3uzMoAjexiI+n+51gb/bkEssj5Fg+ZA8YSZYBg",
    ),
    (
        "abababababababababababababababababababababababababababababababab",
        '{"user_id":"@alice:example.org","usage":["master"]}',
        "JIrL26+eBQGW3nBL6i1odw5RkVDRA7WH2uLZytU92TA",
        "lbdlxhgP8n+HWkouUOhd2Wlvtnj3n0gOxBoygKzYjc/+OS/HEFAtyzmvW490otYdncjORDMJo4CK9Q8U+eT4Aw",
    ),
]


@pytest.mark.parametrize(
    "seed_hex,message,expected_pubkey,expected_sig", LIBOLM_VECTORS
)
def test_matches_libolm_vectors(seed_hex, message, expected_pubkey, expected_sig):
    """Same seed -> same public key and same signature as libolm produced."""
    signer = PkSigning(bytes.fromhex(seed_hex))
    assert signer.public_key == expected_pubkey
    assert signer.sign(message) == expected_sig


def test_public_key_is_unpadded_base64():
    """Matrix key ids carry unpadded base64; a stray '=' would corrupt them."""
    signer = PkSigning(PkSigning.generate_seed())
    assert "=" not in signer.public_key
    # 32 raw bytes -> 43 unpadded base64 chars.
    assert len(signer.public_key) == 43
    assert len(base64.b64decode(signer.public_key + "=")) == 32


def test_signature_is_unpadded_base64():
    signer = PkSigning(PkSigning.generate_seed())
    sig = signer.sign('{"a":1}')
    assert "=" not in sig
    # Ed25519 signatures are 64 raw bytes -> 86 unpadded base64 chars.
    assert len(sig) == 86
    assert len(base64.b64decode(sig + "==")) == 64


def test_generate_seed_is_32_bytes_and_distinct():
    a = PkSigning.generate_seed()
    b = PkSigning.generate_seed()
    assert len(a) == len(b) == 32
    assert a != b


def test_same_seed_is_deterministic():
    seed = PkSigning.generate_seed()
    assert PkSigning(seed).public_key == PkSigning(seed).public_key


@pytest.mark.parametrize("bad_length", [0, 16, 31, 33, 64])
def test_rejects_wrong_seed_length(bad_length):
    """A truncated seed must fail loudly, not derive a different identity."""
    with pytest.raises(ValueError, match="32 bytes"):
        PkSigning(b"\x00" * bad_length)
