"""Ed25519 signing for Matrix cross-signing keys.

Drop-in replacement for ``olm.PkSigning``, which is the only thing that kept
python-olm in the dependency tree once matrix-nio moved its E2E backend from
libolm to vodozemac in 0.26.0. vodozemac exposes ``Ed25519PublicKey`` and
``Ed25519Signature`` but no seed-driven signing primitive, so cross-signing
needs its own.

libolm's PkSigning is plain Ed25519 with the 32-byte seed used directly as the
private key, so `cryptography`'s Ed25519 is bit-compatible with it. Verified
against python-olm 3.2.16 before the swap: for the same seed, both the derived
public key and the signature over the same message are byte-identical (see
``tests/test_pksigning.py``). That compatibility is the whole point — the
persisted seeds in ``<workspace>/.matrix-store/cross_signing.json`` must keep
deriving the *same* public keys, or every device that already trusts this
user's cross-signing identity would break.

`cryptography` is installed globally by Homebrew and inherited by the claw
venv through --system-site-packages, so this adds no new dependency.

Encoding note: Matrix (and libolm) use unpadded standard base64 for keys and
signatures, hence the ``rstrip("=")``.
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

__all__ = ["PkSigning"]

_SEED_LENGTH = 32


def _unpadded_b64(data: bytes) -> str:
    """Standard base64 with the ``=`` padding stripped, as Matrix expects."""
    return base64.b64encode(data).decode("ascii").rstrip("=")


class PkSigning:
    """Sign Matrix canonical JSON with an Ed25519 key derived from a seed.

    Mirrors the subset of ``olm.PkSigning`` that claw uses: construction from
    a seed, the ``public_key`` property, ``sign()``, and ``generate_seed()``.
    """

    __slots__ = ("_key", "_public_key")

    def __init__(self, seed: bytes) -> None:
        if len(seed) != _SEED_LENGTH:
            raise ValueError(
                f"cross-signing seed must be {_SEED_LENGTH} bytes, got {len(seed)}"
            )
        self._key = Ed25519PrivateKey.from_private_bytes(seed)
        self._public_key = _unpadded_b64(
            self._key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )

    @staticmethod
    def generate_seed() -> bytes:
        """A fresh 32-byte seed. Callers persist this hex-encoded."""
        return os.urandom(_SEED_LENGTH)

    @property
    def public_key(self) -> str:
        """Unpadded-base64 Ed25519 public key, as it appears in a key id."""
        return self._public_key

    def sign(self, message: str) -> str:
        """Sign ``message`` (already canonical JSON), returning unpadded b64."""
        return _unpadded_b64(self._key.sign(message.encode("utf-8")))
