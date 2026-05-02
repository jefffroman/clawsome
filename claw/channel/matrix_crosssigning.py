"""One-shot cross-signing bootstrap for matrix-nio bots.

matrix-nio 0.25 has no client-side cross-signing surface — bots that don't
publish a master signing key show up as "user verification unavailable" in
Element X (and as red-exclamation messages in Element Web). matrix-js-sdk
auto-bootstraps these on first run; we have to do it ourselves.

This module generates the standard three-key chain (master / self-signing /
user-signing) the first time, uploads them via
``/_matrix/client/v3/keys/device_signing/upload`` (with the UIA password
challenge), and signs the current device's identity key with the
self-signing key via ``/_matrix/client/v3/keys/signatures/upload``.

Idempotent: if the homeserver already has a ``master_keys`` entry for the
user, ``ensure_cross_signing`` returns ``False`` and skips. Seeds are
persisted at ``seed_store_path`` (mode 0600) so a homeserver re-bootstrap
reuses the same keys.

Olm/Curve25519/Ed25519 primitives come from ``python-olm`` (already
installed via ``matrix-nio[e2e]``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from olm import PkSigning

log = logging.getLogger("claw.channel.matrix_crosssigning")


def _canonical_json(obj: dict) -> str:
    """Matrix canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sign_with(obj: dict, signer: PkSigning) -> str:
    """Sign obj's canonical-JSON form using a PkSigning key.

    Strips ``signatures`` and ``unsigned`` per Matrix signing rules.
    """
    to_sign = {k: v for k, v in obj.items() if k not in ("signatures", "unsigned")}
    return signer.sign(_canonical_json(to_sign))


def _add_signature(obj: dict, user_id: str, key_id: str, sig: str) -> None:
    sigs = obj.setdefault("signatures", {})
    user_sigs = sigs.setdefault(user_id, {})
    user_sigs[key_id] = sig


async def ensure_cross_signing(
    *,
    homeserver: str,
    user_id: str,
    device_id: str,
    access_token: str,
    password: str,
    seed_store_path: Path,
    olm_account_sign: Callable[[str], str],
    force_replace: bool = False,
) -> bool:
    """Bootstrap cross-signing for ``user_id`` if not already present on the
    homeserver. Returns ``True`` if an upload happened, ``False`` if skipped.

    When ``force_replace`` is True, skip the idempotence check and upload new
    keys even if the homeserver already has master_keys for this user. The
    upload still goes through UIA, so a new master/SSK/USK triple replaces
    the existing ones. *Destructive* — prior device signatures by the old
    SSK become invalid, and Element prompts users to re-verify. Use only
    for one-shot account migrations where the original SSK private key
    isn't recoverable.

    ``olm_account_sign`` is the device's own Ed25519 sign function — passed
    in rather than imported because matrix-nio owns the OlmAccount and we
    want to anchor the master key with a device signature (matrix-js-sdk
    does this; Element trusts that pattern).
    """
    async with httpx.AsyncClient(base_url=homeserver, timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {access_token}"}

        # 1. Query keys (need the device info downstream regardless).
        resp = await client.post(
            "/_matrix/client/v3/keys/query",
            headers=headers,
            json={"device_keys": {user_id: []}},
        )
        resp.raise_for_status()
        keys_response = resp.json()
        existing_master = keys_response.get("master_keys", {}).get(user_id)
        if existing_master and not force_replace:
            log.info("[%s] cross-signing already published; skip", user_id)
            return False
        if existing_master and force_replace:
            log.warning(
                "[%s] force_replace=True — REPLACING existing cross-signing "
                "keys. Prior device signatures become invalid; users will "
                "need to re-verify in Element.",
                user_id,
            )

        # 2. Load or generate the three signing seeds.
        if seed_store_path.exists():
            seeds = json.loads(seed_store_path.read_text())
            log.info("[%s] reusing existing cross-signing seeds", user_id)
        else:
            seeds = {
                "master": PkSigning.generate_seed().hex(),
                "self_signing": PkSigning.generate_seed().hex(),
                "user_signing": PkSigning.generate_seed().hex(),
            }
            seed_store_path.parent.mkdir(parents=True, exist_ok=True)
            seed_store_path.write_text(json.dumps(seeds))
            seed_store_path.chmod(0o600)
            log.info("[%s] generated new cross-signing seeds", user_id)

        master = PkSigning(bytes.fromhex(seeds["master"]))
        ssk = PkSigning(bytes.fromhex(seeds["self_signing"]))
        usk = PkSigning(bytes.fromhex(seeds["user_signing"]))

        master_kid = f"ed25519:{master.public_key}"
        ssk_kid = f"ed25519:{ssk.public_key}"
        usk_kid = f"ed25519:{usk.public_key}"

        # 3. Build the three signed cross-signing key objects.
        master_obj: dict[str, Any] = {
            "user_id": user_id,
            "usage": ["master"],
            "keys": {master_kid: master.public_key},
        }
        # Master is self-signed.
        _add_signature(master_obj, user_id, master_kid, _sign_with(master_obj, master))
        # And device-signed (matrix-js-sdk's trust-anchor pattern; Element X
        # treats the device->master link as a strong "this device claims
        # this user identity" signal).
        _add_signature(
            master_obj, user_id, f"ed25519:{device_id}",
            olm_account_sign(_canonical_json(
                {k: v for k, v in master_obj.items() if k != "signatures"}
            )),
        )

        ssk_obj: dict[str, Any] = {
            "user_id": user_id,
            "usage": ["self_signing"],
            "keys": {ssk_kid: ssk.public_key},
        }
        _add_signature(ssk_obj, user_id, master_kid, _sign_with(ssk_obj, master))

        usk_obj: dict[str, Any] = {
            "user_id": user_id,
            "usage": ["user_signing"],
            "keys": {usk_kid: usk.public_key},
        }
        _add_signature(usk_obj, user_id, master_kid, _sign_with(usk_obj, master))

        # 4. Upload via /keys/device_signing/upload (UIA-gated).
        await _post_with_uia(
            client,
            "/_matrix/client/v3/keys/device_signing/upload",
            access_token,
            {
                "master_key": master_obj,
                "self_signing_key": ssk_obj,
                "user_signing_key": usk_obj,
            },
            user_id,
            password,
        )
        log.info("[%s] uploaded master / self_signing / user_signing keys", user_id)

        # 5. Sign the current device with the self-signing key.
        device_keys = (
            keys_response.get("device_keys", {}).get(user_id, {}).get(device_id)
        )
        if not device_keys:
            log.warning(
                "[%s] device %s not yet in /keys/query; cross-signing keys "
                "uploaded but device unsigned. Restart after first sync.",
                user_id, device_id,
            )
            return True

        device_obj = {
            "user_id": user_id,
            "device_id": device_id,
            "algorithms": device_keys["algorithms"],
            "keys": device_keys["keys"],
        }
        device_ssk_sig = _sign_with(device_obj, ssk)
        device_obj["signatures"] = dict(device_keys.get("signatures", {}))
        _add_signature(device_obj, user_id, ssk_kid, device_ssk_sig)

        resp = await client.post(
            "/_matrix/client/v3/keys/signatures/upload",
            headers=headers,
            json={user_id: {device_id: device_obj}},
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("failures"):
            log.warning(
                "[%s] signatures/upload returned failures: %s",
                user_id, result["failures"],
            )
        else:
            log.info("[%s] device %s cross-signed by self-signing key", user_id, device_id)
        return True


async def _post_with_uia(
    client: httpx.AsyncClient,
    path: str,
    access_token: str,
    body: dict,
    user_id: str,
    password: str,
) -> dict:
    """POST handling Synapse's m.login.password UIA challenge inline."""
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = await client.post(path, headers=headers, json=body)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code != 401:
        resp.raise_for_status()
    challenge = resp.json()
    session = challenge.get("session")
    if not session:
        raise RuntimeError(f"UIA challenge missing session: {challenge}")
    localpart = user_id.split(":", 1)[0].lstrip("@")
    auth_body = {
        **body,
        "auth": {
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": localpart},
            "password": password,
            "session": session,
        },
    }
    resp = await client.post(path, headers=headers, json=auth_body)
    resp.raise_for_status()
    return resp.json()
