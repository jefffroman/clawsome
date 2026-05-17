"""matrix-nio per-account Matrix client.

One ``MatrixChannel`` per agent account. matrix-nio's ``AsyncClient`` handles
E2E transparently when ``encryption_enabled=True`` and ``store_path`` points
at a writable directory.

Auth is non-interactive: tokens are generated out-of-band via the Synapse
admin API and written to ``access_token_file`` (one line, mode 0600).
``restore_login`` re-attaches without re-fetching keys; ``sync_forever``
then drives the conversation.

DM rooms (≤2 members): drop messages from non-allowlisted senders silently.
Group rooms: bot replies only when ``@``-mentioned (``allow_bots: mentions``).
Auto-join: any invite is accepted (``auto_join: always``); paired with the
out-of-band invite control from Element/etc.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncContextManager

from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    KeyVerificationCancel,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    LocalProtocolError,
    MatrixRoom,
    RoomMessageText,
    RoomMessageUnknown,
)
import json as _json
from olm import PkSigning
from nio.crypto import Sas
from nio.crypto.sas import SasState
from nio.events.room_events import UnknownEvent
from markdown_it import MarkdownIt

from claw.channel.base import InboundHandler, InboundMessage
from claw.config import MatrixAccountConfig

log = logging.getLogger("claw.channel.matrix")

# Matrix has no hard message size cap, but very long messages render
# awkwardly on most clients. Soft-cap with paragraph-boundary chunking.
MATRIX_CHUNK_MAX = 16000

# CommonMark + tables + strikethrough + linkify. `breaks=True` turns single
# newlines into <br> so chat-style line-wrapped LLM output renders the way
# it reads in the source. `html=False` (the default) escapes any literal
# `<tag>` the model emits instead of executing it.
_MD = (
    MarkdownIt("commonmark", {"linkify": True, "breaks": True})
    .enable(["table", "strikethrough", "linkify"])
)


def _to_html(text: str) -> str:
    return _MD.render(text)

# Typing indicator must be renewed periodically; 6s leaves headroom inside
# Matrix's 8s typing timeout.
TYPING_RENEW_INTERVAL_S = 6.0


class MatrixChannel:
    name = "matrix"

    def __init__(self, account: MatrixAccountConfig) -> None:
        self.account = account
        self.user_id = account.user_id
        self._client: AsyncClient | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._on_message: InboundHandler | None = None
        self._allow_from = set(account.allow_from)
        self._own_localpart = account.user_id.split(":", 1)[0].lstrip("@").lower()
        # In-room SAS verifications, keyed by request event id (which serves
        # as the transaction id under MSC2241).
        self._inroom_sas: dict[str, tuple[Sas, str]] = {}

    async def start(self, on_message: InboundHandler) -> None:
        self._on_message = on_message
        token = self._read_token(self.account.access_token_file)
        store_path = Path(self.account.store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        config = AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=self.account.encryption,
        )
        self._client = AsyncClient(
            homeserver=self.account.homeserver,
            user=self.user_id,
            device_id=self.account.device_id,
            store_path=str(store_path),
            config=config,
        )
        self._client.restore_login(
            user_id=self.user_id,
            device_id=self.account.device_id,
            access_token=token,
        )

        # Push device keys synchronously so cross-signing bootstrap can find
        # the device on /keys/query. (sync_forever would do this on its first
        # iteration, but we want to bootstrap before going live.) On warm
        # boots matrix-nio raises LocalProtocolError("No key upload needed")
        # because the cached store already has fresh keys — that's not an
        # error, just a no-op signal.
        if self.account.encryption:
            try:
                await self._client.keys_upload()
            except LocalProtocolError as e:
                log.debug("[%s] keys_upload no-op: %s", self.user_id, e)
            except Exception:
                log.exception("[%s] keys_upload failed; continuing", self.user_id)

            if self.account.password_file:
                await self._bootstrap_cross_signing(token)
            else:
                log.info(
                    "[%s] no password_file in config; skipping cross-signing bootstrap "
                    "(bot will appear unverified in Element)",
                    self.user_id,
                )

        self._client.add_event_callback(self._on_room_message, RoomMessageText)
        self._client.add_event_callback(self._on_invite, InviteMemberEvent)
        # Element X uses MSC2241 in-room verification requests rather than the
        # legacy to-device start. They arrive as m.room.message with
        # msgtype="m.key.verification.request", which matrix-nio surfaces as
        # RoomMessageUnknown. We respond with an m.key.verification.ready
        # event in the same room; Element then transitions to to-device SAS
        # where the to-device callbacks below take over.
        self._client.add_event_callback(
            self._on_room_message_unknown, RoomMessageUnknown,
        )
        # MSC2241 in-room SAS protocol events arrive as bare event_types
        # (m.key.verification.start, .key, .mac, .done, .cancel). They get
        # parsed as UnknownEvent because matrix-nio has no specific class.
        # This callback dispatches them into the in-room SAS state machine.
        self._client.add_event_callback(
            self._on_room_unknown_event, UnknownEvent,
        )
        # Auto-accept SAS verification from allowlisted users. matrix-nio
        # surfaces the SAS protocol via to-device events; the bot must
        # actively participate (accept the start, confirm the SAS) for
        # Element to consider the device verified. Without this, Element
        # X shows "user identity changed" warnings indefinitely.
        self._client.add_to_device_callback(
            self._on_key_verification_start, KeyVerificationStart,
        )
        self._client.add_to_device_callback(
            self._on_key_verification_key, KeyVerificationKey,
        )
        self._client.add_to_device_callback(
            self._on_key_verification_mac, KeyVerificationMac,
        )
        self._client.add_to_device_callback(
            self._on_key_verification_cancel, KeyVerificationCancel,
        )

        log.info("[%s] starting sync_forever (device=%s)", self.user_id, self.account.device_id)
        self._sync_task = asyncio.create_task(
            self._client.sync_forever(timeout=30_000, full_state=True),
            name=f"matrix-sync-{self.account.device_id}",
        )

    async def _bootstrap_cross_signing(self, access_token: str) -> None:
        from claw.channel.matrix_crosssigning import ensure_cross_signing
        try:
            password = self.account.password_file.read_text().strip()
        except OSError:
            log.exception("[%s] cannot read password_file; skipping cross-signing", self.user_id)
            return
        try:
            await ensure_cross_signing(
                homeserver=self.account.homeserver,
                user_id=self.user_id,
                device_id=self.account.device_id,
                access_token=access_token,
                password=password,
                seed_store_path=Path(self.account.store_path) / "cross_signing.json",
                olm_account_sign=self._client.olm.account.sign,
                force_replace=self.account.force_cross_signing_replace,
            )
        except Exception:
            log.exception("[%s] cross-signing bootstrap failed; continuing", self.user_id)

    @staticmethod
    def _read_token(path: Path) -> str:
        try:
            return path.read_text().strip()
        except OSError as e:
            raise SystemExit(f"matrix access_token_file unreadable: {path} ({e})")

    async def _on_room_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if self._client is None or self._on_message is None:
            return
        if event.sender == self.user_id:
            return
        is_dm = len(room.users) <= 2
        if is_dm:
            if self._allow_from and event.sender not in self._allow_from:
                log.warning("[%s] dropping DM from non-allowlisted %s", self.user_id, event.sender)
                return
        else:
            if self.account.allow_bots == "none":
                return
            if self.account.allow_bots == "mentions" and not self._mentioned(event.body):
                return

        sender_name = room.user_name(event.sender) or event.sender
        try:
            await self._on_message(InboundMessage(
                peer_id=room.room_id,
                sender_name=sender_name,
                text=event.body,
                channel=self.name,
                sender_id=event.sender,
            ))
        except Exception:
            log.exception("[%s] inbound handler raised", self.user_id)

    async def _on_room_unknown_event(
        self,
        room: MatrixRoom,
        event: UnknownEvent,
    ) -> None:
        """Dispatch in-room SAS protocol events to their handlers.

        Element X uses MSC2241 + in-room SAS — every protocol step
        (start / accept / key / mac / done / cancel) arrives as its own
        event_type in the room rather than as to-device events. matrix-nio
        doesn't surface these as typed events, so we route them ourselves
        and drive the Sas state machine by hand.
        """
        if event.sender == self.user_id:
            return
        if not event.type or not event.type.startswith("m.key.verification."):
            return
        content = event.source.get("content", {})
        if event.sender not in self._allow_from:
            log.warning(
                "[%s] dropping in-room verification event %s from non-allowlisted %s",
                self.user_id, event.type, event.sender,
            )
            return
        try:
            if event.type == "m.key.verification.start":
                await self._inroom_sas_start(room, content, event.sender)
            elif event.type == "m.key.verification.key":
                await self._inroom_sas_key(content, event.sender)
            elif event.type == "m.key.verification.mac":
                await self._inroom_sas_mac(content, event.sender)
            elif event.type == "m.key.verification.done":
                await self._inroom_sas_done(content, event.sender)
            elif event.type == "m.key.verification.cancel":
                await self._inroom_sas_cancel(content, event.sender)
        except Exception:
            log.exception("[%s] in-room SAS handler raised", self.user_id)

    def _request_id_from(self, content: dict) -> str | None:
        """Pull the request event_id out of m.relates_to (MSC2241 transaction id)."""
        rel = content.get("m.relates_to") or {}
        return rel.get("event_id")

    async def _send_inroom_sas_event(
        self,
        room_id: str,
        event_type: str,
        content_from_sas: dict,
        request_event_id: str,
    ) -> None:
        """Send a SAS protocol event in-room with MSC2241 m.relates_to.

        ``content_from_sas`` is the to-device-shaped content dict that
        matrix-nio's Sas methods produce. Strip ``transaction_id`` (the
        in-room flow uses m.relates_to instead) and add the relation.
        """
        if self._client is None:
            return
        c = {k: v for k, v in content_from_sas.items() if k != "transaction_id"}
        c["m.relates_to"] = {
            "event_id": request_event_id,
            "rel_type": "m.reference",
        }
        try:
            await self._client.room_send(
                room_id=room_id,
                message_type=event_type,
                content=c,
                ignore_unverified_devices=True,
            )
        except Exception:
            log.exception("[%s] room_send for %s failed", self.user_id, event_type)

    async def _inroom_sas_start(
        self,
        room: MatrixRoom,
        content: dict,
        sender: str,
    ) -> None:
        request_event_id = self._request_id_from(content)
        if not request_event_id:
            log.warning("[%s] SAS start has no m.relates_to; ignoring", self.user_id)
            return
        from_device = content.get("from_device")
        if not from_device:
            return
        # Look up the OlmDevice for the sender's claimed device.
        other_device = None
        for d in self._client.device_store.active_user_devices(sender):
            if d.device_id == from_device:
                other_device = d
                break
        if other_device is None:
            log.warning(
                "[%s] SAS start: device %s for %s not in device store; cancelling",
                self.user_id, from_device, sender,
            )
            await self._send_inroom_sas_event(
                room.room_id, "m.key.verification.cancel",
                {"code": "m.user", "reason": "device unknown"},
                request_event_id,
            )
            return

        start_event = KeyVerificationStart(
            source={"type": "m.key.verification.start", "sender": sender, "content": content},
            sender=sender,
            transaction_id=request_event_id,
            from_device=from_device,
            method=content.get("method", "m.sas.v1"),
            key_agreement_protocols=content.get("key_agreement_protocols", []),
            hashes=content.get("hashes", []),
            message_authentication_codes=content.get("message_authentication_codes", []),
            short_authentication_string=content.get("short_authentication_string", []),
        )
        own_fp_key = self._client.olm.account.identity_keys["ed25519"]

        sas = Sas.from_key_verification_start(
            own_user=self.user_id,
            own_device=self.account.device_id,
            own_fp_key=own_fp_key,
            other_olm_device=other_device,
            event=start_event,
        )
        self._inroom_sas[request_event_id] = (sas, room.room_id)

        accept_msg = sas.accept_verification()
        log.info(
            "[%s] SAS accept -> %s (txn %s)",
            self.user_id, sender, request_event_id,
        )
        await self._send_inroom_sas_event(
            room.room_id, "m.key.verification.accept",
            accept_msg.content, request_event_id,
        )

    async def _inroom_sas_key(self, content: dict, sender: str) -> None:
        request_event_id = self._request_id_from(content)
        if not request_event_id or request_event_id not in self._inroom_sas:
            return
        sas, room_id = self._inroom_sas[request_event_id]
        if sas.canceled:
            return
        their_key = content.get("key")
        if not their_key:
            return
        sas.set_their_pubkey(their_key)

        # Send our key.
        our_key_msg = sas.share_key()
        await self._send_inroom_sas_event(
            room_id, "m.key.verification.key",
            our_key_msg.content, request_event_id,
        )
        # State transitions are normally handled by OlmMachine in the
        # to-device flow. Driving Sas directly from in-room events means
        # we must transition manually: both keys exchanged -> key_received.
        sas.state = SasState.key_received
        # Auto-confirm the SAS without comparing emojis: the sender is on
        # the bot's allow_from list, so we trust them on a fresh master key.
        try:
            sas.accept_sas()
            log.info(
                "[%s] SAS auto-accepted for %s (txn %s)",
                self.user_id, sender, request_event_id,
            )
        except Exception:
            log.exception("[%s] sas.accept_sas() raised", self.user_id)

    async def _inroom_sas_mac(self, content: dict, sender: str) -> None:
        request_event_id = self._request_id_from(content)
        if not request_event_id or request_event_id not in self._inroom_sas:
            return
        sas, room_id = self._inroom_sas[request_event_id]
        if sas.canceled:
            return

        # Feed the incoming MAC into the Sas state machine.
        mac_event = KeyVerificationMac(
            source={"type": "m.key.verification.mac", "sender": sender, "content": content},
            sender=sender,
            transaction_id=request_event_id,
            keys=content.get("keys", ""),
            mac=content.get("mac", {}),
        )
        sas.receive_mac_event(mac_event)
        if sas.canceled:
            log.warning(
                "[%s] SAS CANCELLED post-mac. sas.cancel_reason=%r cancel_code=%r",
                self.user_id, getattr(sas, "cancel_reason", None),
                getattr(sas, "cancel_code", None),
            )
            return

        # Send our MAC. matrix-nio's Sas.get_mac() only covers our device
        # key; for Element to upgrade user-level cross-signing trust we also
        # need to MAC our master signing key (and recompute the keys MAC
        # over the combined set). Without this, Element validates the
        # device MAC, completes the protocol, but never marks the USER
        # verified — so re-running the SAS flow loops indefinitely.
        try:
            our_mac_msg = sas.get_mac()
        except Exception:
            log.exception("[%s] sas.get_mac() raised", self.user_id)
            return
        extended_content = self._extend_mac_with_master_key(sas, our_mac_msg.content)
        await self._send_inroom_sas_event(
            room_id, "m.key.verification.mac",
            extended_content, request_event_id,
        )
        if getattr(sas, "verified", False):
            log.info(
                "[%s] SAS VERIFIED with %s (txn %s)",
                self.user_id, sender, request_event_id,
            )
        # Send done to wrap up the protocol.
        await self._send_inroom_sas_event(
            room_id, "m.key.verification.done",
            {}, request_event_id,
        )

    def _extend_mac_with_master_key(
        self,
        sas: Sas,
        original_content: dict,
    ) -> dict:
        """Add our master-signing-key MAC to the SAS mac content + recompute
        the ``keys`` MAC over the combined key id list. Required for Element
        to actually advance cross-signing trust on completion.

        Loads the master public key from our persisted cross-signing seeds
        (we never need its private here — Element side trusts the public
        key once the MAC validates).
        """
        seed_path = Path(self.account.store_path) / "cross_signing.json"
        if not seed_path.exists():
            log.warning(
                "[%s] no cross_signing.json — sending device-only MAC",
                self.user_id,
            )
            return original_content
        try:
            seeds = _json.loads(seed_path.read_text())
            master = PkSigning(bytes.fromhex(seeds["master"]))
        except Exception:
            log.exception("[%s] failed to load master seed for MAC extension", self.user_id)
            return original_content

        master_pubkey = master.public_key
        master_kid = f"ed25519:{master_pubkey}"

        # Mirror Sas.get_mac's info string + MAC method selection.
        info = (
            "MATRIX_KEY_VERIFICATION_MAC"
            f"{sas.own_user}{sas.own_device}"
            f"{sas.other_olm_device.user_id}{sas.other_olm_device.id}"
            f"{sas.transaction_id}"
        )
        if sas.chosen_mac_method == sas._mac_normal:
            calc = sas.calculate_mac
        else:
            calc = sas.calculate_mac_long_kdf

        new_content = dict(original_content)
        new_mac = dict(new_content.get("mac", {}))
        new_mac[master_kid] = calc(master_pubkey, info + master_kid)
        new_content["mac"] = new_mac
        new_content["keys"] = calc(",".join(sorted(new_mac.keys())), info + "KEY_IDS")
        log.info(
            "[%s] extended MAC with master key %s (keys covered: %s)",
            self.user_id, master_kid, sorted(new_mac.keys()),
        )
        return new_content

    async def _inroom_sas_done(self, content: dict, sender: str) -> None:
        request_event_id = self._request_id_from(content)
        if not request_event_id:
            return
        # Other side acknowledged completion. Clear state.
        if request_event_id in self._inroom_sas:
            log.info(
                "[%s] SAS done with %s (txn %s); cleaning up",
                self.user_id, sender, request_event_id,
            )
            self._inroom_sas.pop(request_event_id, None)

    async def _inroom_sas_cancel(self, content: dict, sender: str) -> None:
        request_event_id = self._request_id_from(content)
        log.info(
            "[%s] SAS cancelled by %s (txn %s): %s",
            self.user_id, sender, request_event_id,
            content.get("reason", "(no reason)"),
        )
        if request_event_id and request_event_id in self._inroom_sas:
            self._inroom_sas.pop(request_event_id, None)

    # --- key verification: in-room request (MSC2241) ------------------

    async def _on_room_message_unknown(
        self,
        room: MatrixRoom,
        event: RoomMessageUnknown,
    ) -> None:
        """Handle Element X's in-room m.key.verification.request by sending
        back m.key.verification.ready. After ready, Element transitions to
        to-device m.key.verification.start, and the to_device callbacks
        below complete the SAS exchange.
        """
        if self._client is None:
            return
        if event.sender == self.user_id:
            return
        if event.msgtype != "m.key.verification.request":
            return
        if event.sender not in self._allow_from:
            log.warning(
                "[%s] dropping in-room verification request from non-allowlisted %s",
                self.user_id, event.sender,
            )
            return
        # Confirm the request is addressed to us.
        target = event.content.get("to")
        if target and target != self.user_id:
            return

        log.info(
            "[%s] received in-room verification request from %s (event %s) — sending ready",
            self.user_id, event.sender, event.event_id,
        )
        ready_content = {
            "from_device": self.account.device_id,
            "methods": ["m.sas.v1"],
            "m.relates_to": {
                "event_id": event.event_id,
                "rel_type": "m.reference",
            },
        }
        try:
            await self._client.room_send(
                room_id=room.room_id,
                message_type="m.key.verification.ready",
                content=ready_content,
                ignore_unverified_devices=True,
            )
        except Exception:
            log.exception("[%s] failed to send m.key.verification.ready", self.user_id)

    # --- key verification: to-device SAS (auto-accept from allowlisted) ---

    async def _on_key_verification_start(self, event: KeyVerificationStart) -> None:
        if self._client is None:
            return
        if event.sender not in self._allow_from:
            log.warning(
                "[%s] dropping key-verification start from non-allowlisted %s",
                self.user_id, event.sender,
            )
            return
        log.info(
            "[%s] accepting key verification from %s (txn %s)",
            self.user_id, event.sender, event.transaction_id,
        )
        try:
            await self._client.accept_key_verification(event.transaction_id)
        except Exception:
            log.exception("[%s] accept_key_verification failed", self.user_id)

    async def _on_key_verification_key(self, event: KeyVerificationKey) -> None:
        if self._client is None:
            return
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas is None:
            return
        if sas.other_olm_device.user_id not in self._allow_from:
            return
        # Trust-on-first-use: confirm the SAS without actual emoji compare.
        # The bot is a closed-system actor in a private homeserver; we accept
        # whatever the user's device claims for allowlisted user IDs.
        log.info(
            "[%s] auto-confirming SAS for txn %s (allowlisted %s)",
            self.user_id, event.transaction_id, sas.other_olm_device.user_id,
        )
        try:
            await self._client.confirm_short_auth_string(event.transaction_id)
        except Exception:
            log.exception("[%s] confirm_short_auth_string failed", self.user_id)

    async def _on_key_verification_mac(self, event: KeyVerificationMac) -> None:
        if self._client is None:
            return
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas is None:
            return
        if getattr(sas, "verified", False):
            log.info(
                "[%s] key verification with %s VERIFIED",
                self.user_id, sas.other_olm_device.user_id,
            )

    async def _on_key_verification_cancel(self, event: KeyVerificationCancel) -> None:
        log.info(
            "[%s] key verification cancelled by %s: %s",
            self.user_id, event.sender, getattr(event, "reason", "(no reason)"),
        )

    async def _on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        if self._client is None:
            return
        if self.account.auto_join != "always":
            return
        if event.state_key != self.user_id:
            return
        try:
            log.info("[%s] auto-joining %s (invited by %s)", self.user_id, room.room_id, event.sender)
            await self._client.join(room.room_id)
        except Exception:
            log.exception("[%s] join failed for %s", self.user_id, room.room_id)

    def _mentioned(self, body: str) -> bool:
        b = body.lower()
        return self.user_id.lower() in b or f"@{self._own_localpart}" in b

    def _resolve_room_for_user(self, mxid: str) -> str | None:
        """Find an existing 1:1 DM room shared with ``mxid``. Returns the
        room_id, or None if no such room is currently joined. Picks the
        most recently active match if multiple exist (rare, but matrix
        permits it)."""
        if self._client is None:
            return None
        candidates: list[tuple[float, str]] = []
        for room_id, room in self._client.rooms.items():
            members = set(getattr(room, "users", {}).keys())
            if members == {self.user_id, mxid}:
                # MatrixRoom.last_event_timestamp is ms-since-epoch when set,
                # 0 otherwise — fine as a tiebreaker.
                ts = float(getattr(room, "last_event_timestamp", 0) or 0)
                candidates.append((ts, room_id))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    async def send(self, peer_id: str, text: str) -> None:
        if self._client is None:
            log.warning("send() before start(); dropping")
            return
        if not text.strip():
            return
        # peer_id is either a MXID (when a cron job's deliver_to is set to a
        # user, or an inbound came from a DM keyed by user) or a room_id (an
        # inbound from a group room). MXIDs need a 1:1 DM room lookup before
        # nio's room_send can use them.
        if peer_id.startswith("@"):
            room_id = self._resolve_room_for_user(peer_id)
            if room_id is None:
                log.warning(
                    "[%s] no 1:1 room with %s; dropping send. Invite the bot "
                    "to a DM to establish one.", self.user_id, peer_id)
                return
        elif peer_id.startswith("!"):
            room_id = peer_id
        else:
            # Synthetic InboundMessage channels (e.g., the "bootstrap"
            # sentinel from initial_prompt) have no real Matrix recipient.
            # The agent's reply has nowhere to go — log and drop.
            log.warning(
                "[%s] message sent to non-existent channel %r: %s",
                self.user_id, peer_id, text,
            )
            return
        for chunk in _chunk(text):
            content: dict[str, Any] = {"msgtype": "m.text", "body": chunk}
            try:
                html = _to_html(chunk)
            except Exception:
                # Never let a render failure drop a message — fall through
                # to plain-body-only send.
                log.exception("[%s] markdown render failed; sending plain",
                              self.user_id)
            else:
                content["format"] = "org.matrix.custom.html"
                content["formatted_body"] = html
            try:
                await self._client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content=content,
                    # TOFU for v0: closed system, two known accounts on a private
                    # homeserver. Without this matrix-nio refuses to encrypt to
                    # any unverified device and raises OlmUnverifiedDeviceError.
                    ignore_unverified_devices=True,
                )
            except Exception:
                log.exception("[%s] room_send failed for %s (room=%s)",
                              self.user_id, peer_id, room_id)

    def typing(self, peer_id: str) -> AsyncContextManager[None]:
        # Same MXID -> room resolution as send(). If no DM room exists yet,
        # or peer_id is a synthetic-channel sentinel (no Matrix room behind
        # it), the typing context becomes a no-op (returns a benign room_id
        # of "") and _TypingContext skips the API call.
        if peer_id.startswith("@"):
            room_id = self._resolve_room_for_user(peer_id) or ""
        elif peer_id.startswith("!"):
            room_id = peer_id
        else:
            room_id = ""
        return _TypingContext(self._client, room_id)

    async def clear_typing(self, peer_id: str) -> None:
        """Force the room's typing state OFF (one-shot, no renew loop).

        Used after a control-plane reply lands mid-turn: that reply
        clears the bot's typing *client-side*, but the turn's typing
        heartbeat only ever re-asserts ``True`` (never a transition), so
        the server never re-broadcasts ``m.typing`` and the indicator
        would stay gone for the rest of the turn. Dropping server typing
        here makes the heartbeat's next re-assert a real ``false→true``
        the client actually renders. Same MXID→room resolution as
        ``typing()``; a no-op when there's no room behind ``peer_id``.
        """
        if self._client is None:
            return
        if peer_id.startswith("@"):
            room_id = self._resolve_room_for_user(peer_id) or ""
        elif peer_id.startswith("!"):
            room_id = peer_id
        else:
            room_id = ""
        if not room_id:
            return
        try:
            await self._client.room_typing(room_id, False)
        except Exception:
            log.debug("clear_typing failed for %s", peer_id, exc_info=True)

    async def shutdown(self) -> None:
        log.info("[%s] shutting down", self.user_id)
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass


class _TypingContext:
    def __init__(self, client: AsyncClient | None, room_id: str) -> None:
        self._client = client
        self._room_id = room_id
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> None:
        if self._client is None or not self._room_id:
            return
        self._task = asyncio.create_task(self._loop())

    async def __aexit__(self, *exc: Any) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None and self._room_id:
            try:
                await self._client.room_typing(self._room_id, False)
            except Exception:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                if self._client is not None:
                    await self._client.room_typing(self._room_id, True, timeout=8000)
                await asyncio.sleep(TYPING_RENEW_INTERVAL_S)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("typing loop error; pausing")
                await asyncio.sleep(TYPING_RENEW_INTERVAL_S)


def _chunk(text: str, limit: int = MATRIX_CHUNK_MAX) -> list[str]:
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks
