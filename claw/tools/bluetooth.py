"""Bluetooth radio control, via ``blueutil``.

macOS exposes no scriptable Bluetooth surface of its own: pairing is a GUI
action, and ``blueutil`` is the only CLI that reaches IOBluetooth. Reaching it
at all needs a TCC grant for ``kTCCServiceBluetoothAlways``, and TCC attributes
that request to the *responsible* process rather than to blueutil — a launchd
job (how the gateway runs) is judged on its own and can be granted, while an
interactive ssh session is judged on sshd and refused outright, grant or no
grant. These tools therefore work from the daemon and **cannot be reproduced by
hand at an ssh prompt**; a human reaching for blueutil to check the agent's work
will be told the radio is off. ``system_profiler`` is the ground truth that
needs neither a session nor TCC, and is used here to tell "radio is off" apart
from "we are blind", which look identical through blueutil alone.

Two independent gates, because they answer different questions:
``bluetooth.exposed_to`` says *who* may drive the radio, ``bluetooth.verbs``
says *what* may be done to it. A verb that is not enabled is never built into a
Tool, so the model is not shown a capability it would only be refused.

**Every call is routed through the console GUI session**, never run in the
gateway's own daemon context, because the two are not equivalent. Measured on
macOS 26: from a LaunchDaemon, ``--power`` and ``--paired`` answer correctly,
while ``--inquiry`` returns **empty with exit 0** — a device that a session-context
scan finds on its first sweep is invisible across four minutes of daemon-context
sweeps. Discovery and pairing are per-user operations in ``bluetoothd`` (it logs
``gConsoleUserID = 0`` with no console user), and the daemon reports their absence
as "nothing there" rather than as an error. A silently empty scan is worse than a
refusal, so all verbs take the one path that is known to be correct rather than
splitting by verb and hoping the split stays true.

The route works because launchd lets a process manage its **own** user's GUI
domain: the gateway runs as the console user, so it can bootstrap a one-shot job
into ``gui/<uid>`` with no privilege escalation and no sudoers entry. It follows
that Bluetooth here depends on the console session existing at all — on a Mac
that means working auto-login, which is a fragile thing to rest on and so is
reported as such when it is missing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable

from claw.tools.base import Tool

log = logging.getLogger(__name__)

# Every verb this module knows how to build. Config is validated against it, so
# a typo in claw.yaml fails the load instead of silently dropping a capability.
ALL_VERBS: tuple[str, ...] = (
    "list", "scan", "pair", "unpair", "connect", "disconnect", "power",
)

_ADDR_RE = re.compile(r"^[0-9a-fA-F]{2}([:-][0-9a-fA-F]{2}){5}$")
# blueutil prints addresses lowercase and dash-separated; accept either form
# from the model and normalize, so a MAC copied out of system_profiler works.
_SYSPROFILER = "/usr/sbin/system_profiler"


def _normalize(address: str) -> str | None:
    a = (address or "").strip()
    return a.replace(":", "-").lower() if _ADDR_RE.match(a) else None


async def _exec(program: str, args: Iterable[str], timeout: float) -> tuple[int, str, str]:
    """Run a program with argv (never a shell) under a hard wall-clock cap."""
    try:
        proc = await asyncio.create_subprocess_exec(
            program, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return 127, "", f"{program} not found"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout:g}s"
    return (
        proc.returncode or 0,
        out.decode(errors="replace").strip(),
        err.decode(errors="replace").strip(),
    )


_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array><string>/bin/sh</string><string>-c</string><string>{script}</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"""


async def run_in_console_session(
    binary: Path, args: Iterable[str], timeout: float
) -> tuple[int, str, str]:
    """Run a program inside the console GUI session as a one-shot launchd job.

    Exit status travels back through a sentinel file rather than by polling
    launchd for the job's state: a one-shot job is torn down as soon as it
    exits, so "the job is gone" and "the job never started" look identical from
    outside, and only the sentinel distinguishes them.
    """
    uid = os.getuid()
    domain = f"gui/{uid}"
    label = f"claw.bt.{uuid.uuid4().hex[:8]}"
    tmp = Path(tempfile.mkdtemp(prefix="claw-bt-"))
    out_p, err_p, rc_p = tmp / "out", tmp / "err", tmp / "rc"
    cmd = " ".join(shlex.quote(str(a)) for a in (binary, *args))
    script = (
        f"{cmd} >{shlex.quote(str(out_p))} 2>{shlex.quote(str(err_p))}; "
        f"echo $? >{shlex.quote(str(rc_p))}"
    )
    plist_p = tmp / "job.plist"
    plist_p.write_text(_PLIST.format(label=label, script=_xml_escape(script)))
    try:
        rc, _, err = await _exec("/bin/launchctl", ["bootstrap", domain, str(plist_p)], 30)
        if rc != 0:
            return 126, "", (
                f"could not reach the console session ({domain}): {err or rc}. "
                "Bluetooth needs a logged-in console session; on this machine that "
                "means auto-login is working. Nothing can be done about the radio "
                "until it is."
            )
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if rc_p.exists():
                break
            await asyncio.sleep(0.25)
        else:
            await _exec("/bin/launchctl", ["bootout", f"{domain}/{label}"], 15)
            return 124, "", f"timed out after {timeout:g}s"
        try:
            job_rc = int(rc_p.read_text().strip() or 1)
        except ValueError:
            job_rc = 1
        out = out_p.read_text(errors="replace").strip() if out_p.exists() else ""
        job_err = err_p.read_text(errors="replace").strip() if err_p.exists() else ""
        return job_rc, out, job_err
    finally:
        await _exec("/bin/launchctl", ["bootout", f"{domain}/{label}"], 15)
        shutil.rmtree(tmp, ignore_errors=True)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


async def _controller_is_on() -> bool | None:
    """Ground truth from system_profiler. None if it could not be read.

    Reads the IORegistry rather than the per-user keychain, so it needs neither
    a login session nor a TCC grant — which is exactly why it can arbitrate.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            _SYSPROFILER, "SPBluetoothDataType",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None
    for line in out.decode(errors="replace").splitlines():
        if line.strip().startswith("State:"):
            return line.split(":", 1)[1].strip().lower() == "on"
    return None


_BLIND = (
    "blueutil reports the controller off, but system_profiler reports it on — "
    "so this is not a radio state, it is a lost TCC grant for "
    "kTCCServiceBluetoothAlways. The grant is pinned to blueutil's cdhash, so a "
    "blueutil upgrade breaks it silently. Bluetooth cannot be driven until it is "
    "reinstated; report this rather than retrying."
)


async def _guard_blind(binary: Path, timeout: float) -> str | None:
    """Return an explanation if blueutil is blind, else None."""
    rc, out, _ = await run_in_console_session(binary, ["--power"], timeout)
    if rc == 0 and out.strip() == "0" and await _controller_is_on():
        return _BLIND
    return None


def _fmt(rc: int, out: str, err: str, empty: str) -> str:
    if rc != 0:
        return f"error: {err or out or f'blueutil exited {rc}'}"
    return out if out.strip() else empty


def build_bluetooth_tools(
    binary: Path,
    verbs: Iterable[str],
    max_scan_seconds: int,
    timeout_s: float,
) -> dict[str, Tool]:
    """Build the enabled subset of the ``bluetooth_*`` family."""
    enabled = set(verbs)
    tools: dict[str, Tool] = {}

    async def _addr_op(args: dict[str, Any], flag: str, ok: str) -> str:
        addr = _normalize(args.get("address", ""))
        if addr is None:
            return "error: address must be a MAC like 'ee-db-24-fe-83-e1' or 'EE:DB:24:FE:83:E1'"
        if (blind := await _guard_blind(binary, timeout_s)) is not None:
            return blind
        argv = [flag, addr]
        if flag == "--pair" and (pin := (args.get("pin") or "").strip()):
            argv.append(pin)
        rc, out, err = await run_in_console_session(binary, argv, timeout_s)
        if rc != 0:
            return f"error: {err or out or f'blueutil exited {rc}'}"
        return f"{ok} {addr}" + (f" ({out})" if out.strip() else "")

    if "list" in enabled:
        async def _list(args: dict[str, Any]) -> str:
            which = (args.get("which") or "paired").lower()
            if which not in {"paired", "connected", "recent"}:
                return "error: which must be one of: paired, connected, recent"
            if (blind := await _guard_blind(binary, timeout_s)) is not None:
                return blind
            rc, out, err = await run_in_console_session(binary, [f"--{which}"], timeout_s)
            return _fmt(rc, out, err, f"no {which} devices")

        tools["bluetooth_list"] = Tool(
            name="bluetooth_list",
            description=(
                "List Bluetooth devices this machine knows about. 'paired' is "
                "everything remembered, 'connected' only what is live right now, "
                "'recent' what was used lately. Pairing is remembered across "
                "reboots, so a paired device that is not connected usually just "
                "needs bluetooth_connect."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "which": {
                        "type": "string",
                        "enum": ["paired", "connected", "recent"],
                        "description": "Which set to list. Defaults to paired.",
                    },
                },
            },
            run=_list,
        )

    if "scan" in enabled:
        async def _scan(args: dict[str, Any]) -> str:
            try:
                secs = int(args.get("seconds", 10))
            except (TypeError, ValueError):
                return "error: seconds must be an integer"
            secs = max(1, min(secs, max_scan_seconds))
            if (blind := await _guard_blind(binary, timeout_s)) is not None:
                return blind
            rc, out, err = await run_in_console_session(binary, ["--inquiry", str(secs)], timeout_s + secs)
            if rc != 0:
                return f"error: {err or out or f'blueutil exited {rc}'}"
            if not out.strip():
                return (
                    f"no devices found in {secs}s. A device only appears while it is "
                    "actively advertising, which usually means holding its pairing "
                    "button — and that mode times out fast on most hardware, so ask "
                    "for it to be put back into pairing mode rather than rescanning "
                    "blindly. Already-paired devices do not appear here; use "
                    "bluetooth_list. Bluetooth Low Energy devices never appear here."
                )
            return out

        tools["bluetooth_scan"] = Tool(
            name="bluetooth_scan",
            description=(
                "Scan for Bluetooth devices in pairing mode nearby, and return "
                "their addresses. The radio is occupied for the whole scan. Only "
                "finds classic (BR/EDR) devices that are actively advertising — "
                "not BLE devices, and not devices already paired."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": f"How long to scan. Clamped to {max_scan_seconds}s. Defaults to 10.",
                    },
                },
            },
            run=_scan,
        )

    if "pair" in enabled:
        tools["bluetooth_pair"] = Tool(
            name="bluetooth_pair",
            description=(
                "Pair with a device by address, from bluetooth_scan. The device "
                "must still be in pairing mode. Pairing is remembered across "
                "reboots — do it once, then use bluetooth_connect. Most audio "
                "devices need no PIN."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Device MAC address."},
                    "pin": {"type": "string", "description": "Only if the device demands one, e.g. '0000'."},
                },
                "required": ["address"],
            },
            run=lambda a: _addr_op(a, "--pair", "paired"),
        )

    if "unpair" in enabled:
        tools["bluetooth_unpair"] = Tool(
            name="bluetooth_unpair",
            description=(
                "Forget a paired device. It cannot be reconnected afterwards "
                "without physically putting it back into pairing mode, so treat "
                "this as a last resort for a device that genuinely will not work, "
                "never as a way to tidy the list."
            ),
            input_schema={
                "type": "object",
                "properties": {"address": {"type": "string", "description": "Device MAC address."}},
                "required": ["address"],
            },
            run=lambda a: _addr_op(a, "--unpair", "unpaired"),
        )

    if "connect" in enabled:
        tools["bluetooth_connect"] = Tool(
            name="bluetooth_connect",
            description=(
                "Connect to an already-paired device. Fails if the device is off "
                "or out of range — which is information, not something to retry "
                "in a loop."
            ),
            input_schema={
                "type": "object",
                "properties": {"address": {"type": "string", "description": "Device MAC address."}},
                "required": ["address"],
            },
            run=lambda a: _addr_op(a, "--connect", "connected"),
        )

    if "disconnect" in enabled:
        tools["bluetooth_disconnect"] = Tool(
            name="bluetooth_disconnect",
            description="Disconnect a connected device. It stays paired.",
            input_schema={
                "type": "object",
                "properties": {"address": {"type": "string", "description": "Device MAC address."}},
                "required": ["address"],
            },
            run=lambda a: _addr_op(a, "--disconnect", "disconnected"),
        )

    if "power" in enabled:
        async def _power(args: dict[str, Any]) -> str:
            state = str(args.get("state", "")).strip().lower()
            if state not in {"on", "off"}:
                return "error: state must be 'on' or 'off'"
            if state == "on" and (blind := await _guard_blind(binary, timeout_s)) is not None:
                return blind
            rc, out, err = await run_in_console_session(binary, ["--power", "1" if state == "on" else "0"], timeout_s)
            if rc != 0:
                return f"error: {err or out or f'blueutil exited {rc}'}"
            return f"controller powered {state}"

        tools["bluetooth_power"] = Tool(
            name="bluetooth_power",
            description=(
                "Turn the Bluetooth controller on or off. Powering off drops "
                "every connected device, including input devices someone may be "
                "using — on a headless machine there is no GUI to turn it back on "
                "with, so only this tool can undo it."
            ),
            input_schema={
                "type": "object",
                "properties": {"state": {"type": "string", "enum": ["on", "off"]}},
                "required": ["state"],
            },
            run=_power,
        )

    return tools
