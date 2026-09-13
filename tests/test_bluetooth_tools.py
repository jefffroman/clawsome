"""Tests for the bluetooth_* tool family.

The execution layer is stubbed rather than exercised: a real call bootstraps a
launchd job into a GUI session, which exists on the deployment host and nowhere
else. What is worth testing here is everything around that call — the two config
gates, address handling, the scan clamp, and the cases a working radio never
produces: a lost TCC grant, an absent console session, an empty scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claw.config import BluetoothConfig, _parse_bluetooth
from claw.tools import bluetooth as bt

BIN = Path("/opt/homebrew/bin/blueutil")


def _tools(verbs=bt.ALL_VERBS, max_scan: int = 20):
    return bt.build_bluetooth_tools(BIN, verbs, max_scan, 5.0)


@pytest.fixture
def fake_run(monkeypatch):
    """Record argv and reply from a scripted table keyed on the first flag."""
    calls: list[list[str]] = []
    table: dict[str, tuple[int, str, str]] = {"--power": (0, "1", "")}

    async def _run(binary, args, timeout):
        args = [str(a) for a in args]
        calls.append(args)
        return table.get(args[0], (0, "", ""))

    monkeypatch.setattr(bt, "run_in_console_session", _run)
    return calls, table


@pytest.fixture(autouse=True)
def controller_on(monkeypatch):
    async def _on():
        return True
    monkeypatch.setattr(bt, "_controller_is_on", _on)


# --- the two config gates -------------------------------------------------

def test_only_enabled_verbs_are_built():
    assert set(_tools(("list",))) == {"bluetooth_list"}
    assert set(_tools(("list", "scan"))) == {"bluetooth_list", "bluetooth_scan"}


def test_full_verb_set_builds_every_tool():
    assert set(_tools()) == {f"bluetooth_{v}" for v in bt.ALL_VERBS}


def test_absent_block_is_disabled_and_observing_only():
    cfg = _parse_bluetooth(None)
    assert cfg.enabled is False
    assert cfg.verbs == ("list", "scan")


def test_unknown_verb_fails_the_load():
    with pytest.raises(ValueError, match="teleport"):
        _parse_bluetooth({"enabled": True, "verbs": ["list", "teleport"]})


def test_verbs_parse_through_to_config():
    cfg = _parse_bluetooth({"enabled": True, "exposed_to": ["agent-1"], "verbs": ["list", "pair"]})
    assert cfg.enabled and cfg.exposed_to == ("agent-1",) and cfg.verbs == ("list", "pair")


# --- addresses ------------------------------------------------------------

async def test_bad_address_is_refused_without_touching_the_radio(fake_run):
    calls, _ = fake_run
    out = await _tools()["bluetooth_pair"].run({"address": "not-a-mac"})
    assert "error" in out and "MAC" in out
    assert calls == []


@pytest.mark.parametrize("given", ["AA:BB:CC:DD:EE:FF", "aa-bb-cc-dd-ee-ff", "Aa-Bb-Cc-Dd-Ee-Ff"])
async def test_address_is_normalized(fake_run, given):
    calls, _ = fake_run
    await _tools()["bluetooth_connect"].run({"address": given})
    assert calls[-1] == ["--connect", "aa-bb-cc-dd-ee-ff"]


async def test_pin_is_passed_only_when_given(fake_run):
    calls, _ = fake_run
    tools = _tools()
    await tools["bluetooth_pair"].run({"address": "aa-bb-cc-dd-ee-ff"})
    assert calls[-1] == ["--pair", "aa-bb-cc-dd-ee-ff"]
    await tools["bluetooth_pair"].run({"address": "aa-bb-cc-dd-ee-ff", "pin": "0000"})
    assert calls[-1] == ["--pair", "aa-bb-cc-dd-ee-ff", "0000"]


# --- the failures a working radio never produces --------------------------

async def test_blindness_is_named_not_reported_as_radio_off(fake_run):
    calls, table = fake_run
    table["--power"] = (0, "0", "")          # blueutil says off
    out = await _tools()["bluetooth_list"].run({})
    assert "TCC" in out and "cdhash" in out
    assert ["--paired"] not in calls          # never proceeded


async def test_genuinely_powered_off_is_not_mistaken_for_blindness(fake_run, monkeypatch):
    calls, table = fake_run
    table["--power"] = (0, "0", "")

    async def _off():
        return False
    monkeypatch.setattr(bt, "_controller_is_on", _off)

    await _tools()["bluetooth_list"].run({})
    assert ["--paired"] in calls              # proceeded normally


async def test_unreachable_session_says_so(fake_run):
    _, table = fake_run
    table["--connect"] = (126, "", "could not reach the console session (gui/502): x")
    out = await _tools()["bluetooth_connect"].run({"address": "aa-bb-cc-dd-ee-ff"})
    assert "console session" in out


# --- scanning -------------------------------------------------------------

async def test_scan_seconds_are_clamped(fake_run):
    calls, _ = fake_run
    await _tools(max_scan=20)["bluetooth_scan"].run({"seconds": 999})
    assert calls[-1] == ["--inquiry", "20"]


async def test_scan_floor_is_one_second(fake_run):
    calls, _ = fake_run
    await _tools()["bluetooth_scan"].run({"seconds": 0})
    assert calls[-1] == ["--inquiry", "1"]


async def test_empty_scan_explains_itself(fake_run):
    out = await _tools()["bluetooth_scan"].run({"seconds": 5})
    assert "pairing mode" in out and "bluetooth_list" in out


async def test_scan_returns_findings_verbatim(fake_run):
    _, table = fake_run
    table["--inquiry"] = (0, 'address: aa-bb-cc-dd-ee-ff, name: "Some BT Adapter"', "")
    out = await _tools()["bluetooth_scan"].run({"seconds": 5})
    assert "Some BT Adapter" in out


# --- list / power ---------------------------------------------------------

async def test_list_rejects_an_unknown_set(fake_run):
    out = await _tools()["bluetooth_list"].run({"which": "everything"})
    assert "error" in out


async def test_empty_list_is_not_an_error(fake_run):
    out = await _tools()["bluetooth_list"].run({"which": "connected"})
    assert out == "no connected devices"


async def test_power_requires_an_explicit_state(fake_run):
    out = await _tools()["bluetooth_power"].run({"state": "maybe"})
    assert "error" in out


async def test_power_off_does_not_consult_the_blindness_guard(fake_run):
    calls, table = fake_run
    table["--power"] = (0, "0", "")
    out = await _tools()["bluetooth_power"].run({"state": "off"})
    assert out == "controller powered off"
    assert calls[-1] == ["--power", "0"]


# --- the shell script the job runs ---------------------------------------

def test_xml_escaping_of_the_job_script():
    assert bt._xml_escape("a & b < c > d") == "a &amp; b &lt; c &gt; d"
