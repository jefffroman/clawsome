"""A named output device's volume, read and set through CoreAudio directly.

Addressed by the device's **UID** — the part of an mpv ``coreaudio/<UID>``
device string after the prefix — and never through the machine-wide default
output. ``osascript``'s ``set volume output volume`` is the obvious tool and
the wrong one: it has no way to name a device, so it acts on whatever macOS
currently calls the default, which a Bluetooth reconnect can change behind
everyone's back. The same call aimed at one room can move another.

The value is ``VirtualMainVolume`` — the one the menu-bar slider shows, 0.0 to
1.0 — exposed here as a whole percentage. It is the **device** volume: for a
Bluetooth sink without absolute volume it is software attenuation applied
before the codec, so anything under 100 there reaches the encoder quieter than
the file.

ctypes against the system frameworks, so no dependency; loaded on first use so
importing this module off macOS costs nothing. The calls are synchronous and
fast, but they are IPC to ``coreaudiod`` — callers on an event loop should run
them in a thread.
"""
from __future__ import annotations

import ctypes
import struct
import sys
from functools import cache

MPV_PREFIX = "coreaudio/"

_SYSTEM_OBJECT = 1
_UTF8 = 0x08000100


class CoreAudioError(RuntimeError):
    """The device could not be found or would not take the change."""


def _code(four: str) -> int:
    return struct.unpack(">I", four.encode())[0]


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


@cache
def _libs() -> tuple[ctypes.CDLL, ctypes.CDLL, ctypes.CDLL]:
    if sys.platform != "darwin":
        raise CoreAudioError("device volume is only available on macOS")
    fw = "/System/Library/Frameworks/{0}.framework/{0}"
    ca = ctypes.CDLL(fw.format("CoreAudio"))
    at = ctypes.CDLL(fw.format("AudioToolbox"))
    cf = ctypes.CDLL(fw.format("CoreFoundation"))
    get_args = [ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32,
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    ca.AudioObjectGetPropertyData.argtypes = get_args
    at.AudioHardwareServiceGetPropertyData.argtypes = get_args
    at.AudioHardwareServiceSetPropertyData.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    return ca, at, cf


def uid_of(mpv_device: str) -> str:
    """The CoreAudio UID inside an mpv device string."""
    if not mpv_device.startswith(MPV_PREFIX):
        raise CoreAudioError(f"{mpv_device!r} is not a coreaudio device")
    return mpv_device[len(MPV_PREFIX):]


def _device(uid: str) -> int:
    ca, _, cf = _libs()
    ref = ctypes.c_void_p(cf.CFStringCreateWithCString(None, uid.encode(), _UTF8))
    try:
        dev, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
        addr = _Address(_code("uidd"), _code("glob"), 0)
        status = ca.AudioObjectGetPropertyData(
            _SYSTEM_OBJECT, ctypes.byref(addr), ctypes.sizeof(ref), ctypes.byref(ref),
            ctypes.byref(size), ctypes.byref(dev))
    finally:
        cf.CFRelease(ref)
    if status or not dev.value:
        # A Bluetooth sink that is not connected has no device at all.
        raise CoreAudioError(f"no audio device {uid!r} is present")
    return dev.value


_VOLUME = ("vmvc", "outp")   # VirtualMainVolume, output scope


def get_volume(mpv_device: str) -> int:
    """The device's volume, 0–100."""
    _, at, _ = _libs()
    dev = _device(uid_of(mpv_device))
    value, size = ctypes.c_float(), ctypes.c_uint32(4)
    addr = _Address(_code(_VOLUME[0]), _code(_VOLUME[1]), 0)
    status = at.AudioHardwareServiceGetPropertyData(
        dev, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(value))
    if status:
        raise CoreAudioError(f"could not read the volume of {mpv_device!r} (status {status})")
    return round(value.value * 100)


def set_volume(mpv_device: str, percent: int) -> None:
    """Set the device's volume, 0–100."""
    if not 0 <= percent <= 100:
        raise CoreAudioError(f"volume must be 0–100 (got {percent})")
    _, at, _ = _libs()
    dev = _device(uid_of(mpv_device))
    value = ctypes.c_float(percent / 100)
    addr = _Address(_code(_VOLUME[0]), _code(_VOLUME[1]), 0)
    status = at.AudioHardwareServiceSetPropertyData(
        dev, ctypes.byref(addr), 0, None, ctypes.sizeof(value), ctypes.byref(value))
    if status:
        raise CoreAudioError(f"{mpv_device!r} refused a volume change (status {status})")
