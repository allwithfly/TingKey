from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
import re
import shlex
import sys
import tempfile
import threading
import time
import traceback
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from PySide6.QtCore import (
    QBuffer, QEasingCurve, QIODevice, QObject, QPropertyAnimation,
    QThread, QTimer, Qt, QUrl, Signal, Property,
)
from PySide6.QtGui import (
    QBrush, QCloseEvent, QColor, QDesktopServices, QFont,
    QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asr_media_utils import (
    build_time_segments,
    extract_audio_with_ffmpeg,
    expand_media_inputs,
    format_hms,
    get_media_duration_seconds,
    normalize_cli_path,
    prepare_audio_input,
    resolve_model_dir,
    save_srt_file,
    transcribe_with_timestamp_fallback,
)
from desktop_service.config_models import AppConfig
from desktop_service.config_store import ConfigStore
from speech_output import QwenSpeechOutput


APP_STYLE = """
QWidget {
    background: #eaf0f8;
    color: #0f172a;
    font-size: 14px;
}
QMainWindow {
    background: #eaf0f8;
}
QFrame#Sidebar {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffffff, stop:1 #eef3fb);
    border: 1px solid #d4dfee;
    border-radius: 18px;
}
QLabel#Brand {
    font-size: 24px;
    font-weight: 800;
    color: #0b1220;
}
QLabel#Subtle {
    color: #728097;
}
QFrame#MainPanel {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #fbfdff, stop:1 #f5f8fd);
    border: 1px solid #d6e1ef;
    border-radius: 20px;
}
QFrame#Card {
    background: #ffffff;
    border: 1px solid #dce6f2;
    border-radius: 16px;
}
QFrame#HeroCard {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0b1220, stop:1 #1f335a);
    border: 1px solid #122342;
    border-radius: 18px;
}
QFrame#ActionCard, QFrame#StatusCard, QFrame#ResultCard {
    background: #ffffff;
    border: 1px solid #dce6f2;
    border-radius: 16px;
}
QLabel#HeroTitle {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
}
QLabel#HeroSub {
    color: #e6efff;
    font-size: 14px;
    font-weight: 500;
}
QLabel#HeroChip {
    background: rgba(119, 154, 215, 0.24);
    border: 1px solid rgba(190, 218, 255, 0.52);
    border-radius: 8px;
    padding: 4px 10px;
    color: #f8fbff;
    font-size: 12px;
}
QListWidget#NavList {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget#NavList::item {
    margin: 4px 0;
    border-radius: 12px;
    padding: 11px 12px;
    color: #4b5b73;
    font-size: 16px;
}
QListWidget#NavList::item:selected {
    color: #0f172a;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #dbe7f8, stop:1 #e7eefb);
}
QLabel#SectionTitle {
    font-size: 29px;
    font-weight: 800;
    color: #0f172a;
}
QLabel#CardTitle {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}
QLabel#Chip {
    background: #edf2f9;
    border: 1px solid #d8e2ef;
    border-radius: 8px;
    padding: 4px 10px;
    color: #344256;
    font-size: 13px;
}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #d7e0ec;
    border-radius: 11px;
    padding: 8px;
}
QPlainTextEdit, QTextEdit {
    padding: 10px;
}
QPlainTextEdit#LiveText {
    background: #f8fbff;
    border: 1px solid #cfdced;
    color: #0f172a;
}
QPlainTextEdit#LogView {
    background: #0c1629;
    border: 1px solid #1e3357;
    color: #dce9ff;
    font-family: Consolas;
}
QPushButton {
    border: 1px solid #d2ddeb;
    border-radius: 11px;
    padding: 9px 14px;
    background: #f8fbff;
    color: #1f2a3c;
    font-weight: 600;
}
QPushButton:hover {
    border-color: #b9c8dc;
    background: #eef5ff;
}
QPushButton#PrimaryButton {
    background: #0f172a;
    color: #ffffff;
    border-color: #0f172a;
}
QPushButton#PrimaryButton:hover {
    background: #1f2a44;
}
QPushButton#AccentButton {
    background: #2563eb;
    color: #ffffff;
    border-color: #2563eb;
}
QPushButton#AccentButton:hover {
    background: #1d4ed8;
}
QPushButton#DangerButton {
    background: #dc2626;
    color: #ffffff;
    border-color: #dc2626;
}
QPushButton#DangerButton:hover {
    background: #b91c1c;
}
QPushButton#SecondaryButton {
    background: #eef4fc;
    border-color: #d0deef;
    color: #1e293b;
}
QPushButton#SecondaryButton:hover {
    background: #e4eefb;
    border-color: #bfd2e8;
}
QPushButton:disabled {
    background: #e8edf5;
    color: #96a4b7;
    border-color: #e0e7f0;
}
QPushButton:focus {
    border: 2px solid #1d4ed8;
}
QGroupBox {
    border: 1px solid #e2e8f3;
    border-radius: 13px;
    margin-top: 8px;
    padding-top: 16px;
    background: #ffffff;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px;
}
QTabWidget::pane {
    border: 1px solid #e2e8f3;
    border-radius: 13px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: #edf2f8;
    border: 1px solid #dce4f0;
    border-radius: 9px;
    padding: 8px 20px;
    margin-right: 6px;
    color: #55657d;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #c8d3e2;
    border-radius: 9px;
    background: #eef3fa;
}
QCheckBox::indicator:checked {
    border: 1px solid #0f172a;
    border-radius: 9px;
    background: #0f172a;
}
QProgressBar {
    border: 1px solid #d6e0ed;
    border-radius: 9px;
    background: #edf3fb;
    color: #0f172a;
    text-align: center;
    min-height: 19px;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3b82f6, stop:1 #1d4ed8);
}
QListWidget#HistoryList {
    background: #ffffff;
    border: 1px solid #e2e8f3;
    border-radius: 12px;
}
QListWidget#HistoryList::item {
    border-bottom: 1px solid #eef2f7;
    padding: 12px 8px;
}
"""


ACTION_CHOICES: list[tuple[str, str]] = [
    ("语音助手", "speech_assistant"),
    ("语音输入", "voice_input"),
    ("禁用", "disabled"),
]

SYSTEM_LANGUAGE_CHOICES: list[tuple[str, str]] = [
    ("中文(普通话)", "zh-CN"),
    ("English", "en-US"),
]

INPUT_LANGUAGE_HINT_CHOICES: list[tuple[str, str | None]] = [
    ("自动检测", None),
    ("Chinese", "Chinese"),
    ("English", "English"),
]

SKILL_ENHANCE_PRESET_CHOICES: list[tuple[str, str]] = [
    ("语音输入增强", "voice_input"),
    ("代码/命令行", "code_command"),
    ("原样输出", "raw"),
]

WINDOWS_VK_GROUPS: dict[str, tuple[int, ...]] = {
    "ctrl": (0xA2, 0xA3, 0x11),
    "leftctrl": (0xA2,),
    "lctrl": (0xA2,),
    "rightctrl": (0xA3,),
    "rctrl": (0xA3,),
    "shift": (0xA0, 0xA1, 0x10),
    "leftshift": (0xA0,),
    "lshift": (0xA0,),
    "rightshift": (0xA1,),
    "rshift": (0xA1,),
    "alt": (0xA4, 0xA5, 0x12),
    "leftalt": (0xA4,),
    "lalt": (0xA4,),
    "rightalt": (0xA5,),
    "ralt": (0xA5,),
    "win": (0x5B, 0x5C),
    "leftwin": (0x5B,),
    "lwin": (0x5B,),
    "rightwin": (0x5C,),
    "rwin": (0x5C,),
    "space": (0x20,),
    "enter": (0x0D,),
    "tab": (0x09,),
    "esc": (0x1B,),
    "escape": (0x1B,),
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compact_text(text: str, limit: int = 90) -> str:
    t = " ".join(text.strip().split())
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


_RE_CJK = re.compile(r"[\u3400-\u9fff]")
_RE_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_PUNCT = re.compile(r"([，。！？,.!?；;：:])\1+")
_RE_EN_WORD_REPEAT = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{0,63})(?:\s+\1){1,4}\b", re.IGNORECASE)
_RE_CN_REPEAT = re.compile(r"([\u4e00-\u9fff]{1,6})(?:\1){1,3}")

_FILLER_PATTERNS = [
    re.compile(r"(?:(?<=^)|(?<=[\s，。！？,.!?]))(?:嗯+|呃+|额+|啊+|唔+)(?=$|[\s，。！？,.!?])"),
    re.compile(r"(?:(?<=^)|(?<=[\s，。！？,.!?]))(?:就是|然后|这个|那个)(?=$|[\s，。！？,.!?])"),
]

_FULLWIDTH_TO_ASCII = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)

_CODE_WORD_REPLACEMENTS: list[tuple[str, str]] = [
    ("换行", "\n"),
    ("回车", "\n"),
    ("制表符", "\t"),
    ("tab键", "\t"),
    ("空格", " "),
    ("逗号", ","),
    ("句号", "."),
    ("问号", "?"),
    ("感叹号", "!"),
    ("冒号", ":"),
    ("分号", ";"),
    ("左括号", "("),
    ("右括号", ")"),
    ("左中括号", "["),
    ("右中括号", "]"),
    ("左花括号", "{"),
    ("右花括号", "}"),
    ("反引号", "`"),
]


def _contains_cjk(text: str) -> bool:
    return bool(_RE_CJK.search(text))


def _parse_user_dictionary_rules(dictionary_text: str) -> list[tuple[str, str]]:
    rules_map: dict[str, str] = {}
    for raw_line in str(dictionary_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        left = ""
        right = ""
        for sep in ("=>", "->", "=", "\t", ":", "｜", "|"):
            if sep in line:
                left, right = line.split(sep, 1)
                break
        if not left and not right:
            continue

        src = left.strip()
        dst = right.strip()
        if not src or not dst:
            continue
        rules_map[src] = dst

    rules = sorted(rules_map.items(), key=lambda item: len(item[0]), reverse=True)
    return rules


def _apply_dictionary_rules(text: str, rules: list[tuple[str, str]]) -> str:
    out = text
    for src, dst in rules:
        out = out.replace(src, dst)
    return out


def _remove_fillers(text: str) -> str:
    out = text
    for pattern in _FILLER_PATTERNS:
        out = pattern.sub(" ", out)
    out = _RE_MULTI_PUNCT.sub(r"\1", out)
    out = re.sub(r"\s+([，。！？,.!?；;：:])", r"\1", out)
    out = re.sub(r"([，。！？,.!?；;：:])\s*([，。！？,.!?；;：:])", r"\1", out)
    return out


def _dedupe_repeats_text(text: str) -> str:
    out = text
    for _ in range(3):
        new_text = _RE_EN_WORD_REPEAT.sub(r"\1", out)
        new_text = _RE_CN_REPEAT.sub(r"\1", new_text)
        new_text = _RE_MULTI_PUNCT.sub(r"\1", new_text)
        if new_text == out:
            break
        out = new_text
    return out


def _normalize_spacing_text(text: str) -> str:
    out = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    lines: list[str] = []
    for raw_line in out.splitlines():
        line = _RE_SPACE.sub(" ", raw_line).strip()
        line = re.sub(r"([\u3400-\u9fff])([A-Za-z0-9])", r"\1 \2", line)
        line = re.sub(r"([A-Za-z0-9])([\u3400-\u9fff])", r"\1 \2", line)
        line = re.sub(r"\s+([，。！？,.!?；;：:])", r"\1", line)
        line = re.sub(r"([(\[{（【])\s+", r"\1", line)
        line = re.sub(r"\s+([)\]}）】])", r"\1", line)
        lines.append(line)
    out = "\n".join(lines)
    out = _RE_MULTI_NEWLINE.sub("\n\n", out)
    return out.strip()


def _ensure_terminal_punctuation(text: str) -> str:
    lines = text.splitlines()
    punct = set("。！？.!?;；:：")
    out_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            out_lines.append(line)
            continue
        if line[-1] in punct:
            out_lines.append(line)
            continue
        out_lines.append(line + ("。" if _contains_cjk(line) else "."))
    return "\n".join(out_lines).strip()


def _apply_code_command_style(text: str) -> str:
    out = text.translate(_FULLWIDTH_TO_ASCII)
    for src, dst in _CODE_WORD_REPLACEMENTS:
        out = out.replace(src, dst)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\s+\.", ".", out)
    out = re.sub(r"\s+;", ";", out)
    out = re.sub(r"\s+:", ":", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = _RE_SPACE.sub(" ", out)
    return out.strip()


def _apply_personalize_prompt_rules(text: str, prompt: str) -> str:
    tip = str(prompt or "").strip().lower()
    if not tip:
        return text

    out = text
    if "去掉结尾最后一个句号" in prompt or "去掉结尾句号" in prompt:
        out = re.sub(r"[。.]$", "", out.rstrip())
    if "不要标点" in prompt or "去掉标点" in prompt or "无标点" in prompt:
        out = re.sub(r"[，。！？,.!?；;：:]", "", out)
    if "半角标点" in prompt or "英文标点" in prompt:
        out = out.translate(_FULLWIDTH_TO_ASCII)
    if "每句换行" in prompt:
        out = re.sub(r"([。！？.!?])\s*", r"\1\n", out)
        out = re.sub(r"\n{2,}", "\n", out)
    if "全部小写" in prompt or tip == "lower":
        out = out.lower()
    if "全部大写" in prompt or tip == "upper":
        out = out.upper()
    return out.strip()


def _auto_structure_text(text: str) -> str:
    if "\n" in text:
        return text
    chunks = [x.strip() for x in re.split(r"(?<=[。！？.!?])\s*", text) if x.strip()]
    if len(chunks) <= 1:
        return text
    return "\n".join(chunks)


def enhance_recognized_text(text: str, skills_cfg: Any) -> tuple[str, list[str]]:
    raw = str(text or "").strip()
    if not raw:
        return "", []

    enabled = bool(getattr(skills_cfg, "enhancement_enabled", True))
    preset = str(getattr(skills_cfg, "enhancement_preset", "voice_input") or "voice_input")
    if (not enabled) or preset == "raw":
        return raw, ["raw"]

    out = raw
    ops: list[str] = [f"preset:{preset}"]

    if bool(getattr(skills_cfg, "load_user_dictionary", False)):
        dict_text = str(getattr(skills_cfg, "user_dictionary_text", "") or "")
        rules = _parse_user_dictionary_rules(dict_text)
        if rules:
            replaced = _apply_dictionary_rules(out, rules)
            if replaced != out:
                out = replaced
                ops.append(f"dict:{len(rules)}")

    if preset == "code_command":
        code_style = _apply_code_command_style(out)
        if code_style != out:
            out = code_style
            ops.append("code_style")

    if bool(getattr(skills_cfg, "remove_fillers", True)) and preset != "code_command":
        cleaned = _remove_fillers(out)
        if cleaned != out:
            out = cleaned
            ops.append("remove_fillers")

    if bool(getattr(skills_cfg, "dedupe_repeats", True)):
        deduped = _dedupe_repeats_text(out)
        if deduped != out:
            out = deduped
            ops.append("dedupe")

    if bool(getattr(skills_cfg, "normalize_spacing", True)):
        spaced = _normalize_spacing_text(out)
        if spaced != out:
            out = spaced
            ops.append("spacing")

    if bool(getattr(skills_cfg, "personalize_enabled", False)):
        prompt = str(getattr(skills_cfg, "personalize_prompt", "") or "")
        prompted = _apply_personalize_prompt_rules(out, prompt)
        if prompted != out:
            out = prompted
            ops.append("prompt")

    if bool(getattr(skills_cfg, "auto_structure", False)):
        structured = _auto_structure_text(out)
        if structured != out:
            out = structured
            ops.append("structure")

    if bool(getattr(skills_cfg, "auto_punctuation", True)) and preset != "code_command":
        puncted = _ensure_terminal_punctuation(out)
        if puncted != out:
            out = puncted
            ops.append("punct")

    return out.strip(), ops


if sys.platform.startswith("win"):
    _WIN_USER32 = ctypes.WinDLL("user32", use_last_error=True)
    _ULONG_PTR = wintypes.WPARAM

    class _WinKeybdInput(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _WinInputUnion(ctypes.Union):
        _fields_ = [("ki", _WinKeybdInput)]

    class _WinInput(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", _WinInputUnion),
        ]

    _WIN_INPUT_KEYBOARD = 1
    _WIN_KEYEVENTF_KEYUP = 0x0002
    _WIN_KEYEVENTF_UNICODE = 0x0004
    _WIN_WM_CHAR = 0x0102

    class _WinGuiThreadInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("hwndActive", wintypes.HWND),
            ("hwndFocus", wintypes.HWND),
            ("hwndCapture", wintypes.HWND),
            ("hwndMenuOwner", wintypes.HWND),
            ("hwndMoveSize", wintypes.HWND),
            ("hwndCaret", wintypes.HWND),
            ("rcCaret", wintypes.RECT),
        ]

    _WIN_USER32.SendInput.argtypes = (
        wintypes.UINT,
        ctypes.POINTER(_WinInput),
        ctypes.c_int,
    )
    _WIN_USER32.SendInput.restype = wintypes.UINT
    _WIN_USER32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    _WIN_USER32.PostMessageW.restype = wintypes.BOOL
    _WIN_USER32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    )
    _WIN_USER32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _WIN_USER32.GetGUIThreadInfo.argtypes = (
        wintypes.DWORD,
        ctypes.POINTER(_WinGuiThreadInfo),
    )
    _WIN_USER32.GetGUIThreadInfo.restype = wintypes.BOOL
    _WIN_USER32.GetForegroundWindow.restype = wintypes.HWND
    _WIN_USER32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    _WIN_USER32.SetForegroundWindow.restype = wintypes.BOOL
    _WIN_USER32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    _WIN_USER32.ShowWindow.restype = wintypes.BOOL


def _iter_utf16_units(text: str):
    data = text.encode("utf-16-le")
    for i in range(0, len(data), 2):
        unit = int.from_bytes(data[i : i + 2], byteorder="little", signed=False)
        if unit != 0:
            yield unit


def _resolve_focus_hwnd(base_hwnd: int | None) -> int | None:
    if not sys.platform.startswith("win"):
        return None
    hwnd = int(base_hwnd or 0)
    if hwnd <= 0:
        hwnd = int(_WIN_USER32.GetForegroundWindow() or 0)
    if hwnd <= 0:
        return None
    thread_id = int(_WIN_USER32.GetWindowThreadProcessId(hwnd, None))
    if thread_id <= 0:
        return hwnd
    info = _WinGuiThreadInfo()
    info.cbSize = ctypes.sizeof(_WinGuiThreadInfo)
    ok = bool(_WIN_USER32.GetGUIThreadInfo(thread_id, ctypes.byref(info)))
    if not ok:
        return hwnd
    focus_hwnd = int(info.hwndFocus or 0)
    return focus_hwnd or hwnd


def _send_by_unicode_input(text: str) -> tuple[bool, str]:
    units = list(_iter_utf16_units(text))
    if not units:
        return False, "empty_utf16_units"
    chunk_units = 96
    for start in range(0, len(units), chunk_units):
        part = units[start : start + chunk_units]
        payload = (_WinInput * (len(part) * 2))()
        idx = 0
        for unit in part:
            payload[idx].type = _WIN_INPUT_KEYBOARD
            payload[idx].ki = _WinKeybdInput(0, unit, _WIN_KEYEVENTF_UNICODE, 0, 0)
            idx += 1
            payload[idx].type = _WIN_INPUT_KEYBOARD
            payload[idx].ki = _WinKeybdInput(
                0,
                unit,
                _WIN_KEYEVENTF_UNICODE | _WIN_KEYEVENTF_KEYUP,
                0,
                0,
            )
            idx += 1

        sent = int(
            _WIN_USER32.SendInput(
                idx,
                ctypes.cast(payload, ctypes.POINTER(_WinInput)),
                ctypes.sizeof(_WinInput),
            )
        )
        if sent != idx:
            return False, f"SendInput failed sent={sent}/{idx} last_error={ctypes.get_last_error()}"
    return True, "SendInput"


def _send_by_wm_char(text: str, hwnd: int | None) -> tuple[bool, str]:
    target = _resolve_focus_hwnd(hwnd)
    if not target:
        return False, "no_focus_hwnd"
    sent_any = False
    for unit in _iter_utf16_units(text):
        if not bool(_WIN_USER32.PostMessageW(target, _WIN_WM_CHAR, unit, 0)):
            return False, f"WM_CHAR failed hwnd={target} last_error={ctypes.get_last_error()}"
        sent_any = True
    return sent_any, f"WM_CHAR hwnd={target}"


def send_text_without_clipboard(text: str, preferred_hwnd: int | None = None) -> tuple[bool, str]:
    if not text:
        return False, "empty_text"
    if not sys.platform.startswith("win"):
        return False, "unsupported_platform"

    current_fg = int(_WIN_USER32.GetForegroundWindow() or 0)
    if preferred_hwnd and current_fg and int(preferred_hwnd) != current_fg:
        direct_ok, direct_detail = _send_by_wm_char(text, preferred_hwnd)
        if direct_ok:
            return True, f"direct {direct_detail}"

    ok, detail = _send_by_unicode_input(text)
    if ok:
        return True, detail
    fallback_ok, fallback_detail = _send_by_wm_char(text, preferred_hwnd)
    if fallback_ok:
        return True, f"{detail} -> fallback {fallback_detail}"
    return False, f"{detail}; fallback {fallback_detail}"


import math


class _SiriPillCard(QWidget):
    """Siri-style pill card with animated gradient light strip at bottom."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = 0.0
        self._active = False      # recording pulse
        self._flowing = False     # recognizing flow
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(30)
        self._anim_timer.timeout.connect(self._tick)

    def set_recording(self, active: bool) -> None:
        self._active = active
        self._flowing = False
        if active and not self._anim_timer.isActive():
            self._anim_timer.start()

    def set_flowing(self, flowing: bool) -> None:
        self._flowing = flowing
        self._active = False
        if flowing and not self._anim_timer.isActive():
            self._anim_timer.start()

    def set_idle(self) -> None:
        self._active = False
        self._flowing = False

    def _tick(self) -> None:
        self._phase += 0.06
        self.update()
        if not self._active and not self._flowing:
            if self._phase > 100:
                self._anim_timer.stop()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = h / 2.0  # full pill radius

        # ── pill background ──
        pill = QPainterPath()
        pill.addRoundedRect(0.0, 0.0, float(w), float(h), r, r)
        p.fillPath(pill, QColor(0, 0, 0, 185))

        # ── bottom gradient light strip ──
        if self._active or self._flowing:
            strip_h = 3.0
            strip_y = h - strip_h - 1.0
            grad = QLinearGradient(0, strip_y, w, strip_y)
            if self._active:
                # recording: pulsing warm colors
                pulse = 0.5 + 0.5 * math.sin(self._phase * 3.0)
                a = int(140 + 115 * pulse)
                grad.setColorAt(0.0, QColor(255, 90, 90, a))
                grad.setColorAt(0.3, QColor(255, 160, 60, a))
                grad.setColorAt(0.6, QColor(255, 90, 180, a))
                grad.setColorAt(1.0, QColor(255, 90, 90, a))
            else:
                # recognizing: flowing cool colors
                offset = (self._phase * 0.5) % 1.0
                a = 200
                stops = [
                    (0.0, QColor(80, 140, 255, a)),
                    (0.25, QColor(140, 100, 255, a)),
                    (0.5, QColor(255, 100, 200, a)),
                    (0.75, QColor(100, 220, 255, a)),
                    (1.0, QColor(80, 140, 255, a)),
                ]
                for pos, color in stops:
                    grad.setColorAt(min(1.0, max(0.0, (pos + offset) % 1.0)), color)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            strip_path = QPainterPath()
            strip_path.addRoundedRect(8.0, strip_y, float(w) - 16.0, strip_h, 1.5, 1.5)
            p.fillPath(strip_path, QBrush(grad))

        # ── subtle border ──
        p.setPen(QPen(QColor(255, 255, 255, 25), 0.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(0.5, 0.5, w - 1.0, h - 1.0, r, r)
        p.end()


class FloatingPromptBar(QWidget):
    """Minimalist Siri-style floating pill at screen bottom center."""

    def __init__(self) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        if hasattr(Qt.WindowType, "WindowDoesNotAcceptFocus"):
            flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("PromptOverlay")
        self.setMinimumWidth(320)
        self.setMaximumWidth(560)

        self._current_mode = "default"

        # --- pill card ---
        self._pill = _SiriPillCard(self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._pill)

        pill_layout = QVBoxLayout(self._pill)
        pill_layout.setContentsMargins(20, 10, 20, 12)
        pill_layout.setSpacing(2)

        # status line
        self.status_label = QLabel("正在录音...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("PromptStatus")
        pill_layout.addWidget(self.status_label)

        # text line (hidden when empty)
        self.text_label = QLabel("")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setObjectName("PromptText")
        self.text_label.setWordWrap(True)
        self.text_label.setVisible(False)
        pill_layout.addWidget(self.text_label)

        self.setStyleSheet("""
QLabel#PromptStatus {
    color: rgba(255, 255, 255, 0.95);
    font-size: 14px;
    font-weight: 600;
    background: transparent;
}
QLabel#PromptText {
    color: rgba(255, 255, 255, 0.62);
    font-size: 12px;
    font-weight: 400;
    background: transparent;
}
""")

        # timers
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._do_hide)
        self._fade_anim: QPropertyAnimation | None = None

    # ── internal ──────────────────────────────────────────────────

    def _detect_mode(self, status: str) -> str:
        s = status.lower()
        if "录音" in status or "recording" in s:
            return "recording"
        if "写入" in status or ("识别" in status and ("完成" in status or "已完成" in status)):
            return "done"
        if "识别" in status or "recogni" in s or "正在" in status:
            return "recognizing"
        if "失败" in status or "error" in s or "fail" in s:
            return "error"
        if "取消" in status or "cancel" in s:
            return "cancelled"
        if "繁忙" in status or "忙" in status or "加载" in status or "busy" in s:
            return "busy"
        if "完成" in status:
            return "done"
        return "default"

    def _apply_mode(self, mode: str) -> None:
        if mode == self._current_mode:
            return
        self._current_mode = mode
        if mode == "recording":
            self._pill.set_recording(True)
        elif mode == "recognizing":
            self._pill.set_flowing(True)
        else:
            self._pill.set_idle()

    def _reposition(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        width = min(self.maximumWidth(), max(self.minimumWidth(), int(geo.width() * 0.28)))
        self.resize(width, self.sizeHint().height())
        self.move(
            geo.x() + (geo.width() - self.width()) // 2,
            geo.y() + geo.height() - self.height() - 48,
        )

    def _do_hide(self) -> None:
        if self._fade_anim is not None:
            self._fade_anim.stop()
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(300)
        anim.setStartValue(self.windowOpacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InQuad)
        anim.finished.connect(self.hide)
        self._fade_anim = anim
        anim.start()

    def _fade_in(self) -> None:
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(200)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_anim = anim
        anim.start()

    # ── public API (unchanged signatures) ─────────────────────────

    def show_prompt(self, status: str, text: str = "") -> None:
        self.hide_timer.stop()
        self._apply_mode(self._detect_mode(status))
        self.status_label.setText(status)
        self.text_label.setText(compact_text(text, 220) if text else "")
        self.text_label.setVisible(bool(text))
        self._reposition()
        if not self.isVisible():
            self._fade_in()
        else:
            self.setWindowOpacity(1.0)
            self.show()
            self.raise_()

    def update_prompt(self, status: str | None = None, text: str | None = None) -> None:
        if status is not None:
            self._apply_mode(self._detect_mode(status))
            self.status_label.setText(status)
        if text is not None:
            self.text_label.setText(compact_text(text, 220) if text else "")
            self.text_label.setVisible(bool(text))
        self._reposition()
        if not self.isVisible():
            self._fade_in()

    def hide_later(self, ms: int = 1200) -> None:
        self.hide_timer.start(max(120, int(ms)))


def parse_input_tokens(raw: str) -> list[str]:
    try:
        parts = shlex.split(raw, posix=False)
    except ValueError:
        parts = raw.split()
    output: list[str] = []
    for part in parts:
        for token in part.split(","):
            value = normalize_cli_path(token)
            if value:
                output.append(value)
    return output


def default_binary_path(exe_name: str, fallback: str) -> str:
    candidates: list[Path] = []
    env_dir = os.environ.get("QWEN_ASR_FFMPEG_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(Path(__file__).resolve().parent / "ffmpeg" / "bin")
    candidates.append(Path(__file__).resolve().parent)
    for root in candidates:
        target = (root / exe_name).resolve()
        if target.exists():
            return str(target)
    return fallback


def _model_dump(config: AppConfig) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return config.dict()


@dataclass(frozen=True)
class RuntimeAsrConfig:
    model_dir: str
    model_size: str
    backend: str
    dtype: str
    max_new_tokens: int
    max_inference_batch_size: int
    attn_implementation: str
    enable_tf32: bool
    cudnn_benchmark: bool
    device: str

    def key(self) -> tuple[Any, ...]:
        return (
            self.model_dir,
            self.model_size,
            self.backend,
            self.dtype,
            self.max_new_tokens,
            self.max_inference_batch_size,
            self.attn_implementation,
            self.enable_tf32,
            self.cudnn_benchmark,
            self.device,
        )


def build_runtime_config(config: AppConfig, device: str) -> RuntimeAsrConfig:
    model = config.model
    model_dir = str(resolve_model_dir(model.asr_model_dir, model.asr_model_size))
    return RuntimeAsrConfig(
        model_dir=model_dir,
        model_size=model.asr_model_size,
        backend=model.asr_backend,
        dtype=model.asr_dtype,
        max_new_tokens=model.asr_max_new_tokens,
        max_inference_batch_size=model.asr_max_inference_batch_size,
        attn_implementation=model.asr_attn_implementation,
        enable_tf32=model.asr_enable_tf32,
        cudnn_benchmark=model.asr_cudnn_benchmark,
        device=device,
    )


class AsrEngineCache:
    _lock = threading.Lock()
    _infer_lock = threading.Lock()
    _engine: QwenSpeechOutput | None = None
    _config_key: tuple[Any, ...] | None = None

    @classmethod
    def get_engine(cls, cfg: RuntimeAsrConfig) -> QwenSpeechOutput:
        key = cfg.key()
        with cls._lock:
            if cls._engine is None or cls._config_key != key:
                cls._engine = QwenSpeechOutput(
                    model_path=cfg.model_dir,
                    device=cfg.device,
                    dtype=cfg.dtype,  # type: ignore[arg-type]
                    backend=cfg.backend,  # type: ignore[arg-type]
                    max_inference_batch_size=cfg.max_inference_batch_size,
                    max_new_tokens=cfg.max_new_tokens,
                    attn_implementation=cfg.attn_implementation,  # type: ignore[arg-type]
                    enable_tf32=cfg.enable_tf32,
                    cudnn_benchmark=cfg.cudnn_benchmark,
                )
                cls._config_key = key
        assert cls._engine is not None
        return cls._engine

    @classmethod
    def transcribe(
        cls,
        cfg: RuntimeAsrConfig,
        audio: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> list[dict[str, Any]]:
        engine = cls.get_engine(cfg)
        with cls._infer_lock:
            return engine.transcribe(
                audio=audio,
                language=language,
                return_time_stamps=return_time_stamps,
            )

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._engine = None
            cls._config_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HistoryStore:
    def __init__(self, history_path: str | Path, max_entries: int = 500) -> None:
        self.history_path = Path(history_path).expanduser().resolve()
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._lock = threading.Lock()

    def load(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)]
        except Exception:
            return []
        return []

    def save(self, entries: list[dict[str, Any]]) -> None:
        with self._lock:
            data = entries[-self.max_entries :]
            self.history_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def clear(self) -> None:
        self.save([])


class ModelLoadWorker(QObject):
    loaded = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: RuntimeAsrConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            _ = AsrEngineCache.get_engine(self.cfg)
            self.loaded.emit(
                {
                    "model_dir": self.cfg.model_dir,
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=2)}")


class AsrTaskWorker(QObject):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(list)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        inputs: list[str],
        cfg: RuntimeAsrConfig,
        language: str | None,
        return_time_stamps: bool,
        subtitle_dir: str | None,
        ffmpeg_bin: str,
        ffprobe_bin: str,
        chunk_seconds: float = 20.0,
        chunk_enable_threshold_s: float = 70.0,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.cfg = cfg
        self.language = language
        self.return_time_stamps = return_time_stamps
        self.subtitle_dir = subtitle_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.chunk_seconds = max(5.0, float(chunk_seconds))
        self.chunk_enable_threshold_s = max(
            self.chunk_seconds * 2.0,
            float(chunk_enable_threshold_s),
        )
        self._cancel_evt = threading.Event()
        self._timestamps_supported: bool | None = None
        self._timestamp_fallback_warned = False

    def cancel(self) -> None:
        self._cancel_evt.set()

    def _check_cancel(self) -> None:
        if self._cancel_evt.is_set():
            raise RuntimeError("__CANCELLED__")

    def _emit_partial(
        self,
        *,
        index: int,
        total: int,
        source_media: str,
        file_progress: float,
        global_progress: float,
        current_text: str,
        model_time_s: float,
    ) -> None:
        self.progress.emit(
            {
                "event": "file_partial",
                "index": index,
                "total": total,
                "source_media": source_media,
                "file_progress_percent": int(max(0.0, min(1.0, file_progress)) * 100),
                "progress_percent": int(max(0.0, min(1.0, global_progress)) * 100),
                "current_text": current_text,
                "model_inference_time_s": model_time_s,
            }
        )

    def _request_timestamps_for_call(self) -> bool:
        return self.return_time_stamps and self._timestamps_supported is not False

    def _mark_timestamp_fallback(self) -> None:
        self._timestamps_supported = False
        if not self._timestamp_fallback_warned:
            self._timestamp_fallback_warned = True
            self.log.emit(
                "[WARN] forced_aligner 未初始化，时间戳已降级为粗粒度区间（可继续识别并生成字幕）。"
            )

    def _transcribe_row(
        self,
        *,
        audio_path: str,
    ) -> tuple[dict[str, Any], bool, float]:
        request_timestamps = self._request_timestamps_for_call()
        t0 = time.perf_counter()
        row, used_timestamps = transcribe_with_timestamp_fallback(
            lambda return_ts: AsrEngineCache.transcribe(
                cfg=self.cfg,
                audio=audio_path,
                language=self.language,
                return_time_stamps=return_ts,
            ),
            request_timestamps=request_timestamps,
        )
        model_time_s = time.perf_counter() - t0

        if request_timestamps and not used_timestamps:
            self._mark_timestamp_fallback()
        elif request_timestamps and used_timestamps:
            self._timestamps_supported = True
        return row, used_timestamps, model_time_s

    def _transcribe_single(
        self,
        *,
        audio_path: str,
        audio_duration_s: float | None,
    ) -> tuple[str, list[dict[str, Any]], float, str | None]:
        row, used_timestamps, model_time_s = self._transcribe_row(audio_path=audio_path)
        text = str(row.get("text", "") or "")
        segments = (
            build_time_segments(
                row.get("time_stamps") if used_timestamps else [],
                text=text,
                fallback_duration_s=audio_duration_s,
            )
            if self.return_time_stamps
            else []
        )
        return text, segments, model_time_s, row.get("language")

    def _transcribe_chunked_wav(
        self,
        *,
        wav_path: str,
        source_media: str,
        index: int,
        total: int,
        total_duration_s: float | None,
        temp_dir: str,
    ) -> tuple[str, list[dict[str, Any]], float, str | None]:
        text_parts: list[str] = []
        merged_segments: list[dict[str, Any]] = []
        model_time_total = 0.0
        detected_language: str | None = None

        wav_obj = Path(wav_path).resolve()
        with wave.open(str(wav_obj), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()
            chunk_frames = max(1, int(sample_rate * self.chunk_seconds))
            processed_frames = 0
            part = 0

            while True:
                self._check_cancel()
                chunk_data = wav_file.readframes(chunk_frames)
                if not chunk_data:
                    break
                frames_in_chunk = len(chunk_data) // max(1, channels * sample_width)
                if frames_in_chunk <= 0:
                    continue
                part += 1
                chunk_start_s = processed_frames / float(sample_rate)
                chunk_duration_s = frames_in_chunk / float(sample_rate)

                chunk_path = (
                    Path(temp_dir)
                    / f"chunk_{index}_{part}_{uuid.uuid4().hex[:8]}.wav"
                ).resolve()
                with wave.open(str(chunk_path), "wb") as chunk_wav:
                    chunk_wav.setnchannels(channels)
                    chunk_wav.setsampwidth(sample_width)
                    chunk_wav.setframerate(sample_rate)
                    chunk_wav.writeframes(chunk_data)

                row, used_timestamps, chunk_model_time_s = self._transcribe_row(
                    audio_path=str(chunk_path),
                )
                model_time_total += chunk_model_time_s
                if detected_language is None:
                    detected_language = row.get("language")

                piece_text = str(row.get("text", "") or "")
                text_parts.append(piece_text)

                if self.return_time_stamps:
                    piece_segments = build_time_segments(
                        row.get("time_stamps") if used_timestamps else [],
                        text=piece_text,
                        fallback_duration_s=chunk_duration_s,
                    )
                    for seg in piece_segments:
                        merged_segments.append(
                            {
                                "start_time": float(seg.get("start_time", 0.0)) + chunk_start_s,
                                "end_time": float(seg.get("end_time", 0.0)) + chunk_start_s,
                                "text": str(seg.get("text", "") or ""),
                            }
                        )

                processed_frames += frames_in_chunk
                file_ratio = (
                    min(1.0, processed_frames / float(total_frames))
                    if total_frames > 0
                    else 0.0
                )
                global_ratio = ((index - 1) + file_ratio) / float(max(total, 1))
                running_text = "".join(text_parts).strip()
                self._emit_partial(
                    index=index,
                    total=total,
                    source_media=source_media,
                    file_progress=file_ratio,
                    global_progress=global_ratio,
                    current_text=running_text,
                    model_time_s=model_time_total,
                )

        final_text = "".join(text_parts).strip()
        if self.return_time_stamps and not merged_segments and final_text:
            merged_segments = build_time_segments(
                time_stamps=[],
                text=final_text,
                fallback_duration_s=total_duration_s,
            )
        return final_text, merged_segments, model_time_total, detected_language

    def run(self) -> None:
        try:
            media_paths, missing = expand_media_inputs(self.inputs)
            if missing:
                self.log.emit(f"[WARN] 未匹配到文件: {', '.join(missing)}")
            if not media_paths:
                raise RuntimeError("没有可识别的输入文件。")

            self.log.emit(f"[INFO] 正在加载模型: {self.cfg.model_dir}")
            _ = AsrEngineCache.get_engine(self.cfg)
            self.log.emit("[INFO] 模型已就绪，开始识别。")

            results: list[dict[str, Any]] = []
            with tempfile.TemporaryDirectory(prefix="qwen_asr_gui_media_") as temp_dir:
                total = len(media_paths)
                for idx, media_path in enumerate(media_paths, start=1):
                    self._check_cancel()
                    self.progress.emit(
                        {
                            "event": "file_start",
                            "index": idx,
                            "total": total,
                            "source_media": str(Path(media_path).resolve()),
                            "progress_percent": int(((idx - 1) / float(max(total, 1))) * 100),
                            "current_text": "",
                        }
                    )

                    prepared_audio, was_extracted = prepare_audio_input(
                        source_path=media_path,
                        temp_dir=temp_dir,
                        ffmpeg_bin=self.ffmpeg_bin,
                    )
                    audio_duration_s = get_media_duration_seconds(
                        prepared_audio,
                        ffprobe_bin=self.ffprobe_bin,
                    )

                    audio_for_model = prepared_audio
                    use_chunk = False
                    if audio_duration_s is not None and audio_duration_s >= self.chunk_enable_threshold_s:
                        if Path(prepared_audio).suffix.lower() != ".wav":
                            converted = (
                                Path(temp_dir)
                                / f"{Path(prepared_audio).stem}_{uuid.uuid4().hex[:8]}.wav"
                            ).resolve()
                            extract_audio_with_ffmpeg(
                                input_media=prepared_audio,
                                output_wav=converted,
                                ffmpeg_bin=self.ffmpeg_bin,
                            )
                            audio_for_model = str(converted)
                        use_chunk = Path(audio_for_model).suffix.lower() == ".wav"

                    if use_chunk:
                        text, segments, model_time_s, detected_language = self._transcribe_chunked_wav(
                            wav_path=audio_for_model,
                            source_media=str(Path(media_path).resolve()),
                            index=idx,
                            total=total,
                            total_duration_s=audio_duration_s,
                            temp_dir=temp_dir,
                        )
                    else:
                        text, segments, model_time_s, detected_language = self._transcribe_single(
                            audio_path=audio_for_model,
                            audio_duration_s=audio_duration_s,
                        )
                        self._emit_partial(
                            index=idx,
                            total=total,
                            source_media=str(Path(media_path).resolve()),
                            file_progress=1.0,
                            global_progress=idx / float(max(total, 1)),
                            current_text=text,
                            model_time_s=model_time_s,
                        )

                    subtitle_path: str | None = None
                    if self.subtitle_dir and segments:
                        subtitle_root = Path(self.subtitle_dir).expanduser().resolve()
                        subtitle_root.mkdir(parents=True, exist_ok=True)
                        target = subtitle_root / f"{Path(media_path).stem}.srt"
                        save_srt_file(segments, target)
                        subtitle_path = str(target)

                    item = {
                        "event": "file_done",
                        "index": idx,
                        "total": total,
                        "source_media": str(Path(media_path).resolve()),
                        "audio_path": str(Path(audio_for_model).resolve()),
                        "was_extracted": was_extracted,
                        "language": detected_language,
                        "text": text,
                        "audio_duration_s": audio_duration_s,
                        "model_inference_time_s": model_time_s,
                        "segments": segments,
                        "subtitle_path": subtitle_path,
                        "progress_percent": int((idx / float(max(total, 1))) * 100),
                    }
                    self.progress.emit(item)
                    results.append(item)

            self.completed.emit(results)
        except RuntimeError as exc:
            if str(exc) == "__CANCELLED__":
                self.cancelled.emit("用户已取消识别任务。")
                return
            detail = f"{exc}\n{traceback.format_exc(limit=2)}"
            self.failed.emit(detail)
        except Exception as exc:  # noqa: BLE001
            detail = f"{exc}\n{traceback.format_exc(limit=2)}"
            self.failed.emit(detail)


class _StreamPartialWorker(QObject):
    """Worker that transcribes an audio snapshot for streaming partial recognition."""

    result = Signal(str)  # partial transcription text

    def __init__(self, wav_path: str, cfg: "RuntimeAsrConfig", language: str | None) -> None:
        super().__init__()
        self._wav_path = wav_path
        self._cfg = cfg
        self._language = language

    def run(self) -> None:
        try:
            # non-blocking lock: skip if model is already busy
            acquired = AsrEngineCache._infer_lock.acquire(blocking=False)
            if not acquired:
                return
            try:
                engine = AsrEngineCache.get_engine(self._cfg)
                rows = engine.transcribe(
                    audio=self._wav_path,
                    language=self._language,
                    return_time_stamps=False,
                )
                text = ""
                if rows:
                    text = str(rows[0].get("text", "") or "").strip()
                self.result.emit(text)
            finally:
                AsrEngineCache._infer_lock.release()
        except Exception:  # noqa: BLE001
            pass  # silently skip failures for streaming


class DesktopVoiceInputWindow(QMainWindow):
    def __init__(
        self,
        config_path: str,
        history_path: str | None,
        asr_device: str,
        ffmpeg_bin: str,
        ffprobe_bin: str,
        cli_model_size: str | None,
        cli_model_dir: str | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("听键 - 语音输入")
        self.resize(1450, 920)
        self.setMinimumSize(1240, 760)

        self.asr_device = asr_device
        self.ffmpeg_bin = normalize_cli_path(ffmpeg_bin)
        self.ffprobe_bin = normalize_cli_path(ffprobe_bin)

        self.config_store = ConfigStore(config_path)
        self.config = self.config_store.get()
        if cli_model_size:
            self.config.model.asr_model_size = cli_model_size  # type: ignore[assignment]
        if cli_model_dir:
            self.config.model.asr_model_dir = normalize_cli_path(cli_model_dir)
        self.config_store.set(self.config)

        if history_path:
            history_file = history_path
        else:
            rec_dir = normalize_cli_path(self.config.microphone.recording_dir)
            if not rec_dir:
                rec_dir = "recordings"
            rec_dir_path = Path(rec_dir).expanduser()
            if not rec_dir_path.is_absolute():
                rec_dir_path = (Path.cwd() / rec_dir_path).resolve()
            history_file = str((rec_dir_path / "desktop_history.json").resolve())
        self.history_store = HistoryStore(history_file)
        self.history_entries = self.history_store.load()

        self.asr_thread: QThread | None = None
        self.asr_worker: AsrTaskWorker | None = None
        self.asr_busy = False
        self.asr_cancel_requested = False
        self.model_loading = False
        self.model_load_thread: QThread | None = None
        self.model_load_worker: ModelLoadWorker | None = None

        self.audio_source: QAudioSource | None = None
        self.audio_buffer: QBuffer | None = None
        self.record_format: QAudioFormat | None = None
        self.recording_start_t: float | None = None
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(120)
        self.record_timer.timeout.connect(self._tick_record_time)
        self.hotkey_pressed = False
        self.hotkey_started_record = False
        self.insert_after_asr = False
        self._foreground_hwnd: int | None = None  # remember target window for text insertion
        self._focus_hwnd: int | None = None
        self.hotkey_vk_groups: list[tuple[int, ...]] = []
        self.hotkey_poll_timer = QTimer(self)
        self.hotkey_poll_timer.setInterval(18)
        self.hotkey_poll_timer.timeout.connect(self._poll_hotkey_state)
        self._user32 = ctypes.windll.user32 if sys.platform.startswith("win") else None
        self.prompt_bar = FloatingPromptBar()

        # streaming partial recognition during recording
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(2000)
        self._stream_timer.timeout.connect(self._do_stream_snapshot)
        self._stream_thread: QThread | None = None
        self._stream_partial_text = ""

        self.media_devices = QMediaDevices()
        self.media_devices.audioInputsChanged.connect(self._refresh_microphones)

        self._build_ui()
        self._fill_combos()
        self._refresh_microphones()
        self._load_config_to_widgets()
        self._refresh_header_chips()
        self._refresh_hotkey_binding()
        self._render_history()
        self._append_log(
            f"[READY] GUI 已启动，ASR device={self.asr_device}，history={self.history_store.history_path}"
        )
        self.hotkey_poll_timer.start()
        QTimer.singleShot(0, self._preload_model_on_startup)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(200)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 16, 14, 14)
        side_layout.setSpacing(10)

        brand = QLabel("听键")
        brand.setObjectName("Brand")
        side_layout.addWidget(brand)
        brand_sub = QLabel("Desktop Voice Input")
        brand_sub.setObjectName("Subtle")
        side_layout.addWidget(brand_sub)
        side_layout.addSpacing(10)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("NavList")
        self.nav_titles = ["首页", "记忆", "技能", "模型", "设置"]
        for title in self.nav_titles:
            self.nav_list.addItem(title)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        side_layout.addWidget(self.nav_list, 1)

        self.user_center_btn = QPushButton("用户中心")
        self.user_center_btn.setObjectName("SecondaryButton")
        self.user_center_btn.clicked.connect(lambda: self.pages.setCurrentIndex(5))
        side_layout.addWidget(self.user_center_btn)

        outer.addWidget(sidebar, 0)

        panel = QFrame()
        panel.setObjectName("MainPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(12)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._wrap_scroll_page(self._build_home_page()))
        self.pages.addWidget(self._wrap_scroll_page(self._build_memory_page()))
        self.pages.addWidget(self._wrap_scroll_page(self._build_skills_page()))
        self.pages.addWidget(self._wrap_scroll_page(self._build_model_page()))
        self.pages.addWidget(self._wrap_scroll_page(self._build_settings_page()))
        self.pages.addWidget(self._wrap_scroll_page(self._build_user_page()))
        panel_layout.addWidget(self.pages, 1)
        self.nav_list.setCurrentRow(0)

        outer.addWidget(panel, 1)

    def _wrap_scroll_page(self, page: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(page)
        return area

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        title = QLabel("语音输入工作台")
        title.setObjectName("SectionTitle")
        subtitle = QLabel("录音、文件识别、字幕导出统一在一个工作流中完成。")
        subtitle.setObjectName("Subtle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        hero_card = QFrame()
        hero_card.setObjectName("HeroCard")
        hero_layout = QGridLayout(hero_card)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setHorizontalSpacing(16)
        hero_layout.setVerticalSpacing(8)
        hero_layout.setColumnStretch(0, 3)
        hero_layout.setColumnStretch(1, 2)

        hero_title = QLabel("高性能本地语音输入引擎")
        hero_title.setObjectName("HeroTitle")
        hero_sub = QLabel(
            "启动即预加载模型，支持录音与音频/视频文件识别。识别任务可随时中断，长任务实时显示进度与文本。"
        )
        hero_sub.setObjectName("HeroSub")
        hero_sub.setWordWrap(True)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(hero_title)
        left.addWidget(hero_sub)
        hero_layout.addLayout(left, 0, 0)

        hero_info_col = QVBoxLayout()
        hero_info_col.setSpacing(6)
        self.hero_model_chip = QLabel("模型: --")
        self.hero_model_chip.setObjectName("HeroChip")
        self.hero_backend_chip = QLabel("后端: --")
        self.hero_backend_chip.setObjectName("HeroChip")
        hero_info_col.addWidget(self.hero_model_chip)
        hero_info_col.addWidget(self.hero_backend_chip)
        hero_info_col.addStretch(1)
        hero_layout.addLayout(hero_info_col, 0, 1)
        layout.addWidget(hero_card)

        action_card = QFrame()
        action_card.setObjectName("ActionCard")
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(14, 14, 14, 14)
        action_layout.setSpacing(10)

        action_title = QLabel("操作")
        action_title.setObjectName("CardTitle")
        action_layout.addWidget(action_title)

        chips_row = QHBoxLayout()
        self.home_hotkey_chip = QLabel("快捷键: --")
        self.home_hotkey_chip.setObjectName("Chip")
        self.home_short_chip = QLabel("短按: --")
        self.home_short_chip.setObjectName("Chip")
        self.home_long_chip = QLabel("长按: --")
        self.home_long_chip.setObjectName("Chip")
        chips_row.addWidget(self.home_hotkey_chip)
        chips_row.addWidget(self.home_short_chip)
        chips_row.addWidget(self.home_long_chip)
        chips_row.addStretch(1)
        action_layout.addLayout(chips_row)

        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(10)
        action_grid.setVerticalSpacing(10)
        action_grid.setColumnStretch(0, 1)
        action_grid.setColumnStretch(1, 1)
        action_grid.setColumnStretch(2, 1)

        self.btn_start_record = QPushButton("开始录音")
        self.btn_start_record.setObjectName("PrimaryButton")
        self.btn_start_record.clicked.connect(self._start_record)

        self.btn_stop_record = QPushButton("停止并识别")
        self.btn_stop_record.setObjectName("SecondaryButton")
        self.btn_stop_record.setEnabled(False)
        self.btn_stop_record.clicked.connect(self._stop_record_and_transcribe)

        self.btn_pick_files = QPushButton("选择音频/视频文件")
        self.btn_pick_files.setObjectName("SecondaryButton")
        self.btn_pick_files.clicked.connect(self._pick_files_and_transcribe)

        self.btn_cancel_asr = QPushButton("中断识别")
        self.btn_cancel_asr.setObjectName("DangerButton")
        self.btn_cancel_asr.setEnabled(False)
        self.btn_cancel_asr.clicked.connect(self._cancel_asr_clicked)

        for button in (
            self.btn_start_record,
            self.btn_stop_record,
            self.btn_pick_files,
            self.btn_cancel_asr,
        ):
            button.setMinimumHeight(40)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        action_grid.addWidget(self.btn_start_record, 0, 0)
        action_grid.addWidget(self.btn_stop_record, 0, 1)
        action_grid.addWidget(self.btn_pick_files, 0, 2)
        action_grid.addWidget(self.btn_cancel_asr, 1, 2)
        action_layout.addLayout(action_grid)

        pattern_row = QHBoxLayout()
        self.input_patterns = QLineEdit()
        self.input_patterns.setPlaceholderText(
            "支持多个路径和通配符，例如: *.wav, C:\\path\\*.mp4"
        )
        self.btn_run_patterns = QPushButton("运行输入")
        self.btn_run_patterns.setObjectName("SecondaryButton")
        self.btn_run_patterns.setMinimumHeight(40)
        self.btn_run_patterns.clicked.connect(self._run_patterns)
        pattern_row.addWidget(self.input_patterns, 1)
        pattern_row.addWidget(self.btn_run_patterns)
        action_layout.addLayout(pattern_row)

        language_row = QHBoxLayout()
        self.run_language_combo = QComboBox()
        self.run_language_combo.setMinimumWidth(160)
        self.chk_return_ts = QCheckBox("输出时间戳")
        self.chk_make_subtitle = QCheckBox("生成字幕(.srt)")
        self.subtitle_dir_edit = QLineEdit()
        self.subtitle_dir_edit.setPlaceholderText("字幕目录(可选)")
        self.subtitle_dir_pick_btn = QPushButton("目录")
        self.subtitle_dir_pick_btn.setObjectName("SecondaryButton")
        self.subtitle_dir_pick_btn.setMinimumWidth(78)
        self.subtitle_dir_pick_btn.clicked.connect(self._pick_subtitle_dir)
        language_row.addWidget(QLabel("语言"))
        language_row.addWidget(self.run_language_combo)
        language_row.addWidget(self.chk_return_ts)
        language_row.addWidget(self.chk_make_subtitle)
        language_row.addStretch(1)
        action_layout.addLayout(language_row)

        subtitle_row = QHBoxLayout()
        subtitle_row.addWidget(self.subtitle_dir_edit, 1)
        subtitle_row.addWidget(self.subtitle_dir_pick_btn)
        action_layout.addLayout(subtitle_row)
        layout.addWidget(action_card)

        status_card = QFrame()
        status_card.setObjectName("StatusCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(14, 14, 14, 14)
        status_layout.setSpacing(10)

        status_title = QLabel("状态")
        status_title.setObjectName("CardTitle")
        status_layout.addWidget(status_title)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(8)
        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)

        self.label_record_state = QLabel("状态: 空闲")
        self.label_recording_time = QLabel("录音时长: 00:00:00.000")
        self.label_audio_duration = QLabel("音频时长: N/A")
        self.label_model_time = QLabel("识别耗时: N/A")
        for label in (
            self.label_record_state,
            self.label_recording_time,
            self.label_audio_duration,
            self.label_model_time,
        ):
            label.setObjectName("Chip")

        metrics_grid.addWidget(self.label_record_state, 0, 0)
        metrics_grid.addWidget(self.label_recording_time, 0, 1)
        metrics_grid.addWidget(self.label_audio_duration, 1, 0)
        metrics_grid.addWidget(self.label_model_time, 1, 1)
        status_layout.addLayout(metrics_grid)

        progress_row = QHBoxLayout()
        self.label_asr_progress = QLabel("进度: 空闲")
        self.label_asr_progress.setObjectName("Chip")
        self.asr_progress_bar = QProgressBar()
        self.asr_progress_bar.setRange(0, 100)
        self.asr_progress_bar.setValue(0)
        self.asr_progress_bar.setFormat("%p%")
        progress_row.addWidget(self.label_asr_progress)
        progress_row.addWidget(self.asr_progress_bar, 1)
        status_layout.addLayout(progress_row)

        self.current_partial_view = QPlainTextEdit()
        self.current_partial_view.setObjectName("LiveText")
        self.current_partial_view.setReadOnly(True)
        self.current_partial_view.setMinimumHeight(110)
        self.current_partial_view.setMaximumHeight(140)
        self.current_partial_view.setPlaceholderText("当前识别到的内容会实时显示在这里")
        status_layout.addWidget(self.current_partial_view)
        layout.addWidget(status_card)

        result_card = QFrame()
        result_card.setObjectName("ResultCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(14, 14, 14, 14)
        result_layout.setSpacing(10)

        header = QHBoxLayout()
        header_label = QLabel("结果")
        header_label.setObjectName("CardTitle")

        self.btn_clear_history = QPushButton("清空历史")
        self.btn_clear_history.setObjectName("SecondaryButton")
        self.btn_clear_history.clicked.connect(self._clear_history)

        self.btn_toggle_log = QPushButton("展开日志")
        self.btn_toggle_log.setObjectName("SecondaryButton")
        self.btn_toggle_log.clicked.connect(self._toggle_log_panel)

        self.btn_clear_output = QPushButton("清空日志")
        self.btn_clear_output.setObjectName("SecondaryButton")
        self.btn_clear_output.clicked.connect(self._clear_output)

        header.addWidget(header_label)
        header.addStretch(1)
        header.addWidget(self.btn_clear_history)
        header.addWidget(self.btn_toggle_log)
        header.addWidget(self.btn_clear_output)
        result_layout.addLayout(header)

        history_label = QLabel("最近识别记录")
        history_label.setObjectName("Subtle")
        result_layout.addWidget(history_label)

        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        self.history_list.itemClicked.connect(self._open_history_item)
        self.history_list.setMinimumHeight(220)
        result_layout.addWidget(self.history_list)

        self.output_view = QPlainTextEdit()
        self.output_view.setObjectName("LogView")
        self.output_view.setReadOnly(True)
        self.output_view.setMinimumHeight(150)
        result_layout.addWidget(self.output_view)

        self._set_log_panel_expanded(False)
        layout.addWidget(result_card, 1)

        return page

    def _build_memory_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("记忆")
        title.setObjectName("SectionTitle")
        subtitle = QLabel("用于管理用户词典与个人记忆空间")
        subtitle.setObjectName("Subtle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        g = QGroupBox("用户词典")
        gl = QVBoxLayout(g)
        info = QLabel("在“技能”页启用“加载用户词典”后，识别会优先考虑你的术语。")
        info.setWordWrap(True)
        gl.addWidget(info)
        btn = QPushButton("打开技能页")
        btn.clicked.connect(lambda: self.nav_list.setCurrentRow(2))
        gl.addWidget(btn, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(g)
        layout.addStretch(1)
        return page

    def _build_skills_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("技能")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        box = QGroupBox("语音输入增强")
        box_l = QVBoxLayout(box)
        box_l.setSpacing(10)

        self.skills_auto_run_chk = QCheckBox("自动运行")
        self.skills_enhancement_chk = QCheckBox("启用识别文本增强")
        self.skills_enhancement_chk.setChecked(True)

        preset_row = QHBoxLayout()
        self.skills_preset_combo = QComboBox()
        self.skills_preset_combo.setMinimumWidth(220)
        preset_row.addWidget(QLabel("增强预设"))
        preset_row.addWidget(self.skills_preset_combo)
        preset_row.addStretch(1)

        rules_grid = QGridLayout()
        rules_grid.setHorizontalSpacing(12)
        rules_grid.setVerticalSpacing(8)
        self.skills_remove_fillers_chk = QCheckBox("去口头词")
        self.skills_auto_punct_chk = QCheckBox("自动补句末标点")
        self.skills_dedupe_chk = QCheckBox("去重重复词/短句")
        self.skills_spacing_chk = QCheckBox("中英空格规范")
        rules_grid.addWidget(self.skills_remove_fillers_chk, 0, 0)
        rules_grid.addWidget(self.skills_auto_punct_chk, 0, 1)
        rules_grid.addWidget(self.skills_dedupe_chk, 1, 0)
        rules_grid.addWidget(self.skills_spacing_chk, 1, 1)

        self.skills_prompt_edit = QTextEdit()
        self.skills_prompt_edit.setPlaceholderText(
            "可选个性化规则，例如：去掉结尾最后一个句号 / 不要标点 / 每句换行"
        )
        self.skills_prompt_edit.setFixedHeight(96)
        self.skills_personalize_chk = QCheckBox("启用个性化提示词")

        self.skills_dict_chk = QCheckBox("加载用户词典")
        self.skills_dict_edit = QPlainTextEdit()
        self.skills_dict_edit.setPlaceholderText(
            "每行一条：原词=替换词\n例如：\nQ 文=Qwen\n扣大=CUDA\n发发mpeg=ffmpeg"
        )
        self.skills_dict_edit.setFixedHeight(120)
        self.skills_auto_structure_chk = QCheckBox("自动结构化")

        box_l.addWidget(self.skills_auto_run_chk)
        box_l.addWidget(self.skills_enhancement_chk)
        box_l.addLayout(preset_row)
        box_l.addLayout(rules_grid)
        box_l.addWidget(self.skills_personalize_chk)
        box_l.addWidget(self.skills_prompt_edit)
        box_l.addWidget(self.skills_dict_chk)
        box_l.addWidget(self.skills_dict_edit)
        box_l.addWidget(self.skills_auto_structure_chk)

        self.skills_enhancement_chk.toggled.connect(self._refresh_skills_controls)
        self.skills_dict_chk.toggled.connect(self._refresh_skills_controls)
        self.skills_personalize_chk.toggled.connect(self._refresh_skills_controls)

        self.btn_save_skills = QPushButton("保存技能设置")
        self.btn_save_skills.setObjectName("PrimaryButton")
        self.btn_save_skills.clicked.connect(self._save_all_settings)
        box_l.addWidget(self.btn_save_skills, 0, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("模型")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        model_box = QGroupBox("语音识别模型")
        grid = QGridLayout(model_box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.model_size_combo = QComboBox()
        self.model_dir_edit = QLineEdit()
        self.model_dir_pick_btn = QPushButton("选择目录")
        self.model_dir_pick_btn.clicked.connect(self._pick_model_dir)
        self.model_backend_combo = QComboBox()
        self.model_dtype_combo = QComboBox()
        self.model_attn_combo = QComboBox()
        self.model_batch_spin = QSpinBox()
        self.model_batch_spin.setRange(1, 64)
        self.model_token_spin = QSpinBox()
        self.model_token_spin.setRange(16, 4096)
        self.model_tf32_chk = QCheckBox("启用 TF32")
        self.model_cudnn_chk = QCheckBox("启用 cudnn benchmark")
        self.model_final_asr_chk = QCheckBox("停止录音后自动终稿识别")

        grid.addWidget(QLabel("模型规格"), 0, 0)
        grid.addWidget(self.model_size_combo, 0, 1)
        grid.addWidget(QLabel("模型目录"), 1, 0)
        grid.addWidget(self.model_dir_edit, 1, 1)
        grid.addWidget(self.model_dir_pick_btn, 1, 2)
        grid.addWidget(QLabel("推理后端"), 2, 0)
        grid.addWidget(self.model_backend_combo, 2, 1)
        grid.addWidget(QLabel("dtype"), 3, 0)
        grid.addWidget(self.model_dtype_combo, 3, 1)
        grid.addWidget(QLabel("attention"), 4, 0)
        grid.addWidget(self.model_attn_combo, 4, 1)
        grid.addWidget(QLabel("max_inference_batch_size"), 5, 0)
        grid.addWidget(self.model_batch_spin, 5, 1)
        grid.addWidget(QLabel("max_new_tokens"), 6, 0)
        grid.addWidget(self.model_token_spin, 6, 1)
        grid.addWidget(self.model_tf32_chk, 7, 1)
        grid.addWidget(self.model_cudnn_chk, 8, 1)
        grid.addWidget(self.model_final_asr_chk, 9, 1)

        self.model_warning = QLabel("")
        self.model_warning.setStyleSheet("color: #b91c1c;")
        grid.addWidget(self.model_warning, 10, 0, 1, 3)

        btn_row = QHBoxLayout()
        self.btn_save_model = QPushButton("保存模型设置")
        self.btn_save_model.setObjectName("PrimaryButton")
        self.btn_save_model.clicked.connect(self._save_all_settings)
        self.btn_release_model = QPushButton("释放模型缓存")
        self.btn_release_model.clicked.connect(self._release_model_cache)
        btn_row.addWidget(self.btn_save_model)
        btn_row.addWidget(self.btn_release_model)
        btn_row.addStretch(1)
        grid.addLayout(btn_row, 11, 0, 1, 3)

        layout.addWidget(model_box)

        llm_box = QGroupBox("大模型配置")
        ll = QGridLayout(llm_box)
        self.llm_provider_edit = QLineEdit()
        self.llm_provider_edit.setPlaceholderText("例如: openai / qwen / local")
        self.llm_model_edit = QLineEdit()
        self.llm_model_edit.setPlaceholderText("例如: gpt-4.1 / qwen-plus")
        ll.addWidget(QLabel("Provider"), 0, 0)
        ll.addWidget(self.llm_provider_edit, 0, 1)
        ll.addWidget(QLabel("Model"), 1, 0)
        ll.addWidget(self.llm_model_edit, 1, 1)
        layout.addWidget(llm_box)
        layout.addStretch(1)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("设置")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        tabs = QTabWidget()

        quick_tab = QWidget()
        quick_layout = QVBoxLayout(quick_tab)
        quick_layout.setSpacing(12)
        quick_box = QGroupBox("快捷键")
        quick_grid = QGridLayout(quick_box)
        self.shortcut_hotkey_edit = QLineEdit()
        self.shortcut_short_combo = QComboBox()
        self.shortcut_long_combo = QComboBox()
        self.shortcut_smart_mouse_chk = QCheckBox("智能鼠标模式(BETA)")
        hotkey_hint = QLabel("语音输入热键采用按住录音、松开即识别的 Push-to-Talk 模式。")
        hotkey_hint.setObjectName("Subtle")
        quick_grid.addWidget(QLabel("热键"), 0, 0)
        quick_grid.addWidget(self.shortcut_hotkey_edit, 0, 1)
        quick_grid.addWidget(QLabel("短按"), 1, 0)
        quick_grid.addWidget(self.shortcut_short_combo, 1, 1)
        quick_grid.addWidget(QLabel("长按"), 2, 0)
        quick_grid.addWidget(self.shortcut_long_combo, 2, 1)
        quick_grid.addWidget(self.shortcut_smart_mouse_chk, 3, 1)
        quick_grid.addWidget(hotkey_hint, 4, 0, 1, 2)
        quick_layout.addWidget(quick_box)
        quick_layout.addStretch(1)
        tabs.addTab(quick_tab, "快捷键")

        mic_tab = QWidget()
        mic_layout = QVBoxLayout(mic_tab)
        mic_box = QGroupBox("麦克风")
        mic_grid = QGridLayout(mic_box)
        self.micro_device_combo = QComboBox()
        self.btn_refresh_micro = QPushButton("刷新设备")
        self.btn_refresh_micro.clicked.connect(self._refresh_microphones)
        self.micro_mute_chk = QCheckBox("录音时静音")
        self.micro_record_dir_edit = QLineEdit()
        self.btn_pick_record_dir = QPushButton("选择录音目录")
        self.btn_pick_record_dir.clicked.connect(self._pick_recording_dir)
        self.btn_open_record_dir = QPushButton("打开录音目录")
        self.btn_open_record_dir.clicked.connect(self._open_recording_dir)

        mic_grid.addWidget(QLabel("设备"), 0, 0)
        mic_grid.addWidget(self.micro_device_combo, 0, 1)
        mic_grid.addWidget(self.btn_refresh_micro, 0, 2)
        mic_grid.addWidget(self.micro_mute_chk, 1, 1)
        mic_grid.addWidget(QLabel("录音目录"), 2, 0)
        mic_grid.addWidget(self.micro_record_dir_edit, 2, 1)
        mic_grid.addWidget(self.btn_pick_record_dir, 2, 2)
        mic_grid.addWidget(self.btn_open_record_dir, 3, 2)
        mic_layout.addWidget(mic_box)
        mic_layout.addStretch(1)
        tabs.addTab(mic_tab, "麦克风")

        sys_tab = QWidget()
        sys_layout = QVBoxLayout(sys_tab)
        sys_box = QGroupBox("系统")
        sys_grid = QGridLayout(sys_box)
        self.sys_auto_start_chk = QCheckBox("开机自动启动")
        self.sys_silent_chk = QCheckBox("静默启动")
        self.sys_language_combo = QComboBox()
        self.sys_logs_chk = QCheckBox("启用日志")
        sys_grid.addWidget(self.sys_auto_start_chk, 0, 0, 1, 2)
        sys_grid.addWidget(self.sys_silent_chk, 1, 0, 1, 2)
        sys_grid.addWidget(QLabel("转写语言"), 2, 0)
        sys_grid.addWidget(self.sys_language_combo, 2, 1)
        sys_grid.addWidget(self.sys_logs_chk, 3, 0, 1, 2)
        sys_layout.addWidget(sys_box)
        sys_layout.addStretch(1)
        tabs.addTab(sys_tab, "系统")

        lab_tab = QWidget()
        lab_layout = QVBoxLayout(lab_tab)
        lab_box = QGroupBox("实验室")
        lab_grid = QGridLayout(lab_box)
        self.lab_no_clip_chk = QCheckBox("不使用剪贴板输入")
        self.lab_norm_chk = QCheckBox("文本正规化")
        lab_grid.addWidget(self.lab_no_clip_chk, 0, 0, 1, 2)
        lab_grid.addWidget(self.lab_norm_chk, 1, 0, 1, 2)
        lab_layout.addWidget(lab_box)
        lab_layout.addStretch(1)
        tabs.addTab(lab_tab, "实验室")

        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_box = QGroupBox("关于")
        ab = QVBoxLayout(about_box)
        logo = QLabel("听键")
        logo.setStyleSheet("font-size: 38px; font-weight: 800; color: #0f172a;")
        info = QLabel(
            "本地离线语音输入 GUI\n"
            "模型: Qwen3-ASR 0.6B / 1.7B\n"
            "后端: transformers (Windows 可用)"
        )
        info.setWordWrap(True)
        btn_site = QPushButton("打开项目目录")
        btn_site.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path.cwd())))
        )
        btn_privacy = QPushButton("隐私政策(占位)")
        btn_privacy.clicked.connect(
            lambda: QMessageBox.information(
                self, "提示", "当前为本地离线版本，未接入在线隐私政策页面。"
            )
        )
        ab.addWidget(logo)
        ab.addWidget(info)
        ab.addWidget(btn_site, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        ab.addWidget(btn_privacy, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        about_layout.addWidget(about_box)
        about_layout.addStretch(1)
        tabs.addTab(about_tab, "关于")

        layout.addWidget(tabs)
        self.btn_save_settings = QPushButton("保存设置")
        self.btn_save_settings.setObjectName("PrimaryButton")
        self.btn_save_settings.clicked.connect(self._save_all_settings)
        layout.addWidget(self.btn_save_settings, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return page

    def _build_user_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("用户中心")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        box = QGroupBox("账户")
        bl = QVBoxLayout(box)
        bl.addWidget(QLabel("当前版本为本地离线版，暂未接入在线账户系统。"))
        bl.addWidget(QLabel(f"配置文件: {self.config_store.config_path}"))
        bl.addWidget(QLabel(f"历史文件: {self.history_store.history_path}"))
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _fill_combos(self) -> None:
        for label, value in INPUT_LANGUAGE_HINT_CHOICES:
            self.run_language_combo.addItem(label, value)

        for label, value in ACTION_CHOICES:
            self.shortcut_short_combo.addItem(label, value)
            self.shortcut_long_combo.addItem(label, value)

        for label, value in SKILL_ENHANCE_PRESET_CHOICES:
            self.skills_preset_combo.addItem(label, value)

        self.model_size_combo.addItem("0.6B", "0.6b")
        self.model_size_combo.addItem("1.7B", "1.7b")
        self.model_backend_combo.addItem("transformers", "transformers")
        self.model_backend_combo.addItem("vllm (Linux/WSL2)", "vllm")
        self.model_dtype_combo.addItem("auto", "auto")
        self.model_dtype_combo.addItem("float16", "float16")
        self.model_dtype_combo.addItem("bfloat16", "bfloat16")
        self.model_dtype_combo.addItem("float32", "float32")
        self.model_attn_combo.addItem("auto", "auto")
        self.model_attn_combo.addItem("flash_attention_2", "flash_attention_2")
        self.model_attn_combo.addItem("sdpa", "sdpa")
        self.model_attn_combo.addItem("eager", "eager")
        self.model_backend_combo.currentIndexChanged.connect(self._refresh_backend_warning)

        for label, value in SYSTEM_LANGUAGE_CHOICES:
            self.sys_language_combo.addItem(label, value)

    def _set_combo_value(self, combo: QComboBox, value: Any) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _combo_value(self, combo: QComboBox) -> Any:
        return combo.itemData(combo.currentIndex())

    def _refresh_skills_controls(self) -> None:
        enhance_enabled = self.skills_enhancement_chk.isChecked()
        personalize_enabled = self.skills_personalize_chk.isChecked()
        dict_enabled = self.skills_dict_chk.isChecked()

        self.skills_preset_combo.setEnabled(enhance_enabled)
        self.skills_remove_fillers_chk.setEnabled(enhance_enabled)
        self.skills_auto_punct_chk.setEnabled(enhance_enabled)
        self.skills_dedupe_chk.setEnabled(enhance_enabled)
        self.skills_spacing_chk.setEnabled(enhance_enabled)
        self.skills_personalize_chk.setEnabled(enhance_enabled)
        self.skills_auto_structure_chk.setEnabled(enhance_enabled)

        self.skills_prompt_edit.setEnabled(enhance_enabled and personalize_enabled)
        self.skills_dict_chk.setEnabled(enhance_enabled)
        self.skills_dict_edit.setEnabled(enhance_enabled and dict_enabled)

    def _load_config_to_widgets(self) -> None:
        c = self.config
        self.shortcut_hotkey_edit.setText(c.shortcuts.hotkey)
        self._set_combo_value(self.shortcut_short_combo, c.shortcuts.short_press_action)
        self._set_combo_value(self.shortcut_long_combo, c.shortcuts.long_press_action)
        self.shortcut_smart_mouse_chk.setChecked(c.shortcuts.smart_mouse_mode)

        self.micro_mute_chk.setChecked(c.microphone.mute_during_recording)
        self.micro_record_dir_edit.setText(c.microphone.recording_dir)

        self.sys_auto_start_chk.setChecked(c.system.auto_start)
        self.sys_silent_chk.setChecked(c.system.silent_start)
        self._set_combo_value(self.sys_language_combo, c.system.transcription_language)
        self.sys_logs_chk.setChecked(c.system.enable_logs)

        self.lab_no_clip_chk.setChecked(c.lab.direct_input_no_clipboard)
        self.lab_norm_chk.setChecked(c.lab.text_normalization)

        self.skills_auto_run_chk.setChecked(c.skills.auto_run)
        self.skills_enhancement_chk.setChecked(c.skills.enhancement_enabled)
        self._set_combo_value(self.skills_preset_combo, c.skills.enhancement_preset)
        self.skills_remove_fillers_chk.setChecked(c.skills.remove_fillers)
        self.skills_auto_punct_chk.setChecked(c.skills.auto_punctuation)
        self.skills_dedupe_chk.setChecked(c.skills.dedupe_repeats)
        self.skills_spacing_chk.setChecked(c.skills.normalize_spacing)
        self.skills_personalize_chk.setChecked(c.skills.personalize_enabled)
        self.skills_prompt_edit.setPlainText(c.skills.personalize_prompt)
        self.skills_dict_chk.setChecked(c.skills.load_user_dictionary)
        self.skills_dict_edit.setPlainText(c.skills.user_dictionary_text)
        self.skills_auto_structure_chk.setChecked(c.skills.auto_structure)
        self._refresh_skills_controls()

        self._set_combo_value(self.model_size_combo, c.model.asr_model_size)
        self.model_dir_edit.setText(c.model.asr_model_dir)
        self._set_combo_value(self.model_backend_combo, c.model.asr_backend)
        self._set_combo_value(self.model_dtype_combo, c.model.asr_dtype)
        self._set_combo_value(self.model_attn_combo, c.model.asr_attn_implementation)
        self.model_batch_spin.setValue(c.model.asr_max_inference_batch_size)
        self.model_token_spin.setValue(c.model.asr_max_new_tokens)
        self.model_tf32_chk.setChecked(c.model.asr_enable_tf32)
        self.model_cudnn_chk.setChecked(c.model.asr_cudnn_benchmark)
        self.model_final_asr_chk.setChecked(c.model.enable_final_asr)
        self.llm_provider_edit.setText(c.model.llm_provider or "")
        self.llm_model_edit.setText(c.model.llm_model or "")
        self._refresh_backend_warning()

        self._set_combo_value(
            self.run_language_combo,
            self._hint_from_system_language(c.system.transcription_language),
        )

    def _sync_widgets_to_config(self) -> None:
        c = self.config
        c.shortcuts.hotkey = self.shortcut_hotkey_edit.text().strip() or "LeftCtrl+LeftWin"
        c.shortcuts.short_press_action = self._combo_value(self.shortcut_short_combo)
        c.shortcuts.long_press_action = self._combo_value(self.shortcut_long_combo)
        c.shortcuts.smart_mouse_mode = self.shortcut_smart_mouse_chk.isChecked()

        c.microphone.device = str(self.micro_device_combo.currentData() or "auto")
        c.microphone.mute_during_recording = self.micro_mute_chk.isChecked()
        c.microphone.recording_dir = self.micro_record_dir_edit.text().strip() or "recordings"

        c.system.auto_start = self.sys_auto_start_chk.isChecked()
        c.system.silent_start = self.sys_silent_chk.isChecked()
        c.system.transcription_language = self._combo_value(self.sys_language_combo)
        c.system.enable_logs = self.sys_logs_chk.isChecked()

        c.lab.direct_input_no_clipboard = self.lab_no_clip_chk.isChecked()
        c.lab.text_normalization = self.lab_norm_chk.isChecked()

        c.skills.auto_run = self.skills_auto_run_chk.isChecked()
        c.skills.enhancement_enabled = self.skills_enhancement_chk.isChecked()
        c.skills.enhancement_preset = self._combo_value(self.skills_preset_combo)
        c.skills.remove_fillers = self.skills_remove_fillers_chk.isChecked()
        c.skills.auto_punctuation = self.skills_auto_punct_chk.isChecked()
        c.skills.dedupe_repeats = self.skills_dedupe_chk.isChecked()
        c.skills.normalize_spacing = self.skills_spacing_chk.isChecked()
        c.skills.personalize_enabled = self.skills_personalize_chk.isChecked()
        c.skills.personalize_prompt = self.skills_prompt_edit.toPlainText().strip()
        c.skills.load_user_dictionary = self.skills_dict_chk.isChecked()
        c.skills.user_dictionary_text = self.skills_dict_edit.toPlainText().strip()
        c.skills.auto_structure = self.skills_auto_structure_chk.isChecked()

        c.model.asr_model_size = self._combo_value(self.model_size_combo)
        c.model.asr_model_dir = self.model_dir_edit.text().strip() or "model"
        c.model.asr_backend = self._combo_value(self.model_backend_combo)
        c.model.asr_dtype = self._combo_value(self.model_dtype_combo)
        c.model.asr_attn_implementation = self._combo_value(self.model_attn_combo)
        c.model.asr_max_inference_batch_size = int(self.model_batch_spin.value())
        c.model.asr_max_new_tokens = int(self.model_token_spin.value())
        c.model.asr_enable_tf32 = self.model_tf32_chk.isChecked()
        c.model.asr_cudnn_benchmark = self.model_cudnn_chk.isChecked()
        c.model.enable_final_asr = self.model_final_asr_chk.isChecked()
        provider = self.llm_provider_edit.text().strip()
        llm_model = self.llm_model_edit.text().strip()
        c.model.llm_provider = provider or None
        c.model.llm_model = llm_model or None

    def _save_all_settings(self, silent: bool = False) -> None:
        old_runtime = build_runtime_config(self.config, self.asr_device).key()
        self._sync_widgets_to_config()
        self.config_store.set(self.config)
        new_runtime = build_runtime_config(self.config, self.asr_device).key()
        if old_runtime != new_runtime:
            AsrEngineCache.clear()
            self._append_log("[INFO] 模型参数已改变，已释放旧模型缓存。")
            self._start_model_preload(trigger="配置变更")
        self._refresh_hotkey_binding()
        self._refresh_header_chips()
        if not silent:
            self.statusBar().showMessage("设置已保存", 3000)

    def _preload_model_on_startup(self) -> None:
        self._start_model_preload(trigger="启动预加载")

    def _start_model_preload(self, trigger: str) -> None:
        if self.model_loading or self.asr_busy or self.audio_source is not None:
            return

        cfg = build_runtime_config(self.config, self.asr_device)
        self.model_loading = True
        self.label_record_state.setText("状态: 模型加载中")
        self._set_asr_progress(None, f"进度: 模型加载中 ({cfg.model_size})", "")
        self._set_controls_busy(self.asr_busy)
        self._append_log(f"[MODEL] {trigger}: loading {cfg.model_dir}")

        worker = ModelLoadWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._on_model_preload_loaded)
        worker.failed.connect(self._on_model_preload_failed)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_model_preload_finished)
        thread.finished.connect(thread.deleteLater)

        self.model_load_worker = worker
        self.model_load_thread = thread
        thread.start()

    def _on_model_preload_loaded(self, payload: dict[str, Any]) -> None:
        model_dir = payload.get("model_dir", "")
        elapsed = payload.get("elapsed_s")
        self._append_log(
            f"[MODEL] loaded: {model_dir} in {format_hms(float(elapsed) if elapsed is not None else None)}"
        )
        self._set_asr_progress(0, "进度: 模型已就绪", None)
        self.label_record_state.setText("状态: 空闲")

    def _on_model_preload_failed(self, detail: str) -> None:
        self._append_log(f"[MODEL][ERROR] {detail}")
        self._set_asr_progress(0, "进度: 模型加载失败", None)
        self.label_record_state.setText("状态: 模型加载失败")
        QMessageBox.critical(self, "模型加载失败", detail.splitlines()[0] if detail else "Unknown error")

    def _on_model_preload_finished(self) -> None:
        self.model_loading = False
        self.model_load_worker = None
        self.model_load_thread = None
        self._set_controls_busy(self.asr_busy)

    def _refresh_header_chips(self) -> None:
        self.home_hotkey_chip.setText(f"快捷键: {self.config.shortcuts.hotkey}")
        self.home_short_chip.setText(f"短按: {self.config.shortcuts.short_press_action}")
        self.home_long_chip.setText(f"长按: {self.config.shortcuts.long_press_action}")
        self.hero_model_chip.setText(f"模型: {self.config.model.asr_model_size}")
        self.hero_backend_chip.setText(f"后端: {self.config.model.asr_backend}")

    def _refresh_hotkey_binding(self) -> None:
        self.hotkey_vk_groups = self._parse_hotkey_groups(self.config.shortcuts.hotkey)
        if self.config.shortcuts.hotkey.strip() and not self.hotkey_vk_groups:
            self._append_log(
                f"[WARN] 无法解析快捷键 `{self.config.shortcuts.hotkey}`。当前仅支持修饰键 + A-Z / 0-9 / F1-F24。"
            )

    def _parse_hotkey_groups(self, hotkey_text: str) -> list[tuple[int, ...]]:
        groups: list[tuple[int, ...]] = []
        for raw_part in hotkey_text.split("+"):
            part = raw_part.strip()
            if not part:
                continue
            norm = part.replace(" ", "").lower()
            if norm in WINDOWS_VK_GROUPS:
                groups.append(WINDOWS_VK_GROUPS[norm])
                continue
            if len(part) == 1 and part.isalpha():
                groups.append((ord(part.upper()),))
                continue
            if len(part) == 1 and part.isdigit():
                groups.append((ord(part),))
                continue
            if norm.startswith("f") and norm[1:].isdigit():
                index = int(norm[1:])
                if 1 <= index <= 24:
                    groups.append((0x70 + index - 1,))
        return groups

    def _is_vk_pressed(self, vk_code: int) -> bool:
        if self._user32 is None:
            return False
        return bool(self._user32.GetAsyncKeyState(vk_code) & 0x8000)

    def _is_hotkey_pressed_now(self) -> bool:
        if not self.hotkey_vk_groups:
            return False
        return all(any(self._is_vk_pressed(vk) for vk in group) for group in self.hotkey_vk_groups)

    def _hotkey_voice_input_enabled(self) -> bool:
        return (
            self.config.shortcuts.short_press_action == "voice_input"
            or self.config.shortcuts.long_press_action == "voice_input"
        )

    def _poll_hotkey_state(self) -> None:
        if self._user32 is None:
            return
        pressed_now = self._is_hotkey_pressed_now()
        if pressed_now and not self.hotkey_pressed:
            self.hotkey_pressed = True
            self._on_hotkey_pressed()
            return
        if (not pressed_now) and self.hotkey_pressed:
            self.hotkey_pressed = False
            self._on_hotkey_released()

    def _on_hotkey_pressed(self) -> None:
        if not self._hotkey_voice_input_enabled():
            return
        if self.model_loading or self.asr_busy or self.audio_source is not None:
            return
        # remember foreground window for text insertion later
        if self._user32 is not None:
            self._foreground_hwnd = int(self._user32.GetForegroundWindow())
            self._focus_hwnd = _resolve_focus_hwnd(self._foreground_hwnd)
        self._start_record()
        self.hotkey_started_record = self.audio_source is not None
        if self.hotkey_started_record:
            self._stream_partial_text = ""
            self._stream_timer.start()
            self.prompt_bar.show_prompt("正在录音...", "")
            self._append_log(f"[HOTKEY] press -> start recording ({self.config.shortcuts.hotkey})")

    def _on_hotkey_released(self) -> None:
        if not self.hotkey_started_record:
            return
        self.hotkey_started_record = False
        if self.audio_source is None:
            return
        self._append_log(f"[HOTKEY] release -> stop and transcribe ({self.config.shortcuts.hotkey})")
        self._stop_record(run_asr=True, insert_after_asr=True)

    def _do_stream_snapshot(self) -> None:
        """Take a snapshot of the current audio buffer and run partial ASR."""
        if self.audio_source is None or self.audio_buffer is None:
            return
        if self.record_format is None:
            return
        # Don't start a new stream inference if one is already running
        if self._stream_thread is not None and self._stream_thread.isRunning():
            return

        raw_data = bytes(self.audio_buffer.data())
        if not raw_data or len(raw_data) < 3200:  # at least 0.1s of 16kHz
            return

        fmt = self.record_format
        try:
            snap_path = Path(tempfile.gettempdir()) / f"_asr_stream_{uuid.uuid4().hex[:8]}.wav"
            with wave.open(str(snap_path), "wb") as wf:
                wf.setnchannels(fmt.channelCount())
                wf.setsampwidth(fmt.bytesPerSample())
                wf.setframerate(fmt.sampleRate())
                wf.writeframes(raw_data)
        except Exception:  # noqa: BLE001
            return

        cfg = build_runtime_config(self.config, self.asr_device)
        language = self._combo_value(self.run_language_combo)

        worker = _StreamPartialWorker(str(snap_path), cfg, language)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._on_stream_result)
        worker.result.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_stream_thread())

        self._stream_thread = thread
        thread.start()

    def _on_stream_result(self, text: str) -> None:
        """Handle partial streaming recognition result."""
        if text:
            self._stream_partial_text = text
            self.prompt_bar.update_prompt(
                status="正在录音...",
                text=text,
            )
            self.current_partial_view.setPlainText(text)

    def _cleanup_stream_thread(self) -> None:
        self._stream_thread = None

    def _refresh_microphones(self) -> None:
        wanted = self.config.microphone.device
        self.micro_device_combo.blockSignals(True)
        self.micro_device_combo.clear()
        self.micro_device_combo.addItem("自动(跟随系统)", "auto")
        for dev in QMediaDevices.audioInputs():
            desc = dev.description().strip()
            if desc:
                self.micro_device_combo.addItem(desc, desc)
        self._set_combo_value(self.micro_device_combo, wanted)
        self.micro_device_combo.blockSignals(False)

    def _refresh_backend_warning(self) -> None:
        backend = self._combo_value(self.model_backend_combo)
        if backend == "vllm" and sys.platform.startswith("win"):
            self.model_warning.setText("vLLM 不支持原生 Windows，请使用 transformers 或迁移到 WSL2/Linux。")
        else:
            self.model_warning.setText("")

    def _hint_from_system_language(self, language_code: str) -> str | None:
        if language_code.lower().startswith("zh"):
            return "Chinese"
        if language_code.lower().startswith("en"):
            return "English"
        return None

    def _current_audio_device(self):
        selected = self.micro_device_combo.currentData()
        if selected in (None, "auto"):
            return QMediaDevices.defaultAudioInput()
        for dev in QMediaDevices.audioInputs():
            if dev.description().strip() == str(selected):
                return dev
        return QMediaDevices.defaultAudioInput()

    def _build_record_format(self, device) -> QAudioFormat:
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isFormatSupported(fmt):
            return fmt

        alt = QAudioFormat()
        alt.setSampleRate(48000)
        alt.setChannelCount(1)
        alt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isFormatSupported(alt):
            return alt

        preferred = device.preferredFormat()
        if preferred.sampleFormat() == QAudioFormat.SampleFormat.Int16:
            return preferred

        raise RuntimeError("麦克风不支持 Int16 PCM 录音，当前版本无法处理该设备。")

    def _start_record(self) -> None:
        if self.model_loading:
            QMessageBox.information(self, "模型加载中", "模型正在加载，请稍候再开始录音。")
            return
        if self.asr_busy:
            QMessageBox.warning(self, "忙碌中", "当前有识别任务在运行，请稍后。")
            return
        if self.audio_source is not None:
            return
        self._save_all_settings(silent=True)
        device = self._current_audio_device()
        try:
            fmt = self._build_record_format(device)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "录音失败", str(exc))
            return

        buffer = QBuffer(self)
        if not buffer.open(QIODevice.OpenModeFlag.ReadWrite):
            QMessageBox.critical(self, "录音失败", "无法创建录音缓冲区。")
            return

        source = QAudioSource(device, fmt, self)
        source.start(buffer)

        self.audio_source = source
        self.audio_buffer = buffer
        self.record_format = fmt
        self.recording_start_t = time.perf_counter()
        self.record_timer.start()

        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.label_record_state.setText(f"状态: 录音中 ({device.description()})")
        self._append_log(
            f"[RECORD] start device={device.description()} "
            f"sr={fmt.sampleRate()} ch={fmt.channelCount()}"
        )

    def _stop_record_and_transcribe(self) -> None:
        self._stop_record(run_asr=True, insert_after_asr=False)

    def _stop_record(self, run_asr: bool, insert_after_asr: bool = False) -> None:
        self._stream_timer.stop()
        if self.audio_source is None:
            self.hotkey_started_record = False
            if not run_asr:
                self.prompt_bar.hide()
            return
        self.record_timer.stop()

        assert self.audio_buffer is not None
        assert self.record_format is not None

        self.audio_source.stop()
        raw_data = bytes(self.audio_buffer.data())
        self.audio_buffer.close()

        self.audio_source.deleteLater()
        self.audio_source = None
        self.audio_buffer.deleteLater()
        self.audio_buffer = None
        fmt = self.record_format
        self.record_format = None
        self.hotkey_started_record = False
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.label_record_state.setText("状态: 空闲")
        self.label_recording_time.setText("录音时长: 00:00:00.000")
        if run_asr:
            self.prompt_bar.show_prompt("正在识别...", "")
        else:
            self.prompt_bar.hide()

        if not raw_data:
            self._append_log("[WARN] 本次录音无数据。")
            self.prompt_bar.show_prompt("没有录到有效音频", "")
            self.prompt_bar.hide_later(1200)
            return

        target = self._next_recording_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(target), "wb") as wav_file:
                wav_file.setnchannels(fmt.channelCount())
                wav_file.setsampwidth(fmt.bytesPerSample())
                wav_file.setframerate(fmt.sampleRate())
                wav_file.writeframes(raw_data)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存录音失败", str(exc))
            return

        audio_duration = self._calc_pcm_duration(raw_data, fmt)
        self.label_audio_duration.setText(f"音频时长: {format_hms(audio_duration)}")
        self._append_log(
            f"[RECORD] saved={target} bytes={len(raw_data)} duration={format_hms(audio_duration)}"
        )

        if run_asr:
            self._run_asr_task([str(target.resolve())], insert_after_asr=insert_after_asr)

    def _calc_pcm_duration(self, raw_data: bytes, fmt: QAudioFormat) -> float:
        sample_width = fmt.bytesPerSample()
        if sample_width <= 0:
            sample_width = 2
        denom = fmt.sampleRate() * fmt.channelCount() * sample_width
        if denom <= 0:
            return 0.0
        return len(raw_data) / float(denom)

    def _next_recording_file(self) -> Path:
        rec_dir = normalize_cli_path(self.micro_record_dir_edit.text().strip())
        if not rec_dir:
            rec_dir = "recordings"
        rec_dir_path = Path(rec_dir).expanduser()
        if not rec_dir_path.is_absolute():
            rec_dir_path = (Path.cwd() / rec_dir_path).resolve()
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
        return rec_dir_path / name

    def _run_patterns(self) -> None:
        tokens = parse_input_tokens(self.input_patterns.text().strip())
        if not tokens:
            QMessageBox.information(self, "提示", "请输入音频/视频路径或通配符。")
            return
        self._run_asr_task(tokens)

    def _pick_files_and_transcribe(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音频或视频文件",
            str(Path.cwd()),
            "Media (*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)",
        )
        if not files:
            return
        self._run_asr_task(files)

    def _run_asr_task(self, inputs: list[str], insert_after_asr: bool = False) -> None:
        if self.model_loading:
            QMessageBox.information(self, "模型加载中", "模型正在加载，请稍后再发起识别。")
            self.prompt_bar.show_prompt("模型加载中", "请稍后再试")
            self.prompt_bar.hide_later(1200)
            return
        if self.asr_busy:
            QMessageBox.warning(self, "忙碌中", "已有识别任务在运行，请稍后。")
            self.prompt_bar.show_prompt("识别任务繁忙", "当前已有任务在运行")
            self.prompt_bar.hide_later(1200)
            return
        self._save_all_settings(silent=True)

        return_ts = self.chk_return_ts.isChecked() or self.chk_make_subtitle.isChecked()
        subtitle_dir = None
        if self.chk_make_subtitle.isChecked():
            subtitle_dir_raw = normalize_cli_path(self.subtitle_dir_edit.text().strip())
            if subtitle_dir_raw:
                subtitle_dir = subtitle_dir_raw
            else:
                subtitle_dir = str((self._next_recording_file().parent / "subtitles").resolve())

        cfg = build_runtime_config(self.config, self.asr_device)
        language = self._combo_value(self.run_language_combo)

        self.insert_after_asr = bool(insert_after_asr)
        self.asr_busy = True
        self.asr_cancel_requested = False
        self._set_controls_busy(True)
        self.label_record_state.setText("状态: 识别中")
        self._set_asr_progress(0, "进度: 准备识别", "")
        self.prompt_bar.show_prompt("正在识别...", "")

        worker = AsrTaskWorker(
            inputs=[normalize_cli_path(x) for x in inputs],
            cfg=cfg,
            language=language,
            return_time_stamps=return_ts,
            subtitle_dir=subtitle_dir,
            ffmpeg_bin=self.ffmpeg_bin,
            ffprobe_bin=self.ffprobe_bin,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.progress.connect(self._on_asr_progress)
        worker.completed.connect(self._on_asr_completed)
        worker.failed.connect(self._on_asr_failed)
        worker.cancelled.connect(self._on_asr_cancelled)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_asr_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self.asr_worker = worker
        self.asr_thread = thread
        thread.start()

    def _enhance_result_text(self, raw_text: str) -> tuple[str, list[str]]:
        return enhance_recognized_text(raw_text, self.config.skills)

    def _on_asr_progress(self, result: dict[str, Any]) -> None:
        event = result.get("event", "file_done")
        idx = result.get("index")
        total = result.get("total")
        source_media = result.get("source_media")
        if event == "file_start":
            percent = int(result.get("progress_percent", 0) or 0)
            self._set_asr_progress(
                percent,
                f"进度: [{idx}/{total}] 准备识别 {Path(str(source_media)).name}",
                "",
            )
            self.prompt_bar.update_prompt(
                status=f"正在识别 {Path(str(source_media)).name}",
                text="",
            )
            return

        if event == "file_partial":
            percent = result.get("progress_percent")
            file_percent = result.get("file_progress_percent")
            model_time = result.get("model_inference_time_s")
            current_text = str(result.get("current_text", "") or "")
            self._set_asr_progress(
                int(percent) if percent is not None else None,
                (
                    f"进度: [{idx}/{total}] 当前文件 {int(file_percent or 0)}% "
                    f"| 模型已耗时 {format_hms(model_time)}"
                ),
                current_text,
            )
            self.prompt_bar.update_prompt(
                status=f"正在识别 [{idx}/{total}] {int(file_percent or 0)}%",
                text=current_text,
            )
            return

        language = result.get("language")
        audio_duration_s = result.get("audio_duration_s")
        model_time = result.get("model_inference_time_s")
        raw_text = str(result.get("text", "") or "")
        enhanced_text, enhance_ops = self._enhance_result_text(raw_text)
        text = enhanced_text or raw_text
        subtitle = result.get("subtitle_path")
        segments = result.get("segments", [])

        lines = [
            f"[{idx}/{total}] audio={source_media}",
            f"language={language}",
            f"audio_duration={format_hms(audio_duration_s)}",
            f"recognition_time={format_hms(model_time)}",
        ]
        for i, seg in enumerate(segments):
            lines.append(
                f"segment[{i}] start={format_hms(seg.get('start_time'))} "
                f"end={format_hms(seg.get('end_time'))} text={seg.get('text', '')}"
            )
        if subtitle:
            lines.append(f"subtitle={subtitle}")
        if raw_text.strip() and raw_text.strip() != text.strip():
            lines.append(f"raw_text={raw_text}")
            if enhance_ops:
                lines.append(f"enhance_ops={','.join(enhance_ops)}")
        lines.append(f"text={text}")
        self._append_log("\n".join(lines))
        self._set_asr_progress(
            int(result.get("progress_percent", 0) or 0),
            f"进度: [{idx}/{total}] 已完成 {Path(str(source_media)).name}",
            str(text or ""),
        )
        self.prompt_bar.update_prompt(
            status=f"已完成 [{idx}/{total}] {Path(str(source_media)).name}",
            text=str(text or ""),
        )

        self.label_audio_duration.setText(f"音频时长: {format_hms(audio_duration_s)}")
        self.label_model_time.setText(f"识别耗时: {format_hms(model_time)}")

        entry = {
            "created_at": now_text(),
            "source_media": source_media,
            "audio_path": result.get("audio_path"),
            "language": language,
            "audio_duration_s": audio_duration_s,
            "model_inference_time_s": model_time,
            "raw_text": raw_text,
            "enhanced_text": text,
            "enhance_ops": enhance_ops,
            "text": text,
            "subtitle_path": subtitle,
            "segments": segments,
        }
        self.history_entries.append(entry)
        self.history_store.save(self.history_entries)
        self._render_history()

    def _on_asr_completed(self, results: list[dict[str, Any]]) -> None:
        self._append_log(f"[DONE] 本次共完成 {len(results)} 个文件。")
        self._set_asr_progress(100, "进度: 全部完成", None)
        combined_parts: list[str] = []
        combined_raw_parts: list[str] = []
        all_ops: set[str] = set()
        changed = False
        for item in results:
            raw_piece = str(item.get("text", "") or "").strip()
            if not raw_piece:
                continue
            enhanced_piece, ops = self._enhance_result_text(raw_piece)
            final_piece = (enhanced_piece or raw_piece).strip()
            combined_raw_parts.append(raw_piece)
            combined_parts.append(final_piece)
            if final_piece != raw_piece:
                changed = True
            all_ops.update(ops)

        combined_text = "\n".join(combined_parts)
        combined_raw_text = "\n".join(combined_raw_parts)
        if changed:
            self._append_log(
                f"[ENHANCE] 已应用文本增强。ops={','.join(sorted(all_ops)) or 'n/a'}"
            )

        inserted = False
        if self.insert_after_asr and combined_text:
            inserted = self._insert_text_to_foreground(combined_text)
        if inserted:
            self.prompt_bar.show_prompt("已写入当前光标位置", combined_text)
        else:
            self.prompt_bar.show_prompt("识别完成", combined_text)
        if changed and combined_raw_text:
            self._append_log(f"[RAW] {combined_raw_text}")
            self._append_log(f"[ENHANCED] {combined_text}")
        self.prompt_bar.hide_later(1500)

    def _on_asr_cancelled(self, message: str) -> None:
        self._append_log(f"[CANCELLED] {message}")
        self._set_asr_progress(0, "进度: 已取消", None)
        self.label_record_state.setText("状态: 已取消")
        self.prompt_bar.show_prompt("识别已取消", "")
        self.prompt_bar.hide_later(1000)

    def _on_asr_failed(self, detail: str) -> None:
        self._append_log(f"[ERROR] {detail}")
        self._set_asr_progress(0, "进度: 失败", None)
        self.prompt_bar.show_prompt("识别失败", detail.splitlines()[0] if detail else "")
        self.prompt_bar.hide_later(1800)
        QMessageBox.critical(self, "识别失败", detail.splitlines()[0] if detail else "Unknown error")

    def _on_asr_thread_finished(self) -> None:
        self.asr_busy = False
        self.insert_after_asr = False
        self.asr_worker = None
        self.asr_thread = None
        self._set_controls_busy(False)
        if self.asr_cancel_requested:
            self.asr_cancel_requested = False
            AsrEngineCache.clear()
        self.label_record_state.setText("状态: 空闲")

    def _set_controls_busy(self, busy: bool) -> None:
        effective_busy = bool(busy or self.model_loading)
        self.btn_pick_files.setEnabled(not effective_busy)
        self.btn_run_patterns.setEnabled(not effective_busy)
        self.btn_start_record.setEnabled((not effective_busy) and self.audio_source is None)
        self.btn_stop_record.setEnabled((not effective_busy) and self.audio_source is not None)
        self.btn_cancel_asr.setEnabled(busy)

    def _set_asr_progress(
        self,
        percent: int | None,
        status: str | None,
        current_text: str | None,
    ) -> None:
        if percent is None:
            self.asr_progress_bar.setRange(0, 0)
        else:
            self.asr_progress_bar.setRange(0, 100)
            self.asr_progress_bar.setValue(max(0, min(100, int(percent))))
        if status is not None:
            self.label_asr_progress.setText(status)
        if current_text is not None:
            self.current_partial_view.setPlainText(current_text)

    def _cancel_asr_clicked(self) -> None:
        self._cancel_running_task(force=False)

    def _cancel_running_task(self, force: bool) -> None:
        if not self.asr_busy:
            return
        self.asr_cancel_requested = True
        self.label_record_state.setText("状态: 正在取消")
        self._set_asr_progress(None, "进度: 正在取消...", None)
        self._append_log("[INFO] 已请求取消当前识别任务。")
        if self.asr_worker is not None:
            self.asr_worker.cancel()
        if force and self.asr_thread is not None and self.asr_thread.isRunning():
            if not self.asr_thread.wait(1200):
                self._append_log("[WARN] 识别线程未及时结束，执行强制终止。")
                self.asr_thread.terminate()
                self.asr_thread.wait(800)

    def _insert_text_to_foreground(self, text: str) -> bool:
        payload = str(text or "")
        if not payload:
            return False
        preferred_hwnd = self._focus_hwnd or self._foreground_hwnd
        current_fg = int(_WIN_USER32.GetForegroundWindow() or 0) if sys.platform.startswith("win") else 0
        ok, detail = send_text_without_clipboard(payload, preferred_hwnd=preferred_hwnd)
        if ok:
            self._append_log(
                f"[INPUT] 已写入当前光标位置（未使用剪贴板）。method={detail} "
                f"target={preferred_hwnd} foreground={current_fg}"
            )
        else:
            self._append_log(
                f"[WARN] 文本注入失败，未能写入当前光标位置。detail={detail} "
                f"target={preferred_hwnd} foreground={current_fg}"
            )
        self._foreground_hwnd = None
        self._focus_hwnd = None
        return ok

    def _append_log(self, text: str) -> None:
        self.output_view.appendPlainText(text.rstrip() + "\n")
        self.output_view.verticalScrollBar().setValue(
            self.output_view.verticalScrollBar().maximum()
        )
        if hasattr(self, "btn_toggle_log") and hasattr(self, "log_panel_expanded"):
            if not self.log_panel_expanded:
                self.btn_toggle_log.setText("展开日志 •")

    def _toggle_log_panel(self) -> None:
        expanded = not bool(getattr(self, "log_panel_expanded", False))
        self._set_log_panel_expanded(expanded)

    def _set_log_panel_expanded(self, expanded: bool) -> None:
        self.log_panel_expanded = bool(expanded)
        if hasattr(self, "output_view"):
            self.output_view.setVisible(self.log_panel_expanded)
        if hasattr(self, "btn_toggle_log"):
            self.btn_toggle_log.setText("收起日志" if self.log_panel_expanded else "展开日志")

    def _clear_output(self) -> None:
        self.output_view.clear()

    def _render_history(self) -> None:
        self.history_list.clear()
        if not self.history_entries:
            return
        for item in reversed(self.history_entries[-300:]):
            timestamp = item.get("created_at", "--")
            source = Path(str(item.get("source_media", ""))).name
            text = compact_text(str(item.get("text", "")))
            title = f"{timestamp}  {source}"
            row_text = f"{title}\n{text}"
            list_item = QListWidgetItem(row_text)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.history_list.addItem(list_item)

    def _open_history_item(self, item: QListWidgetItem) -> None:
        payload = item.data(Qt.ItemDataRole.UserRole) or {}
        lines = [
            f"[HISTORY] time={payload.get('created_at')}",
            f"audio={payload.get('source_media')}",
            f"language={payload.get('language')}",
            f"audio_duration={format_hms(payload.get('audio_duration_s'))}",
            f"recognition_time={format_hms(payload.get('model_inference_time_s'))}",
        ]
        subtitle = payload.get("subtitle_path")
        if subtitle:
            lines.append(f"subtitle={subtitle}")
        segments = payload.get("segments") or []
        for i, seg in enumerate(segments):
            lines.append(
                f"segment[{i}] start={format_hms(seg.get('start_time'))} "
                f"end={format_hms(seg.get('end_time'))} text={seg.get('text', '')}"
            )
        raw_text = str(payload.get("raw_text", "") or "")
        enhanced_text = str(payload.get("enhanced_text", payload.get("text", "")) or "")
        ops = payload.get("enhance_ops") or []
        if raw_text and raw_text != enhanced_text:
            lines.append(f"raw_text={raw_text}")
            if isinstance(ops, list) and ops:
                lines.append(f"enhance_ops={','.join(str(x) for x in ops)}")
        lines.append(f"text={enhanced_text}")
        self._append_log("\n".join(lines))

    def _clear_history(self) -> None:
        reply = QMessageBox.question(
            self,
            "确认",
            "确认清空历史记录吗？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.history_entries.clear()
        self.history_store.clear()
        self._render_history()

    def _tick_record_time(self) -> None:
        if self.recording_start_t is None:
            return
        elapsed = time.perf_counter() - self.recording_start_t
        self.label_recording_time.setText(f"录音时长: {format_hms(elapsed)}")

    def _pick_recording_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择录音目录",
            self.micro_record_dir_edit.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.micro_record_dir_edit.setText(selected)

    def _open_recording_dir(self) -> None:
        path = self._next_recording_file().parent
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _pick_subtitle_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择字幕目录",
            self.subtitle_dir_edit.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.subtitle_dir_edit.setText(selected)

    def _pick_model_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择模型目录",
            self.model_dir_edit.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.model_dir_edit.setText(selected)

    def _release_model_cache(self) -> None:
        AsrEngineCache.clear()
        self._append_log("[INFO] 模型缓存已释放。")

    def _on_nav_changed(self, row: int) -> None:
        if row < 0:
            return
        self.pages.setCurrentIndex(row)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.asr_busy:
            reply = QMessageBox.question(
                self,
                "任务进行中",
                "识别任务仍在运行，是否立即中断并退出？",
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel_running_task(force=True)
        if self.model_loading and self.model_load_thread is not None and self.model_load_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "模型加载中",
                "模型仍在预加载，是否立即退出？",
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if not self.model_load_thread.wait(800):
                self.model_load_thread.terminate()
                self.model_load_thread.wait(600)
        if self.audio_source is not None:
            reply = QMessageBox.question(
                self,
                "提示",
                "当前仍在录音。关闭将停止录音但不自动识别，是否继续？",
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_record(run_asr=False)
        self.hotkey_poll_timer.stop()
        self.prompt_bar.hide()
        self._save_all_settings(silent=True)
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR Desktop GUI")
    parser.add_argument("--config", default="desktop_service_config.json")
    parser.add_argument("--history", default=None)
    parser.add_argument("--device", default="auto", help='ASR runtime device: auto/cpu/cuda:0')
    parser.add_argument(
        "--ffmpeg-bin",
        default=default_binary_path("ffmpeg.exe", "ffmpeg"),
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=default_binary_path("ffprobe.exe", "ffprobe"),
    )
    parser.add_argument("--model-size", choices=["0.6b", "1.7b"], default=None)
    parser.add_argument("--model-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(APP_STYLE)

    win = DesktopVoiceInputWindow(
        config_path=normalize_cli_path(args.config),
        history_path=normalize_cli_path(args.history) if args.history else None,
        asr_device=normalize_cli_path(args.device),
        ffmpeg_bin=normalize_cli_path(args.ffmpeg_bin),
        ffprobe_bin=normalize_cli_path(args.ffprobe_bin),
        cli_model_size=args.model_size,
        cli_model_dir=normalize_cli_path(args.model_dir) if args.model_dir else None,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
