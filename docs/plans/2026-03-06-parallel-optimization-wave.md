# Parallel Optimization Wave Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve GUI, REST API, and CLI/Shell in parallel with low-conflict task boundaries and a single integration wave.

**Architecture:** Split the work into three independent task domains with disjoint write scopes. Keep shared-core files out of Wave 1 so agents can implement and verify in parallel. Integrate only after each subsystem change is individually validated.

**Tech Stack:** Python 3.10+, stdlib `unittest`, PySide6 GUI code, FastAPI service, existing CLI/Shell scripts.

---

### Task 1: GUI worker hardening

**Files:**
- Modify: `desktop_gui_app.py:1245`
- Create: `desktop_gui_runtime_utils.py`
- Test: `tests/test_desktop_gui_runtime_utils.py`

**Step 1: Write the failing test**

Add tests for extracted GUI runtime helpers, for example:

- formatting worker failures into stable user-facing messages
- deciding whether a file should use chunked transcription
- safe handling of partial-stream inference failures without silent data loss

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_desktop_gui_runtime_utils -v`
Expected: FAIL because the helper module does not exist yet.

**Step 3: Write minimal implementation**

Create `desktop_gui_runtime_utils.py` with pure helpers and refactor `desktop_gui_app.py` to call them. Keep all Qt signal wiring in `desktop_gui_app.py`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_desktop_gui_runtime_utils -v`
Expected: PASS.

**Step 5: Record result**

List changed files and verification output. Do not commit unless the user explicitly asks.

### Task 2: REST API error handling and config safety

**Files:**
- Modify: `desktop_service/api_server.py:1`
- Modify: `desktop_service/session_manager.py:1`
- Modify: `desktop_service/config_store.py:1`
- Test: `tests/test_desktop_service_api.py`

**Step 1: Write the failing test**

Add tests covering:

- invalid base64 chunk payload returns clear client error
- invalid session state transitions raise stable exceptions
- broken config file is preserved or reported instead of silently disappearing

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_desktop_service_api -v`
Expected: FAIL because current behavior is too permissive or silent.

**Step 3: Write minimal implementation**

Refine `api_server.py`, `session_manager.py`, and `config_store.py` to make error semantics explicit and safer.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_desktop_service_api -v`
Expected: PASS.

**Step 5: Record result**

List changed files and verification output. Do not commit unless the user explicitly asks.

### Task 3: CLI / Shell consistency pass

**Files:**
- Modify: `asr_cli.py:27`
- Modify: `asr_service_shell.py:28`
- Create: `asr_frontend_common.py`
- Test: `tests/test_asr_frontend_common.py`

**Step 1: Write the failing test**

Add tests for shared frontend helpers, for example:

- FFmpeg / FFprobe binary path resolution
- batch size / numeric argument validation
- stable message formatting shared by CLI and Shell

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asr_frontend_common -v`
Expected: FAIL because the shared module does not exist yet.

**Step 3: Write minimal implementation**

Create `asr_frontend_common.py`, move duplicated logic from CLI / Shell into it, and normalize the user-facing behavior across both entry points.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_asr_frontend_common -v`
Expected: PASS.

**Step 5: Record result**

List changed files and verification output. Do not commit unless the user explicitly asks.

### Task 4: Main integration and verification

**Files:**
- Modify: `desktop_gui_app.py:1245`
- Modify: `desktop_service/api_server.py:1`
- Modify: `desktop_service/session_manager.py:1`
- Modify: `desktop_service/config_store.py:1`
- Modify: `asr_cli.py:27`
- Modify: `asr_service_shell.py:28`
- Modify: `tests/test_desktop_gui_runtime_utils.py:1`
- Modify: `tests/test_desktop_service_api.py:1`
- Modify: `tests/test_asr_frontend_common.py:1`

**Step 1: Review subsystem diffs**

Check that the three task outputs do not overlap on shared-core files and that naming / error semantics are consistent.

**Step 2: Run focused verification**

Run:

- `python -m unittest tests.test_desktop_gui_runtime_utils -v`
- `python -m unittest tests.test_desktop_service_api -v`
- `python -m unittest tests.test_asr_frontend_common -v`
- `python -m py_compile desktop_gui_app.py desktop_service/api_server.py desktop_service/session_manager.py desktop_service/config_store.py asr_cli.py asr_service_shell.py`

Expected: all targeted tests PASS and `py_compile` exits successfully.

**Step 3: Fix integration gaps**

Apply only the minimal glue changes needed to make the three waves coexist cleanly.

**Step 4: Re-run verification**

Repeat the same commands until they pass cleanly.

**Step 5: Record result**

Summarize merged changes, remaining risks, and any suggested follow-up. Do not commit unless the user explicitly asks.
