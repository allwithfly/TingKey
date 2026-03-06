# Model Load Speed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce model cold-start cost across GUI, CLI/Shell, and first-inference scenarios through staged loading, lazy aligner initialization, warmup, and CLI runtime reuse.

**Architecture:** Introduce a shared runtime registry and staged load pipeline inside `speech_output.py`, then integrate preload/warmup behavior into GUI and service entry points. Extend the local service path so CLI can prefer a warm runtime and fall back to local loading when unavailable.

**Tech Stack:** Python 3.10+, stdlib `unittest`, existing `qwen-asr`, PySide6 GUI, FastAPI service, current CLI/Shell entry points.

---

### Task 1: Add runtime-registry tests

**Files:**
- Create: `tests/test_speech_output_runtime.py`
- Modify: `speech_output.py`

**Step 1: Write the failing test**

Add tests for:
- stable runtime key generation from model config
- same-config requests reuse one runtime in-process
- different configs do not share runtime

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_speech_output_runtime -v`
Expected: FAIL because the runtime registry API does not exist yet.

**Step 3: Write minimal implementation**

Add a small runtime registry in `speech_output.py` keyed by normalized load config. Keep public behavior backward compatible.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_speech_output_runtime -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add speech_output.py tests/test_speech_output_runtime.py
git commit -m "test: cover speech output runtime registry"
```

### Task 2: Add lazy aligner and warmup

**Files:**
- Modify: `speech_output.py`
- Modify: `tests/test_speech_output_runtime.py`

**Step 1: Write the failing test**

Add tests for:
- aligner is not loaded for normal transcription
- aligner load is triggered only when timestamps are requested
- warmup failure does not break normal inference availability

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_speech_output_runtime -v`
Expected: FAIL until lazy aligner and warmup hooks exist.

**Step 3: Write minimal implementation**

Implement staged runtime methods in `speech_output.py`:
- core load
- lazy aligner ensure
- optional warmup

Keep current fallback semantics for timestamps.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_speech_output_runtime -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add speech_output.py tests/test_speech_output_runtime.py
git commit -m "feat: add lazy aligner and warmup pipeline"
```

### Task 3: Integrate staged loading into GUI and service

**Files:**
- Modify: `desktop_gui_app.py:1141`
- Modify: `desktop_service/final_asr.py:17`
- Modify: `desktop_service/api_server.py:1`
- Create: `tests/test_model_load_integration.py`

**Step 1: Write the failing test**

Add tests for:
- GUI/service integration calls warmup without blocking core ready state
- final ASR runner reuses staged runtime correctly
- runtime-related API fallback behavior stays safe

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_model_load_integration -v`
Expected: FAIL because staged integration hooks are missing.

**Step 3: Write minimal implementation**

Update GUI and service integration to:
- mark model core ready once registry/core load completes
- schedule warmup separately
- reuse the same runtime abstraction

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_model_load_integration -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add desktop_gui_app.py desktop_service/final_asr.py desktop_service/api_server.py tests/test_model_load_integration.py
 git commit -m "feat: integrate staged model loading into gui and service"
```

### Task 4: Add CLI runtime reuse with fallback

**Files:**
- Modify: `asr_cli.py:174`
- Modify: `asr_service_shell.py:225`
- Modify: `desktop_service/api_server.py:1`
- Create: `tests/test_cli_runtime_reuse.py`

**Step 1: Write the failing test**

Add tests for:
- CLI prefers local runtime when reachable
- CLI falls back to local loading when runtime is unavailable
- shell behavior remains unchanged unless runtime reuse is explicitly enabled

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_cli_runtime_reuse -v`
Expected: FAIL because runtime reuse path does not exist yet.

**Step 3: Write minimal implementation**

Extend the local service API with a direct transcription path suitable for runtime reuse, then update CLI to try it first and safely fall back to current behavior.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_cli_runtime_reuse -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add asr_cli.py asr_service_shell.py desktop_service/api_server.py tests/test_cli_runtime_reuse.py
 git commit -m "feat: reuse local runtime for cli cold starts"
```

### Task 5: Final timing and regression verification

**Files:**
- Modify: `tests/test_speech_output_runtime.py`
- Modify: `tests/test_model_load_integration.py`
- Modify: `tests/test_cli_runtime_reuse.py`
- Modify: `speech_output.py`
- Modify: `desktop_gui_app.py`
- Modify: `desktop_service/api_server.py`
- Modify: `asr_cli.py`

**Step 1: Add final regression tests**

Cover:
- timestamp fallback still works
- runtime reuse does not change output shape
- warmup failures do not block usable transcribe path

**Step 2: Run focused verification**

Run:
- `python -m unittest tests.test_speech_output_runtime -v`
- `python -m unittest tests.test_model_load_integration -v`
- `python -m unittest tests.test_cli_runtime_reuse -v`
- `python -m py_compile speech_output.py desktop_gui_app.py desktop_service/final_asr.py desktop_service/api_server.py asr_cli.py asr_service_shell.py tests/test_speech_output_runtime.py tests/test_model_load_integration.py tests/test_cli_runtime_reuse.py`

Expected: all PASS.

**Step 3: Run timing spot checks**

Run project-specific timing checks to compare:
- GUI model ready elapsed
- CLI first-call elapsed with runtime available vs unavailable
- first real transcription after preload

**Step 4: Fix any integration gaps**

Apply only minimal glue fixes required to keep all entry points aligned.

**Step 5: Commit**

```bash
git add speech_output.py desktop_gui_app.py desktop_service/final_asr.py desktop_service/api_server.py asr_cli.py asr_service_shell.py tests/test_speech_output_runtime.py tests/test_model_load_integration.py tests/test_cli_runtime_reuse.py
 git commit -m "perf: reduce model load latency across entry points"
```
