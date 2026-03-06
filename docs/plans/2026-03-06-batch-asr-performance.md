# Batch ASR Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add low-risk batch transcription and audio-extraction reuse for CLI and resident shell without changing timestamp fallback behavior.

**Architecture:** Introduce shared media-preparation and batch-transcription helpers in `asr_media_utils.py`, then switch `asr_cli.py` and `asr_service_shell.py` to use batching only for non-timestamp flows. Keep timestamp and subtitle flows on the existing per-file fallback path.

**Tech Stack:** Python 3.10+, stdlib `unittest`, existing `qwen-asr`, `torch`, FFmpeg integration.

---

### Task 1: Add tests for shared batching helpers

**Files:**
- Create: `tests/test_asr_media_utils.py`
- Modify: `asr_media_utils.py`

**Step 1: Write the failing test**

Add tests for:

- a helper that chunks prepared audio rows and preserves input/output ordering
- a helper that raises when model output count does not match input count
- cached video extraction reuse when output WAV is newer than source

Example skeleton:

```python
import unittest

from asr_media_utils import batched_transcribe_rows


class BatchTranscribeRowsTests(unittest.TestCase):
    def test_preserves_result_order_across_batches(self):
        ...
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: FAIL because helper functions do not exist yet.

**Step 3: Write minimal implementation**

In `asr_media_utils.py`, add focused helpers for:

- batching prepared rows
- validating output length
- optional FFmpeg extraction cache reuse

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_asr_media_utils.py asr_media_utils.py
git commit -m "test: cover shared batch transcription helpers"
```

### Task 2: Switch CLI non-timestamp flow to batched inference

**Files:**
- Modify: `asr_cli.py`
- Modify: `asr_media_utils.py`
- Test: `tests/test_asr_media_utils.py`

**Step 1: Write the failing test**

Add one behavior test proving the shared batching helper is suitable for CLI-style prepared rows without timestamps.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: FAIL with missing behavior or wrong ordering.

**Step 3: Write minimal implementation**

Update `asr_cli.py` to:

- prepare all media rows first
- if timestamps are not needed, call the batch helper
- if timestamps are needed, keep current per-file fallback path

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add asr_cli.py asr_media_utils.py tests/test_asr_media_utils.py
git commit -m "feat: batch CLI transcription for non-timestamp flow"
```

### Task 3: Switch resident shell non-timestamp flow to batched inference

**Files:**
- Modify: `asr_service_shell.py`
- Modify: `asr_media_utils.py`
- Test: `tests/test_asr_media_utils.py`

**Step 1: Write the failing test**

Add a helper-level behavior test covering shell-style multi-file reuse of the shared batching path.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: FAIL until shell-facing assumptions are encoded in the helper output mapping.

**Step 3: Write minimal implementation**

Update `asr_service_shell.py` to:

- prepare all media rows first
- batch transcribe when timestamps are not requested
- retain the current timestamp fallback path when timestamps are requested

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add asr_service_shell.py asr_media_utils.py tests/test_asr_media_utils.py
git commit -m "feat: batch resident shell transcription"
```

### Task 4: Verify behavior and guard regressions

**Files:**
- Modify: `tests/test_asr_media_utils.py`
- Modify: `asr_media_utils.py`
- Modify: `asr_cli.py`
- Modify: `asr_service_shell.py`

**Step 1: Write the failing test**

Add regression tests for:

- timestamp-request path staying non-batched
- mismatched batch output raising a clear exception
- extraction cache invalidation when source file is newer

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: FAIL until all guards are in place.

**Step 3: Write minimal implementation**

Tighten helper validation and keep CLI / Shell behavior aligned with the design.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_asr_media_utils -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_asr_media_utils.py asr_media_utils.py asr_cli.py asr_service_shell.py
git commit -m "test: add regressions for batch ASR performance path"
```
