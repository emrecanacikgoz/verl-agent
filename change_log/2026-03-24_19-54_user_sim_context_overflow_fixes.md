# User Simulator Context Overflow Fixes — 2026-03-24

## Overview

Fixed user simulator API calls consistently hitting the vLLM server's 32768-token context
limit during solver training. Every `[UserSim]` API call past mid-episode was failing on
first attempt, wasting one full vLLM inference call, then retrying with trimmed history.
This doubled user sim latency for all long episodes across all parallel environments.

3 files modified, 3 bugs fixed. No config or interface changes.

---

## Files Modified

| # | File | What Changed |
|---|------|-------------|
| 1 | `agent_system/environments/env_package/tau2bench/envs.py` | Strip `<think>` blocks before sending agent output to user sim |
| 2 | `agent_system/environments/env_package/tau2bench/user_sim.py` | Persist trimmed history, add retry cap, add proactive budget trim |
| 3 | `algo.md` | Updated data flow spec and critical invariants checklist |

---

## Root Cause Analysis

The user simulator model is identical to tau2-bench (same prompt, same guidelines). The
user sim itself never generates `<think>` tokens. The problem was what the **agent's raw
output** put into the user sim's conversation history.

### How the agent's `<think>` blocks ended up in user sim context

The solver's `max_response_length=2048`. When the agent output lacks proper `<response>`
tags (common early in RL training), `solver_projection()` falls back to sending the
**entire raw decoded output** — including `<think>` reasoning blocks — as a response:

```
projection.py line 107:
    # No recognized tags → treat as raw response
    results.append({"type": "response", "content": action})  ← FULL 2048-token output
```

This full output (with `<think>` blocks) flows through `_handle_response(content)` →
`user_sim.respond(content)` → stored as a "user" role message in user sim history.

**Impact math (from screenshot data):**
- Without `<think>` leak: 54 history entries = **5,800 tokens**. Cannot reach 32k.
- With `<think>` leak: 42 entries = **30,600 tokens** → overflow. Matches screenshot.
- Screenshot reverse-engineers to agent messages averaging **1,300–1,640 tokens each**
  (consistent with full model output with think blocks, `max_response_length=2048`).

### Why it never recovered

The existing `respond()` method trimmed a local `history_view` copy on overflow but never
synced it back to `self.history`. So the next call to `respond()` started over from the
full untrimmed history, hit the limit again on first try, trimmed again locally, and
succeeded on retry — every single turn, doubling API latency permanently for that episode.

---

## Issue-by-Issue Details

### B1: Agent `<think>` Blocks Leak Into User Sim History — FIXED

**Problem:** When `solver_projection()` cannot find `<response>` tags in the agent's output
(common early in training), it falls back to treating the entire raw output as a response
(line 107). This raw output includes `<think>` reasoning blocks (500-1500 tokens). The
content flows unmodified through `_handle_response(content)` → `user_sim.respond(content)`
→ stored as a "user" role message in user sim history. Each agent message averages
1,300-1,640 tokens instead of 50-150, making history grow 10-15× faster than intended.

**Fix:** In `_handle_response()`, strip `<think>...</think>` blocks from `content` before
passing to `user_sim.respond()`. The full content is still recorded in `message_history`
for the tau2-bench evaluator — only the user sim copy is cleaned.

**Files:** `envs.py` (line 376-382: `re.sub()` before `user_sim.respond()`, line 10: added
`import re`)

**What is NOT changed:** The `message_history` used by the tau2 evaluator still gets the
full agent output. The projection logic is unchanged. The agent-side history in
`env_manager.py` already strips `<think>` (C3 fix from previous changelog).

---

### B2: Trimmed History Never Persisted Back — FIXED

**Problem:** `respond()` trimmed a local `history_view` on context overflow, but never
synced it back to `self.history`. On the next call, `respond()` rebuilt `history_view`
from the full, untrimmed `self.history`, guaranteeing another failure on first try.
This meant every user sim call past the overflow point wasted one full vLLM inference.

Additionally, `_consecutive_failures` was reset on every context-length error (line 249),
so even if trimming could never help (e.g., system prompt alone exceeds context), the
retry loop would run forever with no exit.

**Fix (two sub-changes):**

1. **Persist trim:** After the retry loop succeeds with a trimmed `history_view`, set
   `self.history = history_view`. The next call starts from the already-trimmed baseline.

2. **Retry cap:** Added `_MAX_CONTEXT_RETRIES = 5`. If context-length trimming fails
   5 times consecutively, the episode ends with `[API_FAIL]` instead of looping forever.

**Files:** `user_sim.py` (line 262: `trimmed` flag, line 276-283: retry cap, line 307:
`trimmed = True`, lines 320-323: persist trim, line 56: `_MAX_CONTEXT_RETRIES` constant)

---

### B3: No Proactive History Budget — FIXED

**Problem:** The user sim only discovered context overflow reactively — by making an API
call, getting a 400 error, parsing the overshoot, trimming, and retrying. Even with B1
fixed (smaller messages), very long episodes (20+ response turns) could still accumulate
enough history to overflow, wasting one API call per occurrence.

**Fix:** Added `_trim_history_to_budget()` method, called at the top of `respond()` before
the API call. Uses a conservative budget of 24,000 tokens (32,768 minus ~4k system prompt
minus ~4k safety buffer). Drops oldest 2 messages at a time (one user+assistant exchange)
to maintain conversation coherence. Token estimation uses the simple 1 token ≈ 4 chars
heuristic.

**Files:** `user_sim.py` (lines 238-250: new method, line 258: called in `respond()`,
lines 58-60: `_HISTORY_TOKEN_BUDGET` constant)

---

## Documentation Updates

### `algo.md`

- Updated Section 4.5 (Solver Step Dispatch): `_handle_response` now shows the
  `<think>`-stripping step between recording in `message_history` and calling user sim.
- Updated Section 8 (Critical Invariants): Added 3 new checklist items for user sim
  context management.

---

## Expected Impact After Fix

1. **B1 alone eliminates ~90% of overflows** — user sim messages drop from 1,300-1,640
   tokens to 50-200 tokens. A 30-step episode with 18 response turns stays well under
   10k tokens total, far from the 32k limit.

2. **B2 ensures no repeated failures** — if an overflow does occur (edge case), the trim
   is persisted and subsequent turns proceed without wasted API calls.

3. **B3 provides proactive prevention** — for very long episodes or very large system
   prompts, history is trimmed before the API call, so the reactive retry path is rarely
   needed.

4. **No reward regression** — the tau2 evaluator's `message_history` is unchanged (still
   gets full agent output). User sim quality may improve slightly since it no longer
   receives confusing `<think>` reasoning as part of the "customer" conversation.
