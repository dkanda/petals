# Gardener's Journal

## YYYY-MM-DD – [Title]
**Learning:** [What this codebase taught you]
**Impact:** [Why it matters at scale]
**Action:** [How future changes should adapt]

## 2026-01-31 – Improving Reachability Check Observability
**Learning:** In distributed P2P systems like Petals, long-running startup checks (like NAT/relay reachability) can take several minutes. Without periodic feedback, operators may incorrectly assume the process is hung and prematurely terminate the server.
**Impact:** Prevents operator confusion and reduces unnecessary server restarts during the libp2p relay connection phase.
**Action:** Always provide periodic progress updates (e.g., every 60s) and set clear expectations for the maximum wait time in long-running synchronous checks.
