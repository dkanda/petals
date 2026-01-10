# Steward's Ledger

This ledger records non-obvious structural insights into the Petals codebase to aid future maintenance and hardening efforts.

---
## 2024-08-23 – Asymmetric ServerInfo Serialization
**Issue:** `ServerInfo.to_tuple()` always produces a 3-element tuple, but `ServerInfo.from_tuple()` accepts a 2-element tuple, silently creating a default `extra_info` dictionary.
**Risk:** Peers sending truncated data due to bugs or version mismatches could cause silent data loss. This leads to `ServerInfo` objects with missing attributes, risking hard-to-debug runtime errors (e.g., `AttributeError`) in downstream logic that assumes certain fields are present.
**Resolution:** The `from_tuple` deserialization method must enforce a strict 3-element tuple structure to match the serialization format, ensuring that malformed peer messages are rejected immediately.
