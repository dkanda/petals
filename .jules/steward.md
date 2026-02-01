## 2026-02-01 – Hardened UID parsing logic
**Issue:** `parse_uid` used `assert` for validation (which can be disabled in optimized mode), assumed only one dot in UIDs (preventing model names with dots), and didn't validate the index part robustness.
**Risk:** Malformed UIDs could lead to silent failures, incorrect routing, or crashes in production. Lack of support for dots in DHT prefixes limited model naming flexibility.
**Resolution:** `parse_uid` now uses explicit `ValueError` exceptions, employs `rsplit(..., 1)` to support dots in model names, and strictly validates that both prefix and index are non-empty and that the index is an integer.
