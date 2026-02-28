## 2024-08-16 – Harden `ServerInfo` Deserialization

**Issue:** The `ServerInfo.from_tuple` method silently ignored extra elements if a tuple with more than 3 elements was provided. The deserialization logic was too permissive, allowing for ambiguous inputs.

**Risk:** This could lead to silent failures where malformed data from a peer (e.g., from a newer, incompatible client version) is partially processed without error. Such issues are difficult to debug and could lead to inconsistent state across the distributed system.

**Resolution:** The validation logic in `ServerInfo.from_tuple` was strengthened to enforce a strict tuple length of 2 or 3 elements. Any tuple not matching this format will now raise a `ValueError`, making deserialization failures explicit and immediate. This ensures that all peers adhere to a clear, unambiguous data structure format.