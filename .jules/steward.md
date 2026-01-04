## 2024-05-20 – Enforce strict tuple length in ServerInfo deserialization
**Issue:** The `ServerInfo.from_tuple` method accepted tuples of any length greater than or equal to 2, while the corresponding `to_tuple` serialization method always produced a 3-element tuple.
**Risk:** This ambiguity could lead to silent data loss or masking of version incompatibilities if a peer sends a malformed tuple with more than 3 elements.
**Resolution:** The `from_tuple` method was modified to strictly enforce that the input tuple must have a length of exactly 2 or 3. This makes the serialization and deserialization logic symmetric and ensures that any malformed data from peers will result in an explicit error.
