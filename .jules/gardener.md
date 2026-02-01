# Gardener's Journal

## 2025-02-01 – Silent Startups during Reachability Checks
**Learning:** The `validate_reachability` function, which is critical for nodes in the public swarm (especially those behind NAT), can block server startup for up to 7 minutes with no feedback to the operator after the initial "Detected a NAT" message.
**Impact:** This lack of observability leads to operator uncertainty, making it unclear if the server is hanging, crashed, or just waiting for libp2p relays to propagate. In a distributed system, silent waiting periods are a common source of user frustration and incorrect troubleshooting (e.g., premature restarts).
**Action:** Always provide periodic progress signals (e.g., every 60s) for long-running blocking operations at startup. Include elapsed and remaining time to set clear expectations for the operator.
