## 2025-01-30 - Periodic Progress Feedback for Long-Running CLI Operations
**Learning:** In CLI-based distributed systems, long blocking operations (like NAT reachability checks) can create "black holes" in observability, leading to user anxiety or premature termination. Periodic log messages (e.g., every minute) act as a functional equivalent to loading spinners in web UI.
**Action:** Always identify long-running loops with external dependencies and implement periodic progress logging with clear elapsed/remaining time indicators.
