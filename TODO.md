# Petals Enhancement Roadmap

The following TODO list outlines steps to make Petals accessible to non-technical users for home use, including easy private swarms and compatibility with existing Chat UIs.

## 1. OpenAI-Compatible HTTP API (High Priority)
*   **Goal**: Enable users to use Petals with any existing Chat UI (e.g., SillyTavern, generic web frontends) by providing a local API server that mimics the OpenAI API.
*   **Implementation**:
    *   Create `src/petals/cli/run_http_server.py`.
    *   Use `FastAPI` and `Uvicorn` (add to `extras_require` in `setup.cfg`).
    *   Implement `/v1/models`, `/v1/chat/completions`, and `/v1/completions` endpoints.
    *   Map these endpoints to `petals.AutoDistributedModelForCausalLM` inference logic.
    *   Support streaming responses (SSE).

## 2. Interactive Configuration Wizard (High Priority)
*   **Goal**: Simplify the `run_server` process for non-technical users, removing the need to understand complex CLI flags.
*   **Implementation**:
    *   Create `petals configure` or an interactive mode for `petals.cli.run_server` (e.g., if no args provided).
    *   Ask simple questions: "Do you want to host a public or private model?", "Which model?", "Which GPU?".
    *   Auto-detect hardware (GPU VRAM) and suggest optimal block counts.
    *   Save configuration to `config.yml` (already supported by `configargparse` but needs a writer).

## 3. Simplified "Home Swarm" Launcher (Medium Priority)
*   **Goal**: Allow a user to easily set up a private swarm across their home devices without manually managing DHT vs Compute nodes.
*   **Implementation**:
    *   Add a `--mode home-coordinator` flag: Starts a DHT node + Compute Server and prints a "Join Code" (IP + multiaddr).
    *   Add a `--mode home-worker --join <CODE>` flag: Automatically connects to the coordinator without needing full multiaddrs.
    *   Handle local network discovery (MDNS) if possible.

## 4. Local Web Dashboard (Medium Priority)
*   **Goal**: Provide a visual interface for server status, replacing the scrolling terminal logs.
*   **Implementation**:
    *   Serve a simple HTML/JS dashboard from the server process (can be part of the HTTP API server or a separate thread).
    *   Visualize: Connection status, Block health, Inference throughput, GPU usage.

## 5. Native Installers & Helpers (Low Priority)
*   **Goal**: Remove the need for `pip` and Python environment management.
*   **Implementation**:
    *   Create one-click run scripts for Windows (`.bat`) and Mac (`.command`) that handle `venv` creation and `pip install`.
    *   Investigate `PyInstaller` builds for a standalone binary.

## 6. Network Diagnostics Tool (Low Priority)
*   **Goal**: Help users debug NAT/Firewall issues.
*   **Implementation**:
    *   Enhance `reachability.py` to provide specific instructions based on the failure type (e.g., "Open port 31337 on your router").
    *   Add a `petals check-network` command.
