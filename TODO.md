# Petals Enhancement Roadmap

The following TODO list outlines steps to make Petals accessible to non-technical users for home use, including easy private swarms and compatibility with existing Chat UIs.

## 1. OpenAI-Compatible HTTP API (High Priority) [DONE]
*   **Goal**: Enable users to use Petals with any existing Chat UI (e.g., SillyTavern, generic web frontends) by providing a local API server that mimics the OpenAI API.
*   **Implementation**:
    *   [x] Create `src/petals/cli/run_http_server.py`.
    *   [x] Use `FastAPI` and `Uvicorn` (add to `extras_require` in `setup.cfg`).
    *   [x] Implement `/v1/models`, `/v1/chat/completions`, and `/v1/completions` endpoints.
    *   [x] Map these endpoints to `petals.AutoDistributedModelForCausalLM` inference logic.
    *   [x] Support streaming responses (SSE).

## 2. Interactive Configuration Wizard (High Priority) [DONE]
*   **Goal**: Simplify the `run_server` process for non-technical users, removing the need to understand complex CLI flags.
*   **Implementation**:
    *   [x] Create `petals configure` or an interactive mode for `petals.cli.run_server` (e.g., if no args provided).
    *   [x] Ask simple questions: "Do you want to host a public or private model?", "Which model?", "Which GPU?".
    *   [x] Auto-detect hardware (GPU VRAM) and suggest optimal block counts.
    *   [x] Save configuration to `config.yml` (already supported by `configargparse` but needs a writer).

## 3. Simplified "Home Swarm" Launcher (Medium Priority) [DONE]
*   **Goal**: Allow a user to easily set up a private swarm across their home devices without manually managing DHT vs Compute nodes.
*   **Implementation**:
    *   [x] Add a `--mode home-coordinator` flag: Starts a DHT node + Compute Server and prints a "Join Code" (IP + multiaddr).
    *   [x] Add a `--mode home-worker --join <CODE>` flag: Automatically connects to the coordinator without needing full multiaddrs.
    *   [x] Handle local network discovery (MDNS) if possible.

## 4. Local Web Dashboard (Medium Priority) [DONE]
*   **Goal**: Provide a visual interface for server status, replacing the scrolling terminal logs.
*   **Implementation**:
    *   [x] Create an API endpoint `/api/v1/status` in `src/petals/cli/run_http_server.py` returning basic server status (model name, device).
    *   [x] Serve a simple HTML/JS dashboard from the HTTP API server on the `/dashboard` path.
    *   [x] Fetch and visualize Connection status, Block health, Inference throughput, and GPU usage in the dashboard.

## 5. Native Installers & Helpers (Low Priority) [DONE]
*   **Goal**: Remove the need for `pip` and Python environment management.
*   **Implementation**:
    *   [x] Create one-click run scripts for Windows (`.bat`) and Mac (`.command`) that handle `venv` creation and `pip install`.
    *   [x] Investigate `PyInstaller` builds for a standalone binary.

## 6. Network Diagnostics Tool (Low Priority) [DONE]
*   **Goal**: Help users debug NAT/Firewall issues.
*   **Implementation**:
    *   [x] Enhance `reachability.py` to provide specific instructions based on the failure type (e.g., "Open port 31337 on your router").
    *   [x] Add a `petals check-network` command.

## 7. Code Refactoring (High Priority) [DONE]
*   **Goal**: Reduce code duplication across model blocks.
*   **Implementation**:
    *   [x] Extract `_reorder_cache_from_bloom` and `_reorder_cache_to_bloom` into a `ReorderCacheMixin` in `src/petals/models/block_utils.py`.
    *   [x] Update `Mixtral`, `DeepSeek`, and `Llama` blocks to use the mixin.

## 8. Deep P-tuning TODOs (High Priority) [DONE]
*   **Goal**: Fix the inline TODOs in deep ptuning logic.
*   **Implementation**:
    *   [x] Fix intermediate_prompt_embeddings logic in `src/petals/client/ptune.py`.
