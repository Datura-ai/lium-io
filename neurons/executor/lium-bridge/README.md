# lium-bridge

Bridgectl — a set of scripts to manage the Chutes (K3s) stack on Lium GPU machines. Allows switching a GPU machine between Lium executor mode and Chutes mining mode.

## Installation

```bash
sudo ./install_chutes_bridge.sh
```

This creates `/opt/lium-bridge/`, copies scripts, sets up a restricted `lium-bridge` SSH user with `ForceCommand`, and configures sudoers.

## Usage

All commands return JSON on stdout and log to `/var/log/lium-bridge.log`.

```bash
bridgectl setup --validator-hotkey <ss58> --hotkey-ss58 <ss58> --hotkey-seed <hex>
bridgectl start    # Start K3s + Chutes
bridgectl stop     # Stop K3s + Chutes
bridgectl status   # Health check (read-only)
bridgectl uninstall  # Tear down K3s + Chutes, restore host to not_installed
```

## Current behavior

- `setup` installs K3s, GPU Operator, Chutes GPU charts, and leaves the host in `installed_stopped`
- `start` starts K3s, waits for node Ready (up to 2 min) and agent pod healthy (up to 3 min), then saves `state=running`
- `stop` deletes Chutes-managed workload pods with 60s grace, stops K3s, then saves `state=stopped`
- `status` is read-only and no longer treats `stopped` without a running executor as an error
- `uninstall` removes Helm releases when possible, runs `k3s-uninstall.sh`, cleans K3s paths, restarts Docker, and saves `state=not_installed`

Validated on H100 host `64.34.82.167`: repeated `start -> stop` cycles, plus idempotent `start` and `stop`.

## State machine

```
not_installed → [setup] → installed_stopped → [start] → running
                                  ↑              ↑          |
                                  |              +--- [stop] -+
                                  |                          |
                                  +------ [stop] ------------+
                                          (state: stopped)
```

State is persisted in `/opt/lium-bridge/state.json`.

## Files

| File | Description |
|------|-------------|
| `install_chutes_bridge.sh` | One-shot installer: creates dirs, users, SSH config, sudoers |
| `bin/bridgectl` | Dispatcher — parses command verb, delegates to the right script. Supports SSH `ForceCommand` |
| `bin/setup-chutes` | Installs K3s, Helm, GPU Operator, Chutes Helm charts, generates miner kubeconfig |
| `bin/start-chutes` | Starts K3s, waits for node Ready + agent pod healthy, saves state → `running` |
| `bin/stop-chutes` | Deletes chute workload pods (grace period 60s), stops K3s, saves state → `stopped` |
| `bin/status` | Read-only health check: K3s active, agent healthy, pod count, disk free; `stopped` without executor is valid |
| `bin/uninstall-chutes` | Removes Helm releases, uninstalls K3s, restarts Docker, saves state → `not_installed` |
| `bin/miner-rbac.yml` | K8s RBAC manifest for miner service account (used by Helm chart, not applied directly) |
