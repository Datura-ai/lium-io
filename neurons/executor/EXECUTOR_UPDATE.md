# Executor updates

The standard stack (`docker-compose.yml` in this directory) updates itself. This page says how that works, how to make sure that a node is current, and what to run on a node that stopped updating.

## How a node updates itself

The stack runs two containers: `executor-executor-runner-1` and `executor-watchtower-1`.

1. At start, and then every 5 minutes, the updater (`executor-watchtower-1`, image `daturaai/lium-watchtower`, source `../../watchtower`) reads the digest of the current runner image from `https://lium.io/api/watchtower/digest`. A digest is the identity of an image: `sha256:` followed by 64 hex characters. The validator signs the response. The updater checks the signature.
2. When the runner container runs a different digest, the updater pulls `daturaai/compute-subnet-executor-runner@<digest>` and recreates the runner from it with the same name, labels, mounts and restart policy.
3. The runner starts the executor containers from the compose file that is baked into its image. That file pins the executor image by digest.

The updater never pulls by tag. A pull by tag (`runner:latest`) asks the host's Docker daemon, and the daemon asks a registry mirror first when `/etc/docker/daemon.json` has one. A mirror can keep an old copy of the tag, and the pull then "succeeds" with the old image. A pull by digest is content-addressed: the daemon checks the hash of what it receives, so a mirror can answer with the right image or with an error, never with an old one. When the daemon's path fails, the updater pulls the same digest from `registry-1.docker.io` directly, which the daemon's mirror setting does not cover.

Stacks started before this change run `nickfedor/watchtower`, which pulls by tag. Those hosts keep it until the one-time restart below.

## Make sure that a node is current

Ask the node. The executor answers on the host port `EXTERNAL_PORT` from `.env`. Run this in the directory that holds `.env`:

```bash
curl -s "http://127.0.0.1:$(grep -E '^EXTERNAL_PORT=' .env | cut -d= -f2)/update-status" | python3 -m json.tool
```

`runner.running_digest` is the digest of the runner image the host runs. `runner.expected_digest` is the digest the validator signed. `runner.update_pending` is `true` when they differ, `false` when they match, and `null` when one of them is unknown (`runner.error` says which). `executor.running_digest` is the digest of the executor image itself. The validator compares that one to the current release.

Without the executor running, read the two digests by hand:

```bash
curl -s https://lium.io/api/watchtower/digest | python3 -c 'import sys, json; print(json.load(sys.stdin)["digest"])'
docker image inspect --format '{{index .RepoDigests 0}}' "$(docker inspect executor-executor-runner-1 --format '{{.Image}}')" | cut -d@ -f2
```

If the two digests are equal, the node runs the current runner. If they differ, the node is behind: follow the next section.

## One-time restart of a node that stopped updating

If the node is behind, run this on the host. A `lium mine` installation puts the checkout at `compute-subnet` under the directory where `lium mine` was run, usually the home directory. For a manual installation, replace the first path with the directory that holds `neurons/executor/docker-compose.yml`:

```bash
cd ~/compute-subnet/neurons/executor && git pull && docker compose pull watchtower && docker compose up -d watchtower
```

What the command does:

1. `git pull` brings the current compose file with the digest-based updater. The checkout must be on the `main` branch: `git branch --show-current` prints `main`. If it prints another name, run `git checkout main` first.
2. `docker compose pull watchtower` fetches the updater image and nothing else. A pull of the runner tag would return the mirror's old copy again, so the command does not ask for it.
3. `docker compose up -d watchtower` replaces `nickfedor/watchtower` with the updater and does not touch the runner. The updater checks at start, pulls the current runner by digest and recreates the runner. The new runner then pulls the current executor image and recreates `executor-executor-1`.

Pod containers are not part of the compose project, so the command does not touch them. The executor container restarts once, the pods keep running.

A later `docker compose up -d` without a service name recreates the runner from the compose file too, because the runner's service definition changed. That is harmless: the updater brings the runner back to the signed digest at its next check.

If `git pull` stops because of local changes to tracked files, run `git stash && git pull && git stash pop`. Then run `docker compose pull watchtower && docker compose up -d watchtower`. If `git stash pop` reports a conflict, send the output to Lium support.

After about two minutes, make sure that both containers are up:

```bash
docker compose ps executor-runner watchtower
```

The output must list `executor-executor-runner-1` and `executor-watchtower-1` with the state `Up`. Then run the `/update-status` command from the section above: `runner.update_pending` must be `false`. The node earns again at the next validation cycle after the executor image matches the current release.

## If `update_pending` stays `true` after the restart

Read the updater's log:

```bash
docker logs executor-watchtower-1 --tail 50
```

A line `Pull by digest failed through the daemon's registry path, retrying from Docker Hub directly` means the mirror could not serve the digest and the updater went to Docker Hub itself. The next lines say whether that worked. A line `Failed to fetch digest from endpoint` means the host cannot reach `lium.io`. Make sure that the outbound firewall allows `lium.io`. A line `Signature verification failed` means the clock is more than 10 minutes off, or the response did not come from the validator. Make sure that `date -u` is correct. A line `Runner container not identified this cycle, nothing pulled` means two containers carry the runner label. That happens for a moment during `docker compose up -d` and clears at the next check. If it stays, `docker ps -a --filter label=com.docker.compose.service=executor-runner` lists both. If the log does not say, send the 50 lines to Lium support with the node address.

Do not restart the Docker daemon while the node has a rented pod. A daemon restart restarts every container on the host, pods included.
