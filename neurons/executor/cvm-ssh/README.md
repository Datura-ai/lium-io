# lium-cvm-ssh — SSH into the renter CVM itself (DAH-2684)

A renter of a Lium CVM logs in **to the CVM**: root on the guest OS, with the guest's
processes, network, filesystem, docker daemon and GPUs. Not into a container they had to
describe first.

The dstack production image (`dstack-nvidia-0.5.11`) makes that non-trivial: it ships no
sshd, removes every login path (`nologin`), and its rootfs is a read-only, dm-verity-checked
squashfs. So the SSH daemon lives in this container, and every session it opens is moved
into the guest before the renter's shell starts:

```
renter ── ssh -p <host port> root@<node ip> ──▶ host forward ──▶ guest :2200
                                                                    │
                                          this container: sshd authenticates the key
                                                                    │
                                          ChrootDirectory /host  (the guest's own root fs)
                                                                    │
                                          /run/lium-ssh/shell: nsenter -t 1 -m -u -i -n -p
                                                                    │
                                                            the guest's /bin/sh, as root
```

The backend injects this service into every renter compose (`services/cvm_compose.py`),
pinned by digest, with the renter's authorized keys in the same service block. The compose
is measured (`compose_hash` → RTMR3), so the quote covers **which sshd runs and whose keys
it accepts**, and the renter can check both against what they ordered.

## The service block the backend writes

```yaml
services:
  lium-ssh:
    image: ghcr.io/datura-ai/lium-cvm-ssh@sha256:…
    restart: always
    network_mode: host      # sshd binds guest :2200 — the port cvmd forwards and waits on
    pid: host               # PID 1 is the guest's init, the target of nsenter
    privileged: true        # setns into another process's namespaces
    environment:
      - LIUM_SSH_PORT=2200
      - "LIUM_SSH_AUTHORIZED_KEYS=ssh-ed25519 AAAA… renter@laptop\nssh-ed25519 AAAA… ci"
    volumes:
      - /:/host                        # the guest root: the chroot the session lives in
      - /dev/pts:/dev/pts              # the guest's terminals, so a login has a working tty
      - lium-cvm-ssh-hostkeys:/etc/ssh/hostkeys   # host keys survive restarts (fingerprint stays)
volumes:
  lium-cvm-ssh-hostkeys:
```

The entrypoint **refuses to start** — with the missing line named — if any of those is
absent. Each one is load-bearing, and an sshd missing one would still start and quietly
land sessions in the container instead of the guest, which is precisely the thing a
renter must never be handed under the name "your CVM".

| Env | Meaning |
|---|---|
| `LIUM_SSH_AUTHORIZED_KEYS` | One OpenSSH public key per line. Required; nothing else authorizes a login. |
| `LIUM_SSH_PORT` | Guest port to bind (default `2200`, cvmd's `cvm_ssh_guest_port`). |

Keys only: `PermitRootLogin prohibit-password`, `PasswordAuthentication no`, root has no
password, no other account exists. Port forwarding (`-L`/`-R`/`-D`) is on and terminates in
the guest's network namespace. X11 and layer-3 tunnels are off.

## What the renter gets, and what they do not

**On the guest:** root; `docker` (the guest's daemon, GPU-capable through the NVIDIA
container toolkit baked into the image), `nvidia-smi`, `systemctl`, `ip`, `curl`, `jq`,
busybox. Persistent storage is the encrypted data disk: `/dstack/persistent`, and docker
volumes/images under `/var/lib/docker`. `/etc`, `/usr`, `/bin` and `/home` are volatile
overlays — writable, reset on reboot.

**Not on the guest, by construction of the dstack image:** `bash`, `apt`, `rsync`, `scp`
(the server side). `sftp` and modern `scp` (which speaks SFTP) work — the session is
chrooted into the guest root, so they read and write the guest. Legacy `scp -O` and `rsync`
need a binary the guest does not have; the practical answer is `docker run` for tooling, or
`tar | ssh`.

The shell is the guest's busybox `ash`. That is a fact about the image the platform
attests, not a limitation of this service.

## Verifying it locally

```bash
bash tests/test_guest_ssh.sh
```

Runs the built image against *this* machine as the guest — the same compose shape — and
checks: each missing precondition is refused with its reason; a session's hostname, mount
namespace, uid and root are the host's; an interactive login gets a real tty; a quoted
remote command runs; a second listed key logs in; an unlisted key and password auth are
refused; scp writes the host filesystem; the host-key fingerprint survives a container
restart. No CVM needed: chroot + nsenter behave the same on any Linux host.

## Publishing

See `PUBLISHING.md`. Digest only, and a new digest is a catalog event — it changes the
compose hash of every rental derived after it.
