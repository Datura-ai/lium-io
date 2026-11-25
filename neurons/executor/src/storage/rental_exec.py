"""DAH-6447: runs restic as a process of the rental for encrypted backups and restores.

The privileged storage helper (host PID namespace) runs this file's source as
`python3 -I -S -c <source> <pid> <cwd> <check path or ""> <restic> <ca bundle> <restic arguments...>`.
It joins the rental's network, mount and user namespaces, never its PID namespace, so the rental can neither see
restic nor read its environment (the repository credentials). It roots itself in the rental's root, so every path,
symlinks included, resolves there. Nothing is loaded from the rental's filesystem: every import happens first,
restic is executed from a descriptor opened beforehand, and the CA bundle is written to a tmpfs that only this
process's private mount namespace has; restic's temp files go there too.
"""

import ctypes
import os
import sys

CLONE_NEWNS, CLONE_NEWUSER, CLONE_NEWNET = 0x00020000, 0x10000000, 0x40000000
MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REC, MS_PRIVATE = 2, 4, 8, 16384, 1 << 18
SOURCE_MISSING_EXIT = 66
# A rental cannot rename or unmount its /proc mount point, so this path keeps resolving to our tmpfs.
CA_DIRECTORY = "/proc"
# Go reads SSL_CERT_DIR as well as SSL_CERT_FILE; left unset it would trust CAs in the rental's /etc/ssl/certs.
# A directory of its own, so restic's temp packs beside it are never read as certificates.
CERT_DIRECTORY = CA_DIRECTORY + "/certs"
CA_PATH = CERT_DIRECTORY + "/lium-ca.crt"
# restic's temp packs go to this tmpfs, not the rental's /tmp: they must not fail on, or count against, the
# rental's full root disk. Same limit as the helper's own /tmp (restic.RESTIC_TMPFS_BYTES).
TMPFS_BYTES = 512 * 1024 * 1024


def check_source(path):
    """66: missing or empty, so nothing is uploaded (DAH-2002); 1: not a directory or unreadable."""
    try:
        return 0 if os.listdir(path) else SOURCE_MISSING_EXIT
    except FileNotFoundError:
        return SOURCE_MISSING_EXIT
    except OSError:
        return 1


def exec_restic(restic, cwd, check, restic_arguments):
    """Runs in the rental's root: the missing/empty check (inside the rental's user namespace, which the encrypted
    volume's FUSE mount requires), then restic by descriptor with PWD set so snapshot paths stay stable."""
    if check:
        code = check_source(check)
        if code:
            sys.exit(code)
    os.chdir(cwd)
    environment = {**os.environ, "PWD": cwd, "SSL_CERT_FILE": CA_PATH, "SSL_CERT_DIR": CERT_DIRECTORY, "TMPDIR": CA_DIRECTORY}
    os.execve(restic, ["restic", *restic_arguments], environment)


def join_namespaces(setns, namespaces):
    """Two passes, as nsenter does; the user namespace is last in `namespaces`. Privileged, it is joined last: joining
    it first drops the capability to join a network namespace an ancestor user namespace owns (Sysbox). Unprivileged,
    the others are refused until it is joined, so they are retried after it."""
    pending = []
    for namespace, flag in namespaces[:-1]:
        try:
            setns(namespace, flag)
        except PermissionError:
            pending.append((namespace, flag))
    for namespace, flag in [namespaces[-1], *pending]:
        setns(namespace, flag)


def _libc_call(libc, name, argtypes):
    function = getattr(libc, name)
    function.argtypes = argtypes  # mount's flags are an unsigned long

    def call(*arguments):
        if function(*arguments) != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"{name}: {os.strerror(error)}")

    return call


def main(argv):
    pid, cwd, check, restic_path, ca_path, *restic_arguments = argv
    libc = ctypes.CDLL(None, use_errno=True)
    setns = _libc_call(libc, "setns", [ctypes.c_int, ctypes.c_int])
    unshare = _libc_call(libc, "unshare", [ctypes.c_int])
    text = ctypes.c_char_p
    mount = _libc_call(libc, "mount", [text, text, text, ctypes.c_ulong, ctypes.c_void_p])
    restic = os.open(restic_path, os.O_RDONLY | os.O_CLOEXEC)
    with open(ca_path, "rb") as ca_file:
        ca = ca_file.read()
    root = os.open(f"/proc/{pid}/root", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    namespaces = [
        (os.open(f"/proc/{pid}/ns/{name}", os.O_RDONLY | os.O_CLOEXEC), flag)
        for name, flag in (("net", CLONE_NEWNET), ("mnt", CLONE_NEWNS), ("user", CLONE_NEWUSER))
    ]
    join_namespaces(setns, namespaces)
    # as nsenter does: become the rental's root, so restored files get the rental's ownership
    try:
        os.setgroups([])
    except PermissionError:
        pass  # setgroups is denied in this user namespace
    os.setgid(0)
    os.setuid(0)
    os.fchdir(root)
    os.chroot(".")
    unshare(CLONE_NEWNS)
    mount(None, b"/", None, MS_REC | MS_PRIVATE, None)
    options = f"size={TMPFS_BYTES},mode=0700".encode()
    mount(b"tmpfs", CA_DIRECTORY.encode(), b"tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC, options)
    os.mkdir(CERT_DIRECTORY, 0o700)
    with os.fdopen(os.open(CA_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as ca_file:
        ca_file.write(ca)
    exec_restic(restic, cwd, check, restic_arguments)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except OSError as error:
        os.write(2, f"rental exec: {error}\n".encode())
        os._exit(1)
