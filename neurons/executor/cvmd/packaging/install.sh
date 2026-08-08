#!/usr/bin/env bash
#
# Install cvmd from an unpacked release tarball. Run as root from the unpack directory:
#
#     ./install.sh
#
# Idempotent: re-running upgrades the venv in place and leaves an existing config,
# authorized-clients file, and TLS pair untouched.
#
# The DAH-2544 Ansible role calls this after verifying the tarball's sha256.

set -euo pipefail

PREFIX=/opt/cvmd
VENV="${PREFIX}/venv"
CONFIG_DIR=/etc/cvmd
TLS_DIR="${CONFIG_DIR}/tls"
UNIT=/etc/systemd/system/cvmd.service

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
    echo "install.sh must run as root" >&2
    exit 1
fi

# cvmd needs >=3.13: bittensor 11.x is what installs on the CVM host's system Python, and the
# package declares that floor. Fail here with the actual version rather than inside pip.
python_version="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$(printf '%s\n3.13\n' "${python_version}" | sort -V | head -n1)" != "3.13" ]]; then
    echo "cvmd requires Python >= 3.13; this host has ${python_version}" >&2
    exit 1
fi

if ! python3 -c 'import venv' 2>/dev/null; then
    echo "python3 venv module is missing — install python3-venv" >&2
    exit 1
fi

echo "==> creating ${VENV} (python ${python_version})"
mkdir -p "${PREFIX}"
python3 -m venv --upgrade-deps "${VENV}"

# --require-hashes: every dependency is pinned by hash in requirements.lock, so a compromised
# index cannot substitute a package. The lock is generated from pdm.lock at build time.
echo "==> installing dependencies"
"${VENV}/bin/pip" install --disable-pip-version-check --no-input \
    --require-hashes -r "${here}/requirements.lock"

# --force-reinstall because pip resolves by VERSION, not by contents: a rebuilt wheel that keeps
# the same version string is "already satisfied" and silently not installed, so the venv keeps
# running the previous build while every layer above reports a successful upgrade. Measured on
# hardware during the DAH-2575 acceptance run. --no-deps keeps it to the one wheel, so the
# hash-pinned dependency install above is not redone.
echo "==> installing cvmd"
"${VENV}/bin/pip" install --disable-pip-version-check --no-input --no-deps --force-reinstall \
    "${here}"/cvmd-*.whl

echo "==> laying down ${CONFIG_DIR}"
mkdir -p "${CONFIG_DIR}" "${TLS_DIR}"
if [[ -f "${CONFIG_DIR}/config.toml" ]]; then
    echo "    config.toml already present — leaving it alone"
else
    install -m 0644 "${here}/config.toml" "${CONFIG_DIR}/config.toml"
fi

# The authorized-clients file is rendered by the Ansible role, not by this script. Without it
# cvmd refuses to start — which is the intended failure, not something to paper over with a
# default that authorizes nobody or, worse, a placeholder key.
if [[ ! -f "${CONFIG_DIR}/authorized_clients.json" ]]; then
    echo "    NOTE: ${CONFIG_DIR}/authorized_clients.json is absent; cvmd will refuse to start"
    echo "          until the Ansible role renders it."
fi

if [[ -f "${TLS_DIR}/cert.pem" && -f "${TLS_DIR}/key.pem" ]]; then
    echo "==> TLS pair already present — leaving it alone"
else
    echo "==> generating a self-signed TLS pair"
    openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
        -keyout "${TLS_DIR}/key.pem" -out "${TLS_DIR}/cert.pem" \
        -subj "/CN=$(hostname -f)" \
        -addext "subjectAltName=DNS:$(hostname -f)"
    chmod 0600 "${TLS_DIR}/key.pem"
fi

echo "==> installing the systemd unit"
install -m 0644 "${here}/cvmd.service" "${UNIT}"
systemctl daemon-reload
systemctl enable cvmd.service

echo
echo "cvmd installed. Start it with: systemctl start cvmd"
