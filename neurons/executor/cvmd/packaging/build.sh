#!/usr/bin/env bash
#
# Build the cvmd release tarball and print its sha256.
#
#     ./packaging/build.sh
#
# The printed sha256 is the value the DAH-2544 Ansible role takes as
# `lium_cvmd_package_sha256`; the tarball is what `lium_cvmd_package_url` must serve.
#
# A real release pipeline is later work — this is the reproducible local build.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "${here}")"
dist="${root}/dist"

version="$(cd "${root}" && python3 -c '
import tomllib, pathlib
print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])
')"

staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT

echo "==> building the wheel"
rm -rf "${dist}"
(cd "${root}" && pdm build --no-sdist --dest "${dist}")

echo "==> exporting the hash-pinned lock"
# Runtime dependencies only — pytest and friends have no business on a CVM host.
(cd "${root}" && pdm export --prod --format requirements --output "${staging}/requirements.lock")

echo "==> assembling the tarball"
cp "${dist}"/cvmd-*.whl "${staging}/"
cp "${here}/install.sh" "${here}/cvmd.service" "${here}/config.toml" "${staging}/"
chmod 0755 "${staging}/install.sh"

tarball="${dist}/cvmd-${version}.tar.gz"
# --sort=name and a fixed mtime so two builds of the same commit produce the same bytes, which
# is what makes the sha256 worth pinning in the first place.
tar --create --gzip --file "${tarball}" \
    --directory "${staging}" \
    --sort=name \
    --mtime="@0" \
    --owner=0 --group=0 --numeric-owner \
    .

echo
echo "tarball: ${tarball}"
echo "sha256:  $(sha256sum "${tarball}" | cut -d' ' -f1)"
