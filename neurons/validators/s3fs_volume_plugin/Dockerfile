# The s3fs volume plugin the validator installs for every Volume (S3FS_PLUGIN_IMAGE): the upstream driver of
# marcelo-ochoa/docker-volume-plugins at a pinned commit, on Ubuntu 24.04's s3fs. The published mochoa build
# carries Ubuntu 20.04's s3fs 1.86, which resends a failed request without its headers (fixed in s3fs 1.88,
# s3fs-fuse commit 51f65d7): after a network blip the retry goes out unsigned, S3 refuses it and the renter's read
# fails with an I/O error. Built and pushed by .github/workflows/s3fs_volume_plugin.yml.
FROM golang:1.26-bookworm AS driver
ARG DRIVER_COMMIT=4921b1c14f2e3dcdd69a07b52fe276e15de12aab
WORKDIR /src
RUN git init -q . \
 && git fetch -q --depth 1 https://github.com/marcelo-ochoa/docker-volume-plugins.git "$DRIVER_COMMIT" \
 && git checkout -q FETCH_HEAD
# upstream builds with GOPATH `go get` at whatever its dependencies' heads are; these pin them by checksum
COPY go.mod go.sum ./
RUN CGO_ENABLED=0 go build -mod=readonly -trimpath -o /s3fs-volume-plugin ./s3fs-volume-plugin

FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends s3fs rsyslog tini ca-certificates \
 && mkdir -p /var/lib/rsyslog /var/lib/docker-volumes \
 && rm -rf /var/lib/apt/lists/*
# Without bucket_size, s3fs 1.93 reports a 4 GiB bucket to statfs, so `df` in a pod shows a 4 GB Volume (1.86 reported
# 256 TiB; fixed in 1.94). Set here, not in the validator's s3fs options: 1.86 refuses to mount with it, and pods
# rented on 1.86 remount with those options. The real binary keeps the name s3fs: the driver checks the mount's fstype.
RUN mkdir -p /usr/libexec/s3fs \
 && mv /usr/bin/s3fs /usr/libexec/s3fs/s3fs \
 && printf '#!/bin/sh\nexec /usr/libexec/s3fs/s3fs "$@" -o bucket_size=256TiB\n' > /usr/bin/s3fs \
 && chmod 755 /usr/bin/s3fs
COPY rsyslog.conf /etc/rsyslog.conf
COPY fuse.conf /etc/fuse.conf
COPY --from=driver /s3fs-volume-plugin /
