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
 && mkdir -p /var/lib/rsyslog /var/lib/docker-volumes /var/cache/s3fs \
 && rm -rf /var/lib/apt/lists/*
COPY rsyslog.conf /etc/rsyslog.conf
COPY fuse.conf /etc/fuse.conf
COPY --from=driver /s3fs-volume-plugin /
