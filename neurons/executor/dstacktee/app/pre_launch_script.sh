
chmod +x /usr/bin/containerd-shim-runc-v2
# TODO: pin digest
docker run --rm --privileged --pid=host --net=host -v /:/host dstacktee/dstack-sysbox-installer:1.0.0@sha256:2f5dbea99176f3ea0362b85346b31b1160bfb70c1d98d1c8d375d57782127dd1
systemctl restart docker
